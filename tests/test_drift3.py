import json
from pathlib import Path

import numpy as np

from config import Config, load_config
from domains import get_domain
from experiment import hygiene_audit
from sandbox import run_python
from stores import StaticMemoryStore

ROOT = Path(__file__).resolve().parents[1]


def _tasks(name, n, seed=1):
    d = load_config().to_dict()
    d["coding"] = {**d["coding"], "datasets": [name]}
    cfg = Config(d)
    return get_domain("code", cfg), cfg, get_domain("code", cfg).load_tasks(cfg, n, seed)


def _passes(code, row):
    r = run_python(row["setup"] + code + "\n\n" + "\n".join(row["tests"]) + "\n", timeout_s=10)
    return r.returncode == 0 and not r.timed_out


def test_recurring_stream_is_old_new_old_with_unique_ids_and_the_same_tests_in_both_old_phases():
    _, _, t = _tasks("conventions_drift3", 147)
    assert [x["phase"] for x in t] == ["A"] * 49 + ["B"] * 49 + ["A2"] * 49
    assert len({x["id"] for x in t}) == 147
    a = {x["question"]: x for x in t if x["phase"] == "A"}
    for x in t:
        if x["phase"] == "A2":
            assert x["tests"] == a[x["question"]]["tests"] and x["id"].endswith("/A2")
    for x in [y for y in t if y["phase"] == "B"][::10]:          # the builder validated all; spot-check by running
        assert _passes(x["reference"], x) and not _passes(a[x["question"]]["reference"], x)


def test_transfer_stream_never_asks_the_same_function_in_both_phases():
    _, _, t = _tasks("conventions_drift4", 49)
    assert [x["phase"] for x in t].count("A") == 25 and [x["phase"] for x in t].count("B") == 24
    ea = {x["entry_point"] for x in t if x["phase"] == "A"}
    eb = {x["entry_point"] for x in t if x["phase"] == "B"}
    assert not ea & eb


def _verified(fn, ans):
    return f"Verified solution (task):\n```python\ndef {fn}(x):\n    return {ans}\n```\nOutcome: CORRECT"


def test_last_write_wins_deletes_older_notes_about_the_same_function_only():
    e = np.zeros(3)
    s = StaticMemoryStore(last_write_wins=True)
    s.add(_verified("fmt_date", 1), e, True)
    s.add(_verified("other_fn", 2), e, True)
    s.add(_verified("fmt_date", 3), e, True)
    assert [m.text.count("fmt_date") > 0 for m in s.items] == [False, True] or len(s.items) == 2
    assert sum("fmt_date(" in m.text for m in s.items) == 1 and any("other_fn(" in m.text for m in s.items)
    plain = StaticMemoryStore()
    plain.add(_verified("f", 1), e, True); plain.add(_verified("f", 2), e, True)
    assert plain.size() == 2


def test_hygiene_audit_flags_a_solution_that_fails_now_as_wrongly_shown():
    dom, _, t = _tasks("conventions_drift", 58)
    b = next(x for x in t if x["phase"] == "B")
    s = StaticMemoryStore()
    s.add(f"Verified solution (x):\n{b['reference']}\nOutcome: CORRECT", np.zeros(3), True)
    a = next(x for x in t if x["phase"] == "A" and x["question"] == b["question"])
    s.add(f"Verified solution (x):\n{a['reference']}\nOutcome: CORRECT", np.zeros(3), True)
    r = hygiene_audit(s, dom, [b])
    assert r["audited"] == 2 and r["wrongly_shown"] == 1 and r["wrongly_hidden"] == 0
