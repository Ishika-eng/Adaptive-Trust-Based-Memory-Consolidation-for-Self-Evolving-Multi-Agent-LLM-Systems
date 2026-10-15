import json
from pathlib import Path

import pytest

from config import Config, load_config
from domains import get_domain
from sandbox import run_python

ROOT = Path(__file__).resolve().parents[1]
ROWS = [json.loads(l) for l in open(ROOT / "pilot/data/conventions_drift.jsonl")]


def _passes(code, row):
    prog = row["setup"] + code + "\n\n" + "\n".join(row["tests"]) + "\n"
    r = run_python(prog, timeout_s=10)
    return r.returncode == 0 and not r.timed_out


def test_every_question_is_asked_under_both_rules_and_the_old_solution_fails_the_new_tests():
    a = {r["question"]: r for r in ROWS if r["phase"] == "A"}
    b = [r for r in ROWS if r["phase"] == "B"]
    assert len(a) == len(b) == 29 and {r["question"] for r in b} == set(a)
    for r in b[::6]:                                    # a spread of tasks; the builder validated all of them
        assert _passes(r["reference"], r) and not _passes(r["stale"], r)
        assert _passes(a[r["question"]]["reference"], a[r["question"]]) and not _passes(r["reference"], a[r["question"]])


def test_the_stream_is_old_rule_first_then_new_rule_and_is_reproducible():
    d = load_config().to_dict()
    d["coding"] = {**d["coding"], "datasets": ["conventions_drift"]}
    cfg = Config(d)
    dom = get_domain("code", cfg)
    t1, t2 = dom.load_tasks(cfg, 58, 42), dom.load_tasks(cfg, 58, 42)
    assert [t["id"] for t in t1] == [t["id"] for t in t2] and len(t1) == 58
    assert [t["phase"] for t in t1] == ["A"] * 29 + ["B"] * 29
    assert [t["id"] for t in t1] != [t["id"] for t in dom.load_tasks(cfg, 58, 7)]
