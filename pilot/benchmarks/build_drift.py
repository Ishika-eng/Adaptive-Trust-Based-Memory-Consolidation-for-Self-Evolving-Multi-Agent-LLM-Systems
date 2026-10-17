"""Builds pilot/data/conventions_drift.jsonl -- the RULE-CHANGE ("drift") stream.

    pilot/.venv/bin/python pilot/benchmarks/build_drift.py

A team changes a convention. For three families the old rule (phase A) is replaced by its opposite (phase B):
  dates_dmy        DD-MM-YYYY dates            ->  YYYY-MM-DD dates
  sorted_unique    distinct items sorted       ->  distinct items in first-seen order
  result_tuple     (ok, value) tuples          ->  None on failure
The SAME questions are asked in both phases, so every note an assistant saved in phase A about a question is now WRONG for it.
The phase-B hidden tests are produced by running the opposite-convention solution (the "naive" solution written for
build_conventions.py, which already implements the other rule) on the same inputs; every task is validated by running it:
the phase-B solution must pass the phase-B tests and the old phase-A solution must FAIL them.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_conventions as bc  # noqa: E402
from sandbox import run_python  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data" / "conventions_drift.jsonl"
OUT2 = Path(__file__).resolve().parents[1] / "data" / "conventions_drift2.jsonl"
OUT3 = Path(__file__).resolve().parents[1] / "data" / "conventions_drift3.jsonl"
OUT4 = Path(__file__).resolve().parents[1] / "data" / "conventions_drift4.jsonl"
SETUP_B = bc.SETUP + '''

def _raises(label, fn, exc_name):
    try:
        fn()
    except BaseException as e:
        assert type(e).__name__ == exc_name, f"{label}: expected {exc_name} to be raised, got {type(e).__name__}"
        return
    raise AssertionError(f"{label}: expected {exc_name} to be raised, but nothing was raised")
'''
RULE_B2 = {
    "username_norm": "Usernames are case-sensitive: they are used exactly as typed. (The team changed this from lower-cased, stripped names.)",
    "none_on_invalid": "Functions raise the natural Python exception on bad input and never return None for it. (The team changed this from returning None.)",
}
RULE_B = {
    "dates_dmy": "Dates are strings in YYYY-MM-DD format (for example '2024-03-05'); functions that take or return a date use it. "
                 "(The team changed this from DD-MM-YYYY.)",
    "sorted_unique": "Functions that return a collection of distinct items return a list in the order each item first appears, "
                     "never sorted. (The team changed this from ascending order.)",
    "result_tuple": "Functions that can fail return None on failure and the plain result on success; they never raise and never "
                    "return a (ok, value) tuple. (The team changed this from tuples.)",
}
SKIP = {"index_result"}          # its opposite-convention solution returns -1, not a clean None


def _date_args(args: str) -> str:
    return re.sub(r"'(\d{2})-(\d{2})-(\d{4})'", r"'\3-\2-\1'", args)


def _expected(code: str, name: str, arg_list: list[str]):
    prog = code + "\n" + "\n".join(f"try:\n    print(repr({name}({a})))\nexcept Exception:\n    print('__ERR__')" for a in arg_list) + "\n"
    res = run_python(prog, timeout_s=10)
    if res.returncode != 0:
        return None
    outs = res.stdout.strip().splitlines()
    if len(outs) != len(arg_list):
        return None
    vals = []
    for o in outs:
        if o == "__ERR__":
            vals.append(("__ERR__",))
            continue
        try:
            vals.append(ast.literal_eval(o))
        except Exception:
            vals.append(("__ERR__",))
    return vals


def _expected2(code: str, name: str, arg_list: list[str]):
    """Like _expected, but a call that raises is recorded as ("__RAISES__", ExceptionName) instead of being dropped."""
    prog = code + "\n" + "\n".join(f"try:\n    print(repr({name}({a})))\nexcept BaseException as _e:\n    print(repr(('__RAISES__', type(_e).__name__)))" for a in arg_list) + "\n"
    res = run_python(prog, timeout_s=10)
    if res.returncode != 0:
        return None
    outs = res.stdout.strip().splitlines()
    if len(outs) != len(arg_list):
        return None
    try:
        return [ast.literal_eval(o) for o in outs]
    except Exception:
        return None


def _tests_b(name: str, cases: list) -> list[str]:
    out = []
    for a, v in cases:
        label = f"{name}({a})"
        if isinstance(v, tuple) and len(v) == 2 and v[0] == "__RAISES__":
            out.append(f"_raises({label!r}, lambda: {name}({a}), {v[1]!r})")
        else:
            out.append(f"_chk({label!r}, {name}({a}), {v!r})")
    return out


def _run_b(code: str, tests: list[str]) -> bool:
    res = run_python(SETUP_B + code + "\n\n" + "\n".join(tests) + "\n", timeout_s=10)
    return res.returncode == 0 and not res.timed_out


def build2() -> list[dict]:
    """Second stream (different families), where a changed rule can also mean 'now raises instead of returning None'."""
    a_rows = {r["id"]: r for r in (json.loads(l) for l in open(bc.OUT))}
    rows, problems = [], []
    for family, rule_b in RULE_B2.items():
        rule_a, specs = bc.FAMILIES[family]
        for i, (name, question, ref_a, ref_b, cases) in enumerate(specs, 1):
            args = [a for a, _ in cases]
            vals = _expected2(ref_b, name, args)
            if vals is None:
                problems.append(f"{family}/{name}: the phase-B solution failed to run"); continue
            tests = _tests_b(name, list(zip(args, vals)))
            if not _run_b(ref_b, tests):
                problems.append(f"{family}/{name}: phase-B solution fails its own tests"); continue
            if _run_b(ref_a, tests):
                continue
            a = a_rows[f"CONV/{family}/{i:02d}"]
            rows.append({**{k: a[k] for k in ("id", "family", "question", "entry_point", "tests", "setup", "reference")},
                         "dataset": "conventions_drift2", "phase": "A", "role": "rule", "rule": rule_a})
            rows.append({"id": f"DRIFT2/{family}/{i:02d}", "dataset": "conventions_drift2", "family": family, "phase": "B", "role": "rule",
                         "rule": rule_b, "question": question, "entry_point": name, "tests": tests, "setup": SETUP_B,
                         "reference": ref_b, "stale": ref_a})
    if problems:
        raise SystemExit("drift2 stream rejected:\n  " + "\n  ".join(problems))
    return rows


def build() -> list[dict]:
    a_rows = {r["id"]: r for r in (json.loads(l) for l in open(bc.OUT))}
    rows, problems, kept = [], [], 0
    for family, rule_b in RULE_B.items():
        rule_a, specs = bc.FAMILIES[family]
        for i, (name, question, ref_a, ref_b, cases) in enumerate(specs, 1):
            if name in SKIP:
                continue
            args = [_date_args(a) if family == "dates_dmy" else a for a, _ in cases]
            vals = _expected(ref_b, name, args)
            if vals is None:
                problems.append(f"{family}/{name}: the phase-B solution failed to run"); continue
            good = [(a, v) for a, v in zip(args, vals) if v != ("__ERR__",)]
            if len(good) < 2:
                problems.append(f"{family}/{name}: fewer than 2 usable phase-B cases"); continue
            tests = bc._tests(name, good)
            if not bc._run(ref_b, tests):
                problems.append(f"{family}/{name}: phase-B solution fails its own tests")
            if bc._run(ref_a, tests):
                continue                                   # the old solution still passes: nothing changed for this task, drop it
            a = a_rows[f"CONV/{family}/{i:02d}"]
            rows.append({**{k: a[k] for k in ("id", "family", "question", "entry_point", "tests", "setup", "reference")},
                         "dataset": "conventions_drift", "phase": "A", "role": "rule", "rule": rule_a})
            rows.append({"id": f"DRIFT/{family}/{i:02d}", "dataset": "conventions_drift", "family": family, "phase": "B", "role": "rule",
                         "rule": rule_b, "question": question, "entry_point": name, "tests": tests, "setup": bc.SETUP,
                         "reference": ref_b, "stale": ref_a})
            kept += 1
    if problems:
        raise SystemExit("drift stream rejected:\n  " + "\n  ".join(problems))
    return rows


def _pairs(rows: list[dict]) -> list[tuple[dict, dict]]:
    """(phase-A row, phase-B row) per question, in file order (the builders append A then B for each question)."""
    return [(rows[i], rows[i + 1]) for i in range(0, len(rows), 2)]


def build3(rows1: list[dict], rows2: list[dict]) -> list[dict]:
    """Recurring stream: every validated question of both drift streams asked under the old rule (A), the new rule (B), and the OLD
    rule again (A2, identical tests to A, new ids). Tests whether notes demoted in B come back when the rule reverts."""
    out = []
    for a, b in _pairs(rows1) + _pairs(rows2):
        out += [{**a, "dataset": "conventions_drift3"}, {**b, "dataset": "conventions_drift3"},
                {**a, "dataset": "conventions_drift3", "id": a["id"] + "/A2", "phase": "A2"}]
    return out


def build4(rows1: list[dict], rows2: list[dict]) -> list[dict]:
    """Transfer stream: half of the questions (even index inside their family) are asked under the old rule, the OTHER half
    (odd index) under the new rule, so no stored solution is for the same function as a phase-B question."""
    out, seen = [], {}
    for a, b in _pairs(rows1) + _pairs(rows2):
        k = seen.get(a["family"], 0)
        seen[a["family"]] = k + 1
        out.append({**a, "dataset": "conventions_drift4"} if k % 2 == 0 else {**b, "dataset": "conventions_drift4"})
    return out


if __name__ == "__main__":
    rows = build()
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
    from collections import Counter
    c = Counter((r["family"], r["phase"]) for r in rows)
    print(f"wrote {len(rows)} validated tasks ({len(rows) // 2} questions, each asked under the old and the new rule) to {OUT}")
    print({f"{k[0]}/{k[1]}": v for k, v in sorted(c.items())})
    rows2 = build2()
    OUT2.write_text("".join(json.dumps(r) + "\n" for r in rows2))
    c2 = Counter((r["family"], r["phase"]) for r in rows2)
    print(f"wrote {len(rows2)} validated tasks ({len(rows2) // 2} questions) to {OUT2}")
    print({f"{k[0]}/{k[1]}": v for k, v in sorted(c2.items())})
    OUT3.write_text("".join(json.dumps(r) + "\n" for r in build3(rows, rows2)))
    rows4 = build4(rows, rows2)
    OUT4.write_text("".join(json.dumps(r) + "\n" for r in rows4))
    c4 = Counter((r["family"], r["phase"]) for r in rows4)
    print(f"wrote conventions_drift3 ({3 * (len(rows) + len(rows2)) // 2} tasks) and conventions_drift4 ({len(rows4)} tasks)")
    print({f"{k[0]}/{k[1]}": v for k, v in sorted(c4.items())})
