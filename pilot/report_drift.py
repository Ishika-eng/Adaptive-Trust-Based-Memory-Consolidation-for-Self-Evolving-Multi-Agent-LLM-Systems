"""Per-phase results of the rule-change stream.

    pilot/.venv/bin/python pilot/report_drift.py --dir pilot/results/local_drift --exp drift

Phase A = the old rule, phase B = the opposite rule asked about the SAME questions. A memory that serves stale notes does worse in
phase B than a model with no memory at all; a good one recovers. Prints pass rate per phase and per block of phase B, with paired
comparisons on the TRUE outcome.
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ROOT  # noqa: E402
from report import load_runs, paired_comparison, paired_outcomes  # noqa: E402


def phase(task_id: str) -> str:
    return "B" if task_id.startswith("DRIFT") else "A"


def rate(steps, pred):
    xs = [s["correct"] for s in steps if pred(s)]
    return sum(xs) / len(xs) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--exp", default="drift")
    a = ap.parse_args()
    d = Path(a.dir) if Path(a.dir).is_absolute() else ROOT / a.dir
    runs = load_runs(a.exp, "code", d)
    conds = list(runs)
    print(f"{'condition':14s} seeds | phase A (old rule) | phase B (new rule) | B first third | B middle | B last third")
    for c in conds:
        rs = runs[c]
        def m(f):
            xs = [f(r["steps"]) for r in rs]; xs = [x for x in xs if x is not None]
            return st.mean(xs) if xs else float("nan")
        n = lambda steps: [s for s in steps if phase(s["task_id"]) == "B"]
        third = lambda k: (lambda steps: (lambda b: rate(b[len(b) * k // 3:len(b) * (k + 1) // 3], lambda s: True))(n(steps)))
        print(f"{c:14s} {len(rs):4d} | {m(lambda s: rate(s, lambda x: phase(x['task_id']) == 'A')):.3f}             | "
              f"{m(lambda s: rate(s, lambda x: phase(x['task_id']) == 'B')):.3f}              | {m(third(0)):.3f}         | {m(third(1)):.3f}    | {m(third(2)):.3f}")
    print("\npaired comparisons on phase B only (B minus A, 95% CI, exact McNemar):")
    def only_b(rs):
        return [{**r, "steps": [s for s in r["steps"] if phase(s["task_id"]) == "B"]} for r in rs]
    for A, B in [(x, y) for x in conds for y in conds if x != y and conds.index(x) < conds.index(y)]:
        p = paired_comparison(paired_outcomes(only_b(runs[A]), only_b(runs[B])))
        if p.get("n"):
            print(f"  {A:12s} -> {B:12s} {p['diff']:+.3f}  [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]  p={p['p_mcnemar']:.3f}  n={p['n']}")


if __name__ == "__main__":
    main()
