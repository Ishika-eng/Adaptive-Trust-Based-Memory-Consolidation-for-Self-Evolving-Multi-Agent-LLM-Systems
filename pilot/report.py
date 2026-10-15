"""Turns saved runs into a table with statistics.

    python3 pilot/report.py baselines --domain code
    python3 pilot/report.py poison --domain math --ref static

Per condition: mean +/- sd across seeds of each metric. Against a reference
condition: a *paired* comparison -- the same task in the same seed under both
conditions -- giving the pass-rate difference with a bootstrap 95% CI and an
exact McNemar test on the discordant tasks. Conditions that choose their own
tasks (curriculum) cannot be paired in-run; use their held-out score instead,
which is paired because every condition sees the same unseen tasks.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ROOT

EVAL_DIR = ROOT / "pilot" / "results" / "eval"


def load_runs(exp: str, domain: str, eval_dir: Path | None = None) -> dict[str, list[dict]]:
    runs = defaultdict(list)
    for f in sorted((eval_dir or EVAL_DIR).glob(f"{exp}__{domain}__*.json")):
        d = json.loads(f.read_text())
        runs[d["condition"]].append(d)
    return runs


def mean_sd(xs: list[float]) -> tuple[float | None, float | None]:
    """None (JSON null) when there is no data -- NaN is not valid JSON."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact p-value: do the discordant pairs split 50/50?"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired_outcomes(a_runs: list[dict], b_runs: list[dict], key: str = "steps") -> list[tuple[bool, bool]]:
    """(outcome under A, outcome under B) for every (seed, task) both conditions attempted."""
    def index(runs):
        return {(r["seed"], s["task_id"]): s["correct"] for r in runs for s in r.get(key, [])}
    ia, ib = index(a_runs), index(b_runs)
    return [(ia[k], ib[k]) for k in ia if k in ib]


def paired_comparison(pairs: list[tuple[bool, bool]], boots: int = 10000, seed: int = 0) -> dict:
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    diffs = [int(b) - int(a) for a, b in pairs]          # +1: B right where A wrong
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(boots))
    b_only = sum(1 for a, b in pairs if b and not a)
    a_only = sum(1 for a, b in pairs if a and not b)
    return {
        "n": n, "diff": sum(diffs) / n,
        "ci95": [means[int(0.025 * boots)], means[int(0.975 * boots) - 1]],
        "b_better": b_only, "a_better": a_only, "p_mcnemar": mcnemar_exact(a_only, b_only),
    }


METRICS = [
    ("pass_rate", "pass rate", "{:.3f}"),
    ("heldout_pass_rate", "held-out", "{:.3f}"),
    ("poison_exposure_after_warmup", "poison in ctx*", "{:.1f}"),
    ("poison_used", "poison used", "{:.1f}"),
    ("cite_listed_frac", "cites notes", "{:.2f}"),
    ("cite_missing_frac", "USED missing", "{:.2f}"),
    ("trust_auc", "trust AUC", "{:.3f}"),
    ("poison_trust_final", "poison trust", "{:.2f}"),
    ("genuine_trust_final", "genuine trust", "{:.2f}"),
    ("final_memory_size", "memory", "{:.0f}"),
    ("tokens_in", "tokens in/task", "{:.0f}"),
    ("mean_retrieval_ms", "retrieval ms", "{:.2f}"),
]


def build_report(exp: str, domain: str, ref: str | None = None, eval_dir: Path | None = None) -> dict:
    runs = load_runs(exp, domain, eval_dir)
    if not runs:
        return {"experiment": exp, "domain": domain, "conditions": [], "comparisons": []}
    ref = ref or next(iter(runs))
    rows = []
    for cname, rs in runs.items():
        row = {"condition": cname, "seeds": sorted(r["seed"] for r in rs), "n": rs[0]["n"]}
        for key, _, _ in METRICS:
            vals = [r["summary"].get(key) for r in rs]
            if key == "tokens_in":
                vals = [v / r["n"] if v is not None else None for v, r in zip(vals, rs)]
            row[key] = dict(zip(("mean", "sd"), mean_sd(vals)))
        rows.append(row)
    comps = []
    for cname, rs in runs.items():
        if cname == ref or ref not in runs:
            continue
        comp = {"condition": cname, "vs": ref, "in_run": paired_comparison(paired_outcomes(runs[ref], rs))}
        if any(r.get("heldout_steps") for r in rs):
            comp["heldout"] = paired_comparison(paired_outcomes(runs[ref], rs, "heldout_steps"))
        comps.append(comp)
    return {"experiment": exp, "domain": domain, "reference": ref, "warmup_note":
            "* poison counted after the first top_k tasks (before that no genuine memory can exist)",
            "conditions": rows, "comparisons": comps}


def render(rep: dict) -> str:
    if not rep["conditions"]:
        return "no runs found"
    out = [f"## {rep['experiment']} / {rep['domain']}   (reference: {rep['reference']})", ""]
    head = ["condition", "seeds"] + [label for _, label, _ in METRICS]
    out += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rep["conditions"]:
        cells = [r["condition"], str(len(r["seeds"]))]
        for key, _, fmt in METRICS:
            m = r[key]
            cells.append("-" if m["mean"] is None else f"{fmt.format(m['mean'])} ± {fmt.format(m['sd'])}")
        out.append("| " + " | ".join(cells) + " |")
    out += ["", rep["warmup_note"], ""]
    for c in rep["comparisons"]:
        for label, k in (("in-run", "in_run"), ("held-out", "heldout")):
            p = c.get(k)
            if p and p["n"]:
                out.append(f"- {c['condition']} vs {c['vs']} ({label}, {p['n']} paired tasks): "
                           f"diff {p['diff']:+.3f}  95% CI [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]  "
                           f"better/worse {p['b_better']}/{p['a_better']}  McNemar p={p['p_mcnemar']:.3f}")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--domain", default="math")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--dir", default=None, help="results folder (default pilot/results/eval)")
    a = ap.parse_args()
    print(render(build_report(a.experiment, a.domain, a.ref, Path(a.dir) if a.dir else None)))
