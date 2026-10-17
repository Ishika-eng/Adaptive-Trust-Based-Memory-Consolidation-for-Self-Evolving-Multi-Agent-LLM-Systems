"""Analysis for docs/experiments/DRIFT_V3.md (rules fixed before any run).

    pilot/.venv/bin/python pilot/report_drift3.py pilot/results/local_drift_v3 [pilot/results/local_drift_v3_transfer]

First-attempt pass by phase, paired comparisons (95% bootstrap CI, exact McNemar), rules R1-R4, the hygiene audit, a cost table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report import paired_comparison  # noqa: E402

ORDER = ["no_memory", "static_rm", "static_rv", "static_lww", "atmc_v6", "rule_mem"]


def load(d: Path) -> dict:
    runs = {}
    for f in sorted(d.glob("*.json")):
        r = json.load(open(f))
        runs.setdefault(r["condition"], []).append(r)
    return runs


def outcomes(runs, cond, phases):
    return {(r["seed"], s["task_id"]): s["correct"] for r in runs[cond] for s in r["steps"] if s["phase"] in phases}


def pair(runs, a, b, phases):
    ia, ib = outcomes(runs, a, phases), outcomes(runs, b, phases)
    return paired_comparison([(ia[k], ib[k]) for k in ia if k in ib])


def fmt(c):
    lo, hi = c["ci95"]
    return f"{100 * c['diff']:+.1f} pts (CI {100 * lo:+.1f} to {100 * hi:+.1f}, p={c['p_mcnemar']:.3g}, n={c['n']})"


def rate(runs, cond, phases):
    o = outcomes(runs, cond, phases)
    return 100 * sum(o.values()) / len(o)


def main(argv):
    runs = load(Path(argv[0]))
    conds = [c for c in ORDER if c in runs]
    print("## Recurring stream: first-attempt pass rate by phase (%, pooled over seeds", sorted({r['seed'] for r in runs[conds[0]]}), ")")
    print("| condition | A (old) | B (new) | A2 (old again) | memory at end | tokens in / step | model calls / step | seconds / step |")
    print("|---|---|---|---|---|---|---|---|")
    for c in conds:
        rs = runs[c]
        n = sum(r["n"] for r in rs)
        print(f"| {c} | {rate(runs, c, {'A'}):.1f} | {rate(runs, c, {'B'}):.1f} | {rate(runs, c, {'A2'}):.1f} | "
              f"{sum(r['summary']['final_memory_size'] for r in rs) / len(rs):.0f} | {sum(r['summary']['tokens_in'] for r in rs) / n:.0f} | "
              f"{sum(r['summary']['llm_calls'] for r in rs) / n:.2f} | {sum(r['summary']['wall_seconds'] for r in rs) / n:.1f} |")

    print("\n## Registered rules (recurring stream)")
    r1 = pair(runs, "static_rm", "rule_mem", {"B"})
    print(f"R1  phase B, rule_mem vs static_rm: {fmt(r1)}  -> need >= +10 and CI > 0: {'MET' if r1['diff'] >= .10 and r1['ci95'][0] > 0 else 'NOT MET'}")
    ok2 = True
    for rival in ("static_rv", "static_lww"):
        c = pair(runs, rival, "rule_mem", {"B", "A2"})
        met = c["diff"] >= .03 and c["ci95"][0] > 0
        ok2 &= met
        reading = "beats it" if met else ("loses to it" if c["diff"] <= -.03 else "ties it (CI includes 0 or gap < 3)")
        print(f"R2  phases B+A2, rule_mem vs {rival}: {fmt(c)}  -> {reading}")
    print(f"R2  overall (beats both by >= 3 with CI > 0): {'MET' if ok2 else 'NOT MET'}")
    r3 = pair(runs, "static_rm", "rule_mem", {"A"})
    print(f"R3  phase A, rule_mem vs static_rm: {fmt(r3)}  -> need diff >= -5: {'MET' if r3['diff'] >= -.05 else 'NOT MET'}")
    hid = aud = 0
    for r in runs["rule_mem"]:
        h = r["summary"]["hygiene"].get("B", {})
        hid += h.get("wrongly_hidden", 0)
        aud += h.get("audited", 0)
    a_rate, a2_rate = rate(runs, "rule_mem", {"A"}), rate(runs, "rule_mem", {"A2"})
    print(f"R4  end of B: rule_mem wrongly hides {hid}/{aud} audited solutions ({100 * hid / max(aud, 1):.1f}%, need <= 15%); "
          f"A2 {a2_rate:.1f}% vs A {a_rate:.1f}% (need A2 >= A - 10): "
          f"{'MET' if hid <= .15 * max(aud, 1) and a2_rate >= a_rate - 10 else 'NOT MET'}")

    print("\n## Other paired comparisons (rule_mem minus rival), by phase")
    for rival in [c for c in conds if c != "rule_mem"]:
        print(f"- vs {rival}: " + "; ".join(f"{ph}: {fmt(pair(runs, rival, 'rule_mem', {ph}))}" for ph in ("A", "B", "A2")))

    print("\n## Memory hygiene at the end of each phase (summed over seeds: audited / wrongly shown / wrongly hidden)")
    for c in conds:
        if c == "no_memory":
            continue
        cells = []
        for ph in ("A", "B", "A2"):
            hs = [r["summary"]["hygiene"].get(ph, {}) for r in runs[c]]
            cells.append(f"{ph}: {sum(h.get('audited', 0) for h in hs)} / {sum(h.get('wrongly_shown', 0) for h in hs)} / "
                         f"{sum(h.get('wrongly_hidden', 0) for h in hs)}")
        print(f"- {c}: " + " | ".join(cells))

    eq = {}
    base = outcomes(runs, "static_rm", {"B"})
    for c in ("static_rv", "static_lww"):
        o = outcomes(runs, c, {"B"})
        eq[c] = sum(o[k] == base[k] for k in base)
    print(f"\nPer-question agreement with static_rm in phase B: " + ", ".join(f"{c} {v}/{len(base)}" for c, v in eq.items()))

    if len(argv) > 1:
        t = load(Path(argv[1]))
        print("\n## Transfer stream (phase A: half 1 under the old rule; phase B: unseen functions under the new rule)")
        print("| condition | A | B | B vs no_memory |")
        print("|---|---|---|---|")
        for c in [c for c in ORDER if c in t]:
            vs = "" if c == "no_memory" else fmt(pair(t, "no_memory", c, {"B"}))
            print(f"| {c} | {rate(t, c, {'A'}):.1f} | {rate(t, c, {'B'}):.1f} | {vs} |")
        for rival in ("static_rm", "static_rv", "static_lww", "atmc_v6"):
            if rival in t:
                print(f"- B, rule_mem vs {rival}: {fmt(pair(t, rival, 'rule_mem', {'B'}))}")


if __name__ == "__main__":
    main(sys.argv[1:])
