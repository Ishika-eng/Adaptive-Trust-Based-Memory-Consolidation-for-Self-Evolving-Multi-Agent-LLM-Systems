"""Tables, CSV and figures for the local-model learning experiment, rebuilt from the saved result files.

    pilot/.venv/bin/python pilot/report_learning.py --dir pilot/results/local_learning --exp learning --name local_learning
    pilot/.venv/bin/python pilot/report_learning.py --dir pilot/results/local_learning_noisy --exp learning_noisy --name local_learning_noisy

Writes docs/results/<name>/{results.md, per_run.csv, summary.json, *.png}. Nothing is typed by hand: every number comes
from the per-step logs in the result JSONs, so the paper and slides can be regenerated at any time. Works on partial
results too (the tables show how many seeds each condition has).
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ROOT  # noqa: E402
from report import load_runs, paired_comparison, paired_outcomes  # noqa: E402

ORDER = ["no_memory", "static", "atmc_no_trust", "atmc_no_merge", "atmc", "atmc_simrank", "atmc_noforget", "atmc_simrank_noforget", "atmc_pure", "atmc_trust_only", "atmc_nohits", "atmc_ev", "atmc_ev_nohits", "ladder_trust", "ladder_trust_forget", "static_repair", "atmc_v3_nocards", "atmc_v3", "static_rm", "atmc_v4_simonly", "atmc_v4"]
LABEL = {"static_repair": "Plain memory + examples", "atmc_v3_nocards": "ATMC v3 without cards", "atmc_v3": "ATMC v3 (examples + cards)",
         "static_rm": "Plain memory + examples + memory retry", "atmc_v4_simonly": "ATMC cards + memory retry (similarity)", "atmc_v4": "ATMC v4 (utility + signature)",
         "no_memory": "No memory", "static": "Static memory", "atmc_no_trust": "ATMC without trust",
         "atmc_no_merge": "ATMC without merging", "atmc": "ATMC (full)",
         "atmc_simrank": "ATMC, similarity-first ranking", "atmc_noforget": "ATMC, no forgetting",
         "atmc_simrank_noforget": "ATMC, sim-first + no forgetting",
         "atmc_pure": "ATMC, all pillars off + no hit boost", "atmc_trust_only": "ATMC, trust only + no hit boost",
         "atmc_nohits": "ATMC (full), no hit boost",
         "atmc_ev": "ATMC + verified-evidence trust", "atmc_ev_nohits": "ATMC + verified-evidence trust, no hit boost",
         "ladder_trust": "Ladder 1: + trust", "ladder_trust_forget": "Ladder 2: + trust + forgetting"}
COLOUR = {"static_repair": "#e0a030", "atmc_v3_nocards": "#8e44ad", "atmc_v3": "#2f5fb3", "static_rm": "#e0a030", "atmc_v4_simonly": "#8e44ad", "atmc_v4": "#2f5fb3",
          "no_memory": "#9aa0a6", "static": "#e0a030", "atmc_no_trust": "#c0392b", "atmc_no_merge": "#2e8b57", "atmc": "#2f5fb3",
          "atmc_simrank": "#7b3fa0", "atmc_noforget": "#17a2b8", "atmc_simrank_noforget": "#000000",
          "atmc_pure": "#555555", "atmc_trust_only": "#d35400", "atmc_nohits": "#1b4f9c", "atmc_ev": "#0b8a3e", "atmc_ev_nohits": "#6b8e23", "ladder_trust": "#8e44ad", "ladder_trust_forget": "#c2185b"}
KNOWN = {"no_mutation"}         # known-practice control family: the model already does this by default


def role(task_id: str) -> str:
    fam = task_id.split("/")[1]
    return "control" if fam == "control" else "known_practice" if fam in KNOWN else "rule"


def pass_rate(steps, pred=lambda s: True):
    xs = [s["correct"] for s in steps if pred(s)]
    return sum(xs) / len(xs) if xs else None


def rolling(seqs: list[list[int]], w: int) -> list[float]:
    """Mean over seeds of the trailing-window pass rate at each step."""
    n = min(len(s) for s in seqs)
    out = []
    for i in range(n):
        lo = max(0, i - w + 1)
        out.append(st.mean(sum(s[lo:i + 1]) / (i + 1 - lo) for s in seqs))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--exp", default="learning")
    ap.add_argument("--name", default=None, help="output folder under docs/results (default: the results folder's name)")
    ap.add_argument("--also", action="append", default=[], metavar="DIR:EXP", help="add runs from another results folder, e.g. pilot/results/local_learning_diag:learning_diag")
    ap.add_argument("--window", type=int, default=20)
    a = ap.parse_args()
    src = (ROOT / a.dir) if not Path(a.dir).is_absolute() else Path(a.dir)
    out = ROOT / "docs" / "results" / (a.name or src.name)
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(a.exp, "code", src)
    for spec in a.also:
        d2, e2 = spec.rsplit(":", 1)
        for c, rs in load_runs(e2, "code", ROOT / d2 if not Path(d2).is_absolute() else Path(d2)).items():
            runs[c] += rs
    conds = [c for c in ORDER if c in runs] + [c for c in runs if c not in ORDER]
    if not conds:
        sys.exit(f"no result files for experiment {a.exp!r} in {src}")

    summary, lines = {}, [f"# Results: {a.exp} ({src.name})", "",
                          "Local model, hidden-convention benchmark. Pass rates are on the TRUE outcome of each task. "
                          "Seeds = independent task orders; every condition sees the same order for a given seed.", ""]

    # --- per-run CSV
    with open(out / "per_run.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "seed", "n_steps", "pass_rate", "first_half", "second_half", "rule_pass", "control_pass",
                    "final_memory", "tokens_in", "tokens_out", "wall_seconds"])
        for c in conds:
            for r in runs[c]:
                s, steps = r["summary"], r["steps"]
                w.writerow([c, r["seed"], s["n_steps"], f"{s['pass_rate']:.4f}", f"{s['pass_rate_first_half']:.4f}",
                            f"{s['pass_rate_second_half']:.4f}",
                            f"{pass_rate(steps, lambda x: role(x['task_id']) == 'rule'):.4f}",
                            f"{pass_rate(steps, lambda x: role(x['task_id']) == 'control'):.4f}",
                            s["final_memory_size"], s["tokens_in"], s["tokens_out"], s["wall_seconds"]])

    # --- headline table
    lines += ["## Headline (mean over seeds; +/- = standard deviation across seeds)", "",
              "| Condition | Seeds | Pass rate | First half | Second half | Rule tasks | Controls | Memory size |",
              "|---|---|---|---|---|---|---|---|"]
    for c in conds:
        rs = runs[c]
        col = lambda f: [f(r) for r in rs]
        pr = col(lambda r: r["summary"]["pass_rate"])
        fh, sh = col(lambda r: r["summary"]["pass_rate_first_half"]), col(lambda r: r["summary"]["pass_rate_second_half"])
        rule = col(lambda r: pass_rate(r["steps"], lambda x: role(x["task_id"]) == "rule"))
        ctl = col(lambda r: pass_rate(r["steps"], lambda x: role(x["task_id"]) == "control"))
        mem = col(lambda r: r["summary"]["final_memory_size"])
        sd = lambda xs: f"{st.mean(xs):.3f} ± {st.stdev(xs):.3f}" if len(xs) > 1 else f"{st.mean(xs):.3f}"
        lines.append(f"| {LABEL.get(c, c)} | {len(rs)} | {sd(pr)} | {sd(fh)} | {sd(sh)} | {sd(rule)} | {sd(ctl)} | {st.mean(mem):.0f} |")
        summary[c] = {"seeds": len(rs), "pass_rate": st.mean(pr), "first_half": st.mean(fh), "second_half": st.mean(sh),
                      "rule_pass": st.mean(rule), "control_pass": st.mean(ctl)}

    # --- paired comparisons
    lines += ["", "## Paired comparisons (same tasks, same order)", "",
              "Difference = B minus A in pass rate over all (seed, task) pairs; 95% bootstrap CI; exact McNemar p-value. "
              "Only the pairs both conditions completed are used.", "",
              "| A | B | n pairs | B - A | 95% CI | B right / A right | McNemar p |", "|---|---|---|---|---|---|---|"]
    pairs_spec = [("no_memory", c) for c in conds if c != "no_memory"] + [("static", "atmc"), ("atmc_no_trust", "atmc"), ("atmc_no_merge", "atmc")] + [("static", c) for c in ("atmc_simrank", "atmc_noforget", "atmc_simrank_noforget")] + [("atmc", c) for c in ("atmc_simrank", "atmc_noforget", "atmc_simrank_noforget")]
    summary["paired"] = {}
    for A, B in pairs_spec:
        if A in runs and B in runs and A != B:
            pc = paired_comparison(paired_outcomes(runs[A], runs[B]))
            if pc.get("n"):
                lines.append(f"| {LABEL.get(A, A)} | {LABEL.get(B, B)} | {pc['n']} | {pc['diff']:+.3f} | "
                             f"[{pc['ci95'][0]:+.3f}, {pc['ci95'][1]:+.3f}] | {pc['b_better']} / {pc['a_better']} | {pc['p_mcnemar']:.4f} |")
                summary["paired"][f"{A}->{B}"] = pc

    # --- per family
    fams = sorted({s["task_id"].split("/")[1] for r in runs[conds[0]] for s in r["steps"]})
    lines += ["", "## Pass rate by family (mean over seeds)", "", "| Family | " + " | ".join(LABEL.get(c, c) for c in conds) + " |",
              "|---|" + "---|" * len(conds)]
    fam_rates = {c: {} for c in conds}
    for fam in fams:
        row = []
        for c in conds:
            xs = [pass_rate(r["steps"], lambda x, fam=fam: x["task_id"].split("/")[1] == fam) for r in runs[c]]
            xs = [x for x in xs if x is not None]
            fam_rates[c][fam] = st.mean(xs) if xs else None
            row.append(f"{fam_rates[c][fam]:.2f}" if xs else "-")
        lines.append(f"| {fam}{' (control)' if fam == 'control' else ' (known practice)' if fam in KNOWN else ''} | " + " | ".join(row) + " |")

    # --- cost
    lines += ["", "## Cost (local model, no API quota)", "", "| Condition | Tokens in per run | Tokens out per run | Wall time per run (min) |", "|---|---|---|---|"]
    for c in conds:
        rs = runs[c]
        lines.append(f"| {LABEL.get(c, c)} | {st.mean(r['summary']['tokens_in'] for r in rs):.0f} | "
                     f"{st.mean(r['summary']['tokens_out'] for r in rs):.0f} | {st.mean(r['summary']['wall_seconds'] for r in rs) / 60:.1f} |")
    (out / "results.md").write_text("\n".join(lines) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    # --- figures
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, only_rule, title in ((axs[0], False, "All tasks"), (axs[1], True, "Hidden-rule tasks only")):
        for c in conds:
            seqs = []
            for r in runs[c]:
                steps = [s for s in r["steps"] if not only_rule or role(s["task_id"]) == "rule"]
                seqs.append([int(s["correct"]) for s in steps])
            if seqs and min(map(len, seqs)) >= 5:
                y = rolling(seqs, a.window if not only_rule else max(10, a.window // 2))
                ax.plot(range(1, len(y) + 1), y, label=LABEL.get(c, c), color=COLOUR.get(c), lw=2 if c == "atmc" else 1.4)
        ax.set_xlabel("task number in the stream"); ax.set_ylabel(f"pass rate (rolling {a.window if not only_rule else max(10, a.window // 2)})")
        ax.set_title(title); ax.set_ylim(0, 1); ax.grid(alpha=.25)
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Learning curve: does pass rate rise as the agent works?", y=1.02)
    fig.tight_layout(); fig.savefig(out / "learning_curve.png", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    xs = range(len(conds))
    means = [summary[c]["pass_rate"] for c in conds]
    errs = [st.stdev([r["summary"]["pass_rate"] for r in runs[c]]) if len(runs[c]) > 1 else 0 for c in conds]
    ax.bar(xs, means, yerr=errs, color=[COLOUR.get(c, "#888") for c in conds], capsize=3)
    for i, c in enumerate(conds):
        for r in runs[c]:
            ax.plot(i, r["summary"]["pass_rate"], "o", color="black", ms=3)
    ax.set_xticks(list(xs)); ax.set_xticklabels([LABEL.get(c, c).replace(" ", "\n", 1) for c in conds], fontsize=8)
    ax.set_ylabel("pass rate"); ax.set_ylim(0, 1); ax.set_title("Overall pass rate (dots = seeds, bars = mean ± sd)")
    fig.tight_layout(); fig.savefig(out / "overall.png", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3.8))
    wdt = 0.8 / len(conds)
    for j, c in enumerate(conds):
        ax.bar([i + j * wdt for i in range(len(fams))], [fam_rates[c].get(f) or 0 for f in fams], wdt,
               label=LABEL.get(c, c), color=COLOUR.get(c, "#888"))
    ax.set_xticks([i + 0.4 - wdt / 2 for i in range(len(fams))]); ax.set_xticklabels(fams, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("pass rate"); ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=8, ncol=len(conds)); ax.set_title("Pass rate by convention family")
    fig.tight_layout(); fig.savefig(out / "by_family.png", bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}/ (results.md, per_run.csv, summary.json, learning_curve.png, overall.png, by_family.png)")


if __name__ == "__main__":
    main()
