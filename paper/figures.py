"""Figures for the paper, drawn from the saved experiment results (no numbers are typed in by hand).

    pilot/.venv/bin/python paper/figures.py
"""
import glob
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pilot"))
from memory import update_trust  # noqa: E402

OUT = ROOT / "paper" / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})
BLUE, GREY, LIGHT, RED, GREEN = "#2f5fb3", "#9aa0a6", "#c9ced4", "#c0392b", "#2e8b57"


def runs(exp, dom, cond):
    return [json.load(open(f)) for f in sorted(glob.glob(str(ROOT / f"pilot/results/eval/{exp}__{dom}__{cond}__seed*.json")))]


def vals(exp, dom, cond, key):
    return [r["summary"][key] for r in runs(exp, dom, cond)]


def ms(xs):
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0)


def box(ax, x, y, w, h, title, sub=None, fs=7.2):
    ax.add_patch(Rectangle((x, y), w, h, fill=True, fc="white", ec="black", lw=1.0))
    if sub:
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontweight="bold", fontsize=fs)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=fs - 1.2)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontweight="bold", fontsize=fs)


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=1.1, color="black"))


# ---------------------------------------------------------------- Fig 1: end-to-end workflow
def fig_workflow():
    steps = [
        ("User Question", "Python task, optional tests"),
        ("Plan", "Choose strategy; plan never sees memory"),
        ("Retrieve Memory", "Top-3 notes by trust-gated score"),
        ("Write Code", "LLM; cites the notes it relied on"),
        ("Verify", "Sandboxed tests / generated tests / user"),
        ("Update Memory", "Trust, forgetting, merging, lesson"),
        ("Choose Next Task", "Weakness-targeted practice"),
    ]
    h, gap = 1.05, 0.6
    total = len(steps) * h + (len(steps) - 1) * gap
    fig, ax = plt.subplots(figsize=(3.4, 0.34 * (total + 0.4)))
    ax.set_xlim(0, 10); ax.set_ylim(0, total + 0.4); ax.axis("off")
    top = total + 0.2
    for i, (t, s) in enumerate(steps):
        y = top - (i + 1) * h - i * gap
        box(ax, 0.6, y, 8.8, h, t, s, fs=7.8)
        if i < len(steps) - 1:
            arrow(ax, (5, y), (5, y - gap))
    fig.savefig(OUT / "fig1_workflow.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# ---------------------------------------------------------------- Fig 2: architecture
def fig_architecture():
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.set_xlim(0, 20); ax.set_ylim(0, 17.6); ax.axis("off")
    box(ax, 0.4, 14.2, 7.2, 2.8, "CHAT UI", "React + Vite")
    box(ax, 12.4, 14.2, 7.2, 2.8, "BACKEND", "FastAPI, SQLite")
    box(ax, 6.2, 8.9, 7.6, 3.4, "AGENT LOOP", "Plan - Act - Check - Reflect")
    box(ax, 0.4, 3.4, 8.6, 3.6, "ATMC MEMORY", "trust, importance, forgetting")
    box(ax, 11.0, 3.4, 8.6, 3.6, "SANDBOX", "tests, resource limits")
    box(ax, 0.4, 0.0, 8.6, 2.2, "Local embedder", None, fs=6.6)
    box(ax, 11.0, 0.0, 8.6, 2.2, "LLM API (Gemini)", None, fs=6.6)
    arrow(ax, (7.6, 15.6), (12.4, 15.6)); arrow(ax, (12.4, 15.0), (7.6, 15.0))
    arrow(ax, (16.0, 14.2), (11.8, 12.3))
    arrow(ax, (8.0, 8.9), (5.0, 7.0)); arrow(ax, (5.9, 7.0), (8.9, 8.9))
    arrow(ax, (12.0, 8.9), (14.5, 7.0))
    arrow(ax, (14.8, 3.4), (14.8, 2.2))
    arrow(ax, (4.7, 3.4), (4.7, 2.2))
    fig.savefig(OUT / "fig2_architecture.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# ---------------------------------------------------------------- Fig 3: poison exposure
def fig_poison():
    groups = [
        ("GSM8K", [("Static", "poison", "math", "static", GREY), ("ATMC", "poison", "math", "atmc", BLUE)]),
        ("HumanEval/MBPP", [("Static", "baselines_poison", "code", "static", GREY),
                            ("Reflexion-style", "baselines_poison", "code", "reflexion", LIGHT),
                            ("ATMC", "baselines_poison", "code", "atmc", BLUE)]),
    ]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x, ticks, labels, seen = 0.0, [], [], set()
    for gname, bars in groups:
        start = x
        for lab, exp, dom, cond, col in bars:
            xs = vals(exp, dom, cond, "poison_exposure")
            m, s = ms(xs)
            ax.bar(x, m, 0.8, color=col, edgecolor="black", lw=0.6, yerr=s, capsize=2, label=lab if lab not in seen else None)
            seen.add(lab)
            ax.scatter([x] * len(xs), xs, s=7, color="black", zorder=3)
            ax.text(x, m + s + 2.2, f"{m:.1f}", ha="center", fontsize=6.8)
            x += 1
        ticks.append((start + x - 1) / 2); labels.append(gname)
        x += 0.8
    ax.set_xticks(ticks); ax.set_xticklabels(labels)
    ax.set_ylabel("Planted notes shown to the model\n(per 40-task run)")
    ax.set_ylim(0, 80)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, columnspacing=1.0, handlelength=1.2)
    fig.savefig(OUT / "fig3_poison.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# ---------------------------------------------------------------- Fig 4: ablation
def fig_ablation():
    conds = [("full_system", "Full ATMC"), ("wo_trust", "w/o Trust"), ("wo_forgetting", "w/o Forgetting"), ("wo_compression", "w/o Merging")]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    w = 0.38
    for j, (dom, col, name) in enumerate([("math", BLUE, "GSM8K (5 seeds)"), ("code", GREY, "HumanEval/MBPP (3 seeds)")]):
        for i, (c, _) in enumerate(conds):
            xs = vals("ablation", dom, c, "poison_exposure_after_warmup")
            m, s = ms(xs)
            xpos = i + (j - 0.5) * w
            ax.bar(xpos, m, w, color=col, edgecolor="black", lw=0.6, yerr=s, capsize=2, label=name if i == 0 else None)
            ax.scatter([xpos] * len(xs), xs, s=6, color="black", zorder=3)
            ax.text(xpos, m + s + 3, f"{m:.0f}", ha="center", fontsize=6.4)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels([n for _, n in conds], fontsize=6.8)
    ax.set_ylabel("Planted notes in context\n(after 3-task warm-up)")
    ax.set_ylim(0, 150)
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(OUT / "fig4_ablation.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# ---------------------------------------------------------------- Fig 5: trust dynamics (from the update rule)
def fig_trust():
    eta, delta, t_min = 0.05, 0.15, 0.30
    t0 = update_trust(0.5, +1, eta, delta)          # a note that has passed the gate once
    def path(w):
        t, out = t0, [t0]
        for _ in range(5):
            t = update_trust(t, -1, eta * w, delta * w)
            out.append(t)
        return out
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ks = list(range(6))
    ax.plot(ks, path(1.0), "-o", color=BLUE, ms=3.5, lw=1.4, label="ATMC, verified by real tests")
    ax.plot(ks, path(0.5), "-s", color=GREEN, ms=3.5, lw=1.4, label="ATMC, generated-test evidence (half weight)")
    ax.plot(ks, [t0] * 6, "--", color=GREY, lw=1.4, label="Static memory (no trust)")
    ax.axhline(t_min, color=RED, ls=":", lw=1.2)
    ax.text(5.05, t_min + 0.012, "retrieval gate $T_{min}$", color=RED, ha="right", fontsize=6.6)
    ax.set_xlabel("Verified failures in which the note was cited")
    ax.set_ylabel("Trust score $T$")
    ax.set_ylim(0, 0.9); ax.set_xlim(-0.1, 5.1)
    ax.legend(loc="upper right", frameon=False, fontsize=6.3)
    fig.savefig(OUT / "fig5_trust.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# ---------------------------------------------------------------- Fig 6: agent components, held-out
def fig_agent():
    conds = [("atmc", "ATMC"), ("atmc_planner", "+Planner"), ("atmc_reflection", "+Reflect."), ("atmc_planner_reflection", "+Plan+Refl."),
             ("atmc_random_choice", "Random\ntasks"), ("atmc_weakness_choice", "Weakness\ntasks")]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for i, (c, lab) in enumerate(conds):
        xs = [100 * v for v in vals("agent", "code", c, "heldout_pass_rate")]
        m, s = ms(xs)
        ax.bar(i, m, 0.7, color=BLUE if i else GREY, edgecolor="black", lw=0.6, yerr=s, capsize=2)
        ax.scatter([i] * len(xs), xs, s=6, color="black", zorder=3)
        ax.text(i, m + s + 1.2, f"{m:.1f}", ha="center", fontsize=6.6)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels([l for _, l in conds], fontsize=6.4)
    ax.set_ylabel("Held-out pass rate (%)\n(20 unseen tasks per seed)")
    ax.set_ylim(70, 105)
    fig.savefig(OUT / "fig6_agent.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig_workflow, fig_architecture, fig_poison, fig_ablation, fig_trust, fig_agent):
        f()
        print("wrote", f.__name__)
