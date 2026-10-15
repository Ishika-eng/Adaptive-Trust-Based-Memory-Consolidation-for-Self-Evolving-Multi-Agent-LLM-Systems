"""Experiment 1, mechanism test: credit ONLY the notes the model cited, or credit EVERY retrieved note?

    pilot/.venv/bin/python pilot/credit_sim.py --trials 300
    pilot/.venv/bin/python pilot/credit_sim.py --sweep

The memory is the REAL `AdaptiveTrustStore` (same trust rule, gate, ranking as the app). Only the *model* is
simulated: given the notes retrieved for a task it decides whether to copy a wrong note, whether the answer is
correct, and which notes it cites. Every behavioural assumption is a parameter below, so this is a controlled test
of the credit rule, not a claim about a real model -- the real-model experiment (`experiment.py credit`) is the
check on those assumptions.

Setting (mirrors the real experiment): wrong notes are planted *recorded as correct* for every 3rd task; the stream
visits the same tasks several times so planted notes are retrieved repeatedly; forgetting and merging are switched off
so the only difference between the two arms is the credit rule.

Credit rules compared (see `Agent.learn`):
  cited : update trust only for notes the model reported using (the project's rule)
  all   : update trust for every retrieved note (the rule used by outcome-driven memory-value methods)
  asym  : EXPLORATORY -- reward only cited notes after a success, blame every retrieved note after a failure

Main metric: trust AUC = P(a genuine note ends with higher trust than a planted wrong note). 0.5 = no separation.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ROOT, load_config  # noqa: E402
from experiment import trust_auc  # noqa: E402
from factory import build_atmc  # noqa: E402


@dataclass
class SimParams:
    n_tasks: int = 40
    repeats: int = 3
    n_clusters: int = 8
    dim: int = 32
    cluster_noise: float = 0.6     # task vector = normalise(cluster centre + noise * gaussian)
    poison_every: int = 3
    base_accuracy: float = 0.90    # P(correct) when the model solves the task without copying a wrong note
    follow: float = 0.30           # P(copy a retrieved wrong note that is a near-duplicate of the task)
    follow_other: float = 0.10     # P(copy a retrieved wrong note that is not a near-duplicate)
    copy_wrong: float = 0.95       # P(answer is wrong | the model copied a wrong note)
    honesty: float = 0.90          # P(the model cites a note it copied)
    use_genuine: float = 0.50      # P(the model cites a genuine retrieved note)
    spurious: float = 0.05         # P(the model cites a wrong note it did NOT copy)
    omit_citation: float = 0.05    # P(the USED line is missing: the parser then treats every note as used)
    dup_sim: float = 0.95          # similarity above which a note counts as a near-duplicate of the task


MODES = ("cited", "all", "asym")


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) or 1.0)


def make_tasks(p: SimParams, rng: np.random.Generator) -> list[np.ndarray]:
    centres = [_unit(rng.normal(size=p.dim)) for _ in range(p.n_clusters)]
    return [_unit(centres[i % p.n_clusters] + p.cluster_noise * _unit(rng.normal(size=p.dim))) for i in range(p.n_tasks)]


def run_trial(p: SimParams, mode: str, seed: int, cfg=None) -> dict:
    """One stream under one credit rule. Returns the end-of-run trust metrics and behaviour counts."""
    assert mode in MODES
    cfg = cfg or load_config()
    tasks = make_tasks(p, np.random.default_rng(seed))
    rng = random.Random(f"{seed}-{mode}")
    store = build_atmc(cfg, types.SimpleNamespace(embed=lambda t: None), lambda t: "",
                       enable_forgetting=False, enable_compression=False)
    for i, vec in enumerate(tasks):
        if i % p.poison_every == 0:
            store.add(f"planted wrong note {i}", vec, correct=True, is_poison=True)   # a false success
    k = cfg.retrieval.top_k
    exposures_later = harmful = wrong = cited_poison = steps = 0
    for rep in range(p.repeats):
        for t, vec in enumerate(tasks):
            retrieved = store.retrieve(vec, k=k)
            followed = [False] * len(retrieved)
            for j, m in enumerate(retrieved):
                if m.is_poison:
                    sim = float(np.dot(_unit(np.asarray(m.embedding)), vec))
                    followed[j] = rng.random() < (p.follow if sim >= p.dup_sim else p.follow_other)
            copied = any(followed)
            correct = (rng.random() > p.copy_wrong) if copied else (rng.random() < p.base_accuracy)
            cites = []
            for j, m in enumerate(retrieved):
                if m.is_poison:
                    cites.append(rng.random() < (p.honesty if followed[j] else p.spurious))
                else:
                    cites.append(rng.random() < p.use_genuine)
            if rng.random() < p.omit_citation:
                cites = [True] * len(retrieved)            # missing USED line -> everything counted as used
            if rep > 0:
                exposures_later += sum(m.is_poison for m in retrieved)
            cited_poison += sum(c and m.is_poison for c, m in zip(cites, retrieved))
            harmful += copied and not correct
            wrong += not correct
            use_cites = mode == "cited" or (mode == "asym" and correct)
            store.update_from_outcome(retrieved, correct, used_mask=cites if use_cites else None)
            store.add(f"attempt {rep}-{t}", vec, correct=correct)
            steps += 1
    genuine = [m.trust for m in store.items if not m.is_poison and m.correct is True]
    poison = [m.trust for m in store.items if m.is_poison]
    gate = store.t_min
    return {
        "trust_auc": trust_auc(genuine, poison),
        "genuine_trust": float(np.mean(genuine)) if genuine else None,
        "poison_trust": float(np.mean(poison)) if poison else None,
        "genuine_blocked": float(np.mean([t < gate for t in genuine])) if genuine else None,
        "poison_blocked": float(np.mean([t < gate for t in poison])) if poison else None,
        "poison_exposed_after_pass1": exposures_later,
        "harmful_steps": harmful,
        "accuracy": 1 - wrong / steps,
        "poison_cited": cited_poison,
    }


METRICS = ["trust_auc", "genuine_trust", "poison_trust", "genuine_blocked", "poison_blocked",
           "poison_exposed_after_pass1", "harmful_steps", "accuracy", "poison_cited"]


def _ci(xs: list[float], boots: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(xs)
    means = sorted(sum(rng.choices(xs, k=n)) / n for _ in range(boots))
    return sum(xs) / n, means[int(0.025 * boots)], means[int(0.975 * boots) - 1]


PAIRS = [("cited", "all"), ("asym", "all"), ("asym", "cited")]       # (A, B): reports A - B


def compare(p: SimParams, trials: int, base_seed: int = 1000) -> dict:
    runs = {m: [run_trial(p, m, base_seed + i) for i in range(trials)] for m in MODES}
    out = {"params": asdict(p), "trials": trials, "per_mode": {}, "paired": {}}
    for key in METRICS:
        out["per_mode"][key] = {}
        for m in MODES:
            xs = [r[key] for r in runs[m] if r[key] is not None]
            if xs:
                out["per_mode"][key][m] = list(_ci(xs))
        out["paired"][key] = {}
        for a, b in PAIRS:
            pairs = [(x[key], y[key]) for x, y in zip(runs[a], runs[b]) if x[key] is not None and y[key] is not None]
            if pairs:
                diffs = [x - y for x, y in pairs]
                out["paired"][key][f"{a}-{b}"] = list(_ci(diffs)) + [sum((d > 0) + 0.5 * (d == 0) for d in diffs) / len(diffs)]
    return out


def render(res: dict) -> str:
    p = res["params"]
    lines = [f"credit-rule simulation: {res['trials']} paired trials, follow={p['follow']}, honesty={p['honesty']}, "
             f"base_accuracy={p['base_accuracy']}, repeats={p['repeats']}", "",
             f"{'metric':28s}" + "".join(f"{m + ' (95% CI)':27s}" for m in MODES)]
    f = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"
    for key in METRICS:
        row = res["per_mode"].get(key, {})
        lines.append(f"{key:28s}" + "".join(f"{f(row[m]) if m in row else '-':27s}" for m in MODES))
    lines += ["", "paired differences (A - B), 95% CI, and share of trials where A > B:"]
    for key in ("trust_auc", "genuine_trust", "poison_exposed_after_pass1", "harmful_steps", "accuracy"):
        parts = []
        for a, b in PAIRS:
            v = res["paired"].get(key, {}).get(f"{a}-{b}")
            parts.append(f"{a}-{b}: {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}] ({100 * v[3]:.0f}%)" if v else f"{a}-{b}: -")
        lines.append(f"  {key:26s}" + "   ".join(parts))
    return "\n".join(lines)


def sweep(trials: int) -> str:
    rows = [f"{'follow':>6s}{'honest':>7s} | trust AUC: cited   all  asym | AUC cited-all [95% CI]   | harmful steps: cited   all  asym | genuine blocked: all"]
    for follow in (0.0, 0.2, 0.5, 0.8):
        for honesty in (0.95, 0.6):
            r = compare(SimParams(follow=follow, honesty=honesty), trials)
            auc, harm, blk = r["per_mode"]["trust_auc"], r["per_mode"]["harmful_steps"], r["per_mode"]["genuine_blocked"]
            d = r["paired"]["trust_auc"]["cited-all"]
            rows.append(f"{follow:6.1f}{honesty:7.2f} | {auc['cited'][0]:17.3f} {auc['all'][0]:5.3f} {auc['asym'][0]:5.3f} | "
                        f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] | {harm['cited'][0]:20.1f} {harm['all'][0]:5.1f} {harm['asym'][0]:5.1f} | "
                        f"{blk['all'][0]:.3f}")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--sweep", action="store_true", help="grid over how often the model copies wrong notes and how honest its citations are")
    for f in SimParams.__dataclass_fields__.values():
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    a = ap.parse_args()
    out_dir = ROOT / "pilot" / "results" / "sim"
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.sweep:
        text = sweep(a.trials)
        print(text)
        (out_dir / "credit_sweep.txt").write_text(text + "\n")
        return
    p = SimParams(**{k: getattr(a, k) for k in SimParams.__dataclass_fields__})
    res = compare(p, a.trials)
    print(render(res))
    (out_dir / "credit_sim.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
