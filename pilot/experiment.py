"""One harness for every experiment, any domain.

    python3 pilot/experiment.py poison    --domain math --seeds 42,123,456
    python3 pilot/experiment.py ablation  --domain code
    python3 pilot/experiment.py baselines --domain code --n 40
    python3 pilot/experiment.py agent     --domain code --heldout 20
    python3 pilot/experiment.py --list

A *preset* is a set of named *conditions* (memory system + agent options).
Every (condition, seed) run is saved on its own to results/eval/, and skipped
if its file already exists -- so a long experiment can be stopped by the daily
API quota and simply re-run tomorrow. Everything runs through agent.Agent.

After training on n tasks, `--heldout m` scores m unseen tasks with memory frozen
(no learning, no hit counting): "does the memory built help on new problems?"
-- the fair way to evaluate things like the curriculum, which changes *which*
tasks are practised and so cannot be judged by in-run pass rate.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import llm
from agent import Agent, build_agent
from config import ROOT, Config, load_config
from domains import get_domain
from embeddings import make_embedder
from factory import build_store

OUT_DIR = ROOT / "pilot" / "results" / "eval"


@dataclass
class Condition:
    kind: str                                   # none | static | reflexion | atmc
    pillars: dict = field(default_factory=dict)  # enable_trust / enable_forgetting / enable_compression
    agent: dict = field(default_factory=dict)    # overrides for cfg.agent (planner, reflection, curriculum)


@dataclass
class Preset:
    conditions: dict[str, Condition]
    poison: bool = False
    heldout: int = 0
    pool_factor: int = 1       # train pool = n * pool_factor (bigger when the agent chooses its own tasks)


PRESETS: dict[str, Preset] = {
    "poison": Preset({"static": Condition("static"), "atmc": Condition("atmc")}, poison=True),
    "ablation": Preset({
        "full_system": Condition("atmc"),
        "wo_trust": Condition("atmc", {"enable_trust": False}),
        "wo_forgetting": Condition("atmc", {"enable_forgetting": False}),
        "wo_compression": Condition("atmc", {"enable_compression": False}),
    }, poison=True),
    "baselines": Preset({
        "no_memory": Condition("none"), "static": Condition("static"),
        "reflexion": Condition("reflexion"), "atmc": Condition("atmc"),
    }, heldout=0),
    "baselines_poison": Preset({
        "no_memory": Condition("none"), "static": Condition("static"),
        "reflexion": Condition("reflexion"), "atmc": Condition("atmc"),
    }, poison=True),
    "agent": Preset({
        "atmc": Condition("atmc"),
        "atmc_planner": Condition("atmc", agent={"use_planner": True}),
        "atmc_reflection": Condition("atmc", agent={"use_reflection": True}),
        "atmc_planner_reflection": Condition("atmc", agent={"use_planner": True, "use_reflection": True}),
        "atmc_random_choice": Condition("atmc", agent={"curriculum": "random"}),
        "atmc_weakness_choice": Condition("atmc", agent={"curriculum": "weakness"}),
    }, heldout=20, pool_factor=3),
}


def condition_cfg(cfg, cond: Condition) -> Config:
    """Experiments follow the pool in a fixed order unless the condition says otherwise."""
    d = cfg.to_dict()
    d["agent"] = {**d["agent"], "curriculum": "fixed", **cond.agent}
    return Config(d)


def evaluate_frozen(agent: Agent, task: dict) -> bool:
    """Score one unseen task with memory read-only."""
    saved = [(m, m.hits) for m in agent.store.items]
    try:
        plan = agent.planner.plan(task) if agent.planner else None
        retrieved = agent.store.retrieve(agent.embedder.embed(task["question"]), k=agent.cfg.retrieval.top_k)
        raw = llm.complete(agent.domain.build_prompt(task, [m.text for m in retrieved], plan))
        return agent.domain.check(task, agent.domain.extract_answer(raw, task)).correct
    finally:
        for m, h in saved:
            m.hits = h


def run_one(exp: str, cname: str, cond: Condition, preset: Preset, seed: int, cfg, embedder, domain, n: int,
            heldout: int, force: bool) -> str:
    path = OUT_DIR / f"{exp}__{domain.name}__{cname}__seed{seed}.json"
    if path.exists() and not force:
        return "skipped (already done)"

    ccfg = condition_cfg(cfg, cond)
    pool_n = n * preset.pool_factor
    tasks = domain.load_tasks(cfg, pool_n + heldout, seed)
    train_pool, held = tasks[:pool_n], tasks[pool_n:pool_n + heldout]
    sequence = train_pool[:n]

    store = build_store(cond.kind, ccfg, embedder, llm.summarize, **cond.pillars)
    n_poison = 0
    if preset.poison:
        for text, emb in domain.poison_specs(sequence, embedder, cfg.evaluation.poison_every):
            store.add(text, emb, correct=False, is_poison=True)
            n_poison += 1

    agent = build_agent(ccfg, store, embedder, domain, train_pool, seed)
    if ccfg.agent.curriculum == "fixed":
        agent.pool = sequence                      # identical sequence for every fixed-order condition
    usage0, t0 = llm.usage_snapshot(), time.time()
    steps = []
    for i in range(n):
        r = agent.step()
        steps.append(r.to_log())
        print(f"  [{cname} s{seed}] {i+1}/{n} {'PASS' if r.correct else 'fail'} ({r.verdict.stage}) "
              f"poison_in_ctx={r.n_poison_retrieved} mem={r.memory_size}", flush=True)

    held_correct, held_detail = None, []
    if heldout:
        held_detail = [{"task_id": t["id"], "correct": evaluate_frozen(agent, t)} for t in held]
        held_correct = sum(h["correct"] for h in held_detail) / len(held_detail)
        print(f"  [{cname} s{seed}] held-out pass rate {held_correct:.2f} on {len(held)} unseen tasks", flush=True)

    usage1 = llm.usage_snapshot()
    warm = cfg.retrieval.top_k          # first tasks before any genuine note can exist (cold start)
    ok = [s["correct"] for s in steps]
    summary = {
        "pass_rate": sum(ok) / n,
        "pass_rate_first_half": sum(ok[: n // 2]) / max(n // 2, 1),
        "pass_rate_second_half": sum(ok[n // 2:]) / max(n - n // 2, 1),
        "heldout_pass_rate": held_correct,
        "poison_exposure": sum(s["n_poison_retrieved"] for s in steps),
        "poison_exposure_after_warmup": sum(s["n_poison_retrieved"] for s in steps[warm:]),
        "poison_used": sum(s["n_poison_used"] for s in steps),
        "n_poison_seeded": n_poison,
        "final_memory_size": steps[-1]["memory_size"],
        "n_compressions": getattr(store, "n_compressions", 0),
        "n_poison_laundered": getattr(store, "n_poison_laundered", 0),
        "llm_calls": usage1["calls"] - usage0["calls"],
        "api_calls": usage1["api_calls"] - usage0["api_calls"],
        "cache_hits": usage1["cache_hits"] - usage0["cache_hits"],
        "tokens_in": usage1["prompt_tokens"] - usage0["prompt_tokens"],
        "tokens_out": usage1["output_tokens"] - usage0["output_tokens"],
        "mean_retrieval_ms": sum(s["retrieval_ms"] for s in steps) / n,
        "wall_seconds": round(time.time() - t0, 1),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "experiment": exp, "domain": domain.name, "condition": cname, "seed": seed, "n": n, "heldout": heldout,
        "embedder": embedder.name, "condition_spec": {"kind": cond.kind, "pillars": cond.pillars, "agent": cond.agent},
        "config": ccfg.to_dict(), "summary": summary, "steps": steps, "heldout_steps": held_detail,
    }, indent=1))
    return f"pass {summary['pass_rate']:.2f}, poison {summary['poison_exposure']}, {summary['llm_calls']} calls"


def estimate_calls(preset: Preset, n: int, heldout: int, seeds: int, only) -> int:
    total = 0
    for name, c in preset.conditions.items():
        if only and name not in only:
            continue
        per_task = 1 + (1 if c.agent.get("use_planner") else 0) + (0.3 if c.agent.get("use_reflection") else 0)
        compress = (n + 14) // 8 if c.kind == "atmc" and c.pillars.get("enable_compression", True) else 0
        total += seeds * (n * per_task + compress + heldout * (2 if c.agent.get("use_planner") else 1))
    return int(total)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset", nargs="?", choices=PRESETS)
    ap.add_argument("--domain", choices=["math", "code"], default="math")
    ap.add_argument("--seeds", default=None, help="comma-separated; default = evaluation.seeds from the config")
    ap.add_argument("--n", type=int, default=None, help="training tasks per run (default: evaluation.n_tasks)")
    ap.add_argument("--heldout", type=int, default=None, help="unseen tasks scored with frozen memory")
    ap.add_argument("--only", default=None, help="comma-separated condition names")
    ap.add_argument("--embedding", default=None, help="fastembed | gemini | tfidf")
    ap.add_argument("--force", action="store_true", help="re-run even if a result file exists")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list or not args.preset:
        for name, p in PRESETS.items():
            print(f"{name:18s} conditions: {', '.join(p.conditions)}  poison={p.poison} heldout={p.heldout}")
        return

    overrides = {"embedding": {"backend": args.embedding}} if args.embedding else None
    cfg = load_config(overrides=overrides)
    llm.configure(cfg)
    preset = PRESETS[args.preset]
    n = args.n or cfg.evaluation.n_tasks
    heldout = args.heldout if args.heldout is not None else preset.heldout
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else cfg.evaluation.seeds
    only = set(args.only.split(",")) if args.only else None
    domain = get_domain(args.domain, cfg)
    embedder = make_embedder(cfg, corpus=[t["question"] for t in domain.load_tasks(cfg, 200, 0)])

    todo = [(c, s) for s in seeds for c in preset.conditions if not only or c in only]   # seed-major: a quota stop leaves whole seeds
    est = estimate_calls(preset, n, heldout, len(seeds), only)
    print(f"{args.preset} / {args.domain}: {len(todo)} runs, n={n}, heldout={heldout}, seeds={seeds}, "
          f"embedder={embedder.name}\n≈{est} LLM calls (free tier: 500/day for this model)\n", flush=True)

    done = skipped = 0
    try:
        for cname, seed in todo:
            print(f"== {cname} seed {seed}", flush=True)
            msg = run_one(args.preset, cname, preset.conditions[cname], preset, seed, cfg, embedder, domain, n,
                          heldout, args.force)
            print(f"   -> {msg}\n", flush=True)
            skipped += msg.startswith("skipped")
            done += not msg.startswith("skipped")
    except llm.QuotaExhausted as e:
        print(f"\nDAILY API QUOTA EXHAUSTED after {done} new run(s) ({skipped} skipped). "
              f"Completed runs are saved; re-run the same command later to continue.\n{str(e)[:160]}")
        sys.exit(3)
    print(f"finished: {done} new run(s), {skipped} already done. Results in {OUT_DIR}")


if __name__ == "__main__":
    main()
