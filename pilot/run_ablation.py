"""The README's Section 8 ablation study, for real: Full System vs.
w/o Trust Scoring vs. w/o Adaptive Forgetting vs. w/o Memory Compression --
all four built from the same AdaptiveTrustStore class with different pillars
toggled off, so the only thing that varies between conditions is the pillar
being ablated.

Uses the same poisoned-memory setup as run_poison_pilot.py (same reasoning:
GSM8K is too easy for the base model to make organic mistakes worth
filtering, so we test resistance to deliberately planted bad memories
instead). Each condition sees the identical task sequence and identical
poison seed, so differences are attributable to the ablated pillar alone.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import llm
from config import load_experiment_config
from embeddings import make_embedder
from factory import build_atmc
from tasks import build_poison_specs, load_tasks

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# README Section 8: Full System / w/o Trust / w/o Forgetting / w/o Compression.
# Importance is never switched off (the README table has no such row).
CONDITIONS = {
    "full_system":    dict(enable_trust=True,  enable_forgetting=True,  enable_compression=True),
    "wo_trust":       dict(enable_trust=False, enable_forgetting=True,  enable_compression=True),
    "wo_forgetting":  dict(enable_trust=True,  enable_forgetting=False, enable_compression=True),
    "wo_compression": dict(enable_trust=True,  enable_forgetting=True,  enable_compression=False),
}


def run_condition(cond_name, store, tasks, embedder, cfg):
    log = []
    correct_count = 0
    poison_exposure = 0
    for i, task in enumerate(tasks):
        q_emb = embedder.embed(task["question"])
        retrieved = store.retrieve(q_emb, k=cfg.retrieval.top_k)
        mem_texts = [m.text for m in retrieved]
        n_poison_retrieved = sum(1 for m in retrieved if m.is_poison)
        poison_exposure += n_poison_retrieved

        raw = llm.solve(task["question"], mem_texts)
        pred = llm.extract_final_number(raw)
        correct = llm.numbers_match(pred, task["gold"])
        correct_count += int(correct)

        used_mask = llm.extract_used_mask(raw, len(retrieved))
        store.update_from_outcome(retrieved, correct, used_mask=used_mask)

        reflection = (
            f"Q: {task['question']}\n"
            f"A: {raw.strip()[:400]}\n"
            f"Outcome: {'CORRECT' if correct else 'INCORRECT'} (gold={task['gold']})"
        )
        store.add(reflection, q_emb, correct)

        log.append({
            "idx": i,
            "correct": correct,
            "n_poison_retrieved": n_poison_retrieved,
            "poison_exposure_cumulative": poison_exposure,
            "memory_size": store.size(),
        })
        print(f"[{cond_name}] {i+1}/{len(tasks)} correct={correct} "
              f"poison_hit={n_poison_retrieved>0} mem_size={store.size()} "
              f"compressions={store.n_compressions} running_acc={correct_count/(i+1):.3f}")
        time.sleep(cfg.model.request_interval_s)
    return log


def summarize_log(log, store):
    n = len(log)
    return {
        "overall_accuracy": sum(x["correct"] for x in log) / n,
        "total_poison_exposure": log[-1]["poison_exposure_cumulative"],
        "final_memory_size": log[-1]["memory_size"],
        "n_compressions": store.n_compressions,
        "n_poison_laundered": store.n_poison_laundered,
    }


def main():
    cfg, seed = load_experiment_config()
    llm.configure(cfg)
    tasks = load_tasks(cfg, cfg.evaluation.n_tasks, seed)
    embedder = make_embedder(cfg, corpus=[t["question"] for t in tasks])
    print(f"embedder={embedder.name}  seed={seed}  n_tasks={len(tasks)}  model={cfg.model.name}")

    poison_specs = build_poison_specs(tasks, embedder, cfg.evaluation.poison_every)
    print(f"Seeding {len(poison_specs)} poisoned memories into each of {len(CONDITIONS)} conditions.\n")

    RESULTS_DIR.mkdir(exist_ok=True)
    results = {}
    for cond_name, flags in CONDITIONS.items():
        print(f"=== Running {cond_name} ({flags}) ===")
        store = build_atmc(cfg, embedder, llm.summarize, label=cond_name, **flags)
        for text, emb in poison_specs:
            store.add(text, emb, correct=False, is_poison=True)
        log = run_condition(cond_name, store, tasks, embedder, cfg)
        results[cond_name] = {"log": log, "summary": summarize_log(log, store), "flags": flags}
        print()

    out_path = RESULTS_DIR / f"ablation_seed{seed}_{int(time.time())}.json"
    out_path.write_text(json.dumps({
        "n_tasks": cfg.evaluation.n_tasks, "seed": seed, "embedder": embedder.name,
        "config": cfg.to_dict(), "n_poison_seeded": len(poison_specs), "results": results,
    }, indent=2))

    print("\n\n===== ABLATION TABLE (README Section 8) =====")
    print(f"embedder: {embedder.name}")
    print(f"{'Configuration':20s} {'Accuracy':>10s} {'PoisonExp':>10s} {'MemSize':>8s} {'#Compress':>10s} {'#Laundered':>11s}")
    for cond_name in CONDITIONS:
        s = results[cond_name]["summary"]
        print(f"{cond_name:20s} {s['overall_accuracy']:>10.3f} {s['total_poison_exposure']:>10d} "
              f"{s['final_memory_size']:>8d} {s['n_compressions']:>10d} {s['n_poison_laundered']:>11d}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
