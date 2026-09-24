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
import os
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.feature_extraction.text import TfidfVectorizer

from stores import AdaptiveTrustStore
import llm

N_TASKS = int(os.environ.get("PILOT_N_TASKS", "40"))
TOP_K = 3
SEED = int(os.environ.get("PILOT_SEED", "42"))
POISON_EVERY = 3
DATA_PATH = Path(__file__).resolve().parent / "data" / "gsm8k_test.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

CONDITIONS = {
    "full_system":        dict(enable_trust=True,  enable_forgetting=True,  enable_compression=True),
    "wo_trust":            dict(enable_trust=False, enable_forgetting=True,  enable_compression=True),
    "wo_forgetting":       dict(enable_trust=True,  enable_forgetting=False, enable_compression=True),
    "wo_compression":      dict(enable_trust=True,  enable_forgetting=True,  enable_compression=False),
}


def load_tasks(n: int, seed: int):
    tasks = []
    with open(DATA_PATH) as f:
        for line in f:
            tasks.append(json.loads(line))
    rng = random.Random(seed)
    rng.shuffle(tasks)
    tasks = tasks[:n]
    for t in tasks:
        t["gold"] = t["answer"].split("####")[-1].strip()
    return tasks


def make_wrong_answer(gold: str) -> str:
    try:
        g = float(gold.replace(",", ""))
        wrong = g + (7 if g >= 0 else -7)
        return str(int(wrong)) if wrong == int(wrong) else str(wrong)
    except ValueError:
        return gold + "0"


def build_poison_text(question: str, wrong_answer: str) -> str:
    return (
        f"Q: {question}\n"
        f"A: I worked through the problem and calculated the values step by step. "
        f"#### {wrong_answer}\n"
        f"Outcome: INCORRECT (gold={wrong_answer} was wrong)"
    )


def run_condition(cond_name: str, store, tasks, vectorizer, embed_fn):
    log = []
    correct_count = 0
    poison_exposure = 0
    for i, task in enumerate(tasks):
        q_emb = embed_fn(task["question"])
        retrieved = store.retrieve(q_emb, k=TOP_K)
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
        time.sleep(4.5)
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
    tasks = load_tasks(N_TASKS, SEED)
    vectorizer = TfidfVectorizer(max_features=500).fit([t["question"] for t in tasks])

    def embed_fn(text: str):
        return vectorizer.transform([text]).toarray()[0]

    poison_specs = []
    for i, task in enumerate(tasks):
        if i % POISON_EVERY == 0:
            wrong = make_wrong_answer(task["gold"])
            text = build_poison_text(task["question"], wrong)
            poison_specs.append((text, embed_fn(task["question"])))
    print(f"Seeding {len(poison_specs)} poisoned memories into each of {len(CONDITIONS)} conditions.\n")

    RESULTS_DIR.mkdir(exist_ok=True)
    results = {}

    for cond_name, flags in CONDITIONS.items():
        print(f"=== Running {cond_name} ({flags}) ===")
        store = AdaptiveTrustStore(embed_fn=embed_fn, summarize_fn=llm.summarize,
                                    label=cond_name, **flags)
        for text, emb in poison_specs:
            store.add(text, emb, correct=False, is_poison=True)
        log = run_condition(cond_name, store, tasks, vectorizer, embed_fn)
        results[cond_name] = {"log": log, "summary": summarize_log(log, store), "flags": flags}
        print()

    out_path = RESULTS_DIR / f"ablation_seed{SEED}_{int(time.time())}.json"
    out_path.write_text(json.dumps({"n_tasks": N_TASKS, "seed": SEED,
                                     "n_poison_seeded": len(poison_specs),
                                     "results": results}, indent=2))

    print("\n\n===== ABLATION TABLE (README Section 8) =====")
    print(f"{'Configuration':20s} {'Accuracy':>10s} {'PoisonExp':>10s} {'MemSize':>8s} {'#Compress':>10s} {'#Laundered':>11s}")
    for cond_name in CONDITIONS:
        s = results[cond_name]["summary"]
        print(f"{cond_name:20s} {s['overall_accuracy']:>10.3f} {s['total_poison_exposure']:>10d} "
              f"{s['final_memory_size']:>8d} {s['n_compressions']:>10d} {s['n_poison_laundered']:>11d}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
