"""Minimal pilot: does trust-aware memory (AdaptiveTrustStore / ATMC) beat a
static memory baseline (StaticMemoryStore) on sequential GSM8K problem solving?

This is deliberately small in scope -- one seed, ~40 tasks, TF-IDF similarity
instead of a real embedding model, deterministic numeric checker instead of
an LLM judge -- to get a fast directional signal before investing in the
full pipeline (multi-agent orchestration, compression, multiple benchmarks,
multi-seed stats) described in the README.
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

from stores import StaticMemoryStore, AdaptiveTrustStore
import llm

N_TASKS = int(os.environ.get("PILOT_N_TASKS", "40"))
TOP_K = 3
SEED = 42
DATA_PATH = Path(__file__).resolve().parent / "data" / "gsm8k_test.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


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


def run_condition(store, tasks, vectorizer, label: str):
    log = []
    correct_count = 0
    for i, task in enumerate(tasks):
        q_emb = vectorizer.transform([task["question"]]).toarray()[0]
        retrieved = store.retrieve(q_emb, k=TOP_K)
        mem_texts = [m.text for m in retrieved]

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
            "pred": pred,
            "gold": task["gold"],
            "retrieved_n": len(retrieved),
            "memory_size": store.size(),
            "running_acc": correct_count / (i + 1),
        })
        print(f"[{label}] {i+1}/{len(tasks)} correct={correct} "
              f"running_acc={correct_count/(i+1):.3f} mem_size={store.size()}")
        time.sleep(4.5)  # stay under gemini-3.5-flash-lite free tier: 15 req/min
    return log


def summarize(log, label):
    n = len(log)
    first_half = log[: n // 2]
    second_half = log[n // 2 :]
    acc = sum(x["correct"] for x in log) / n
    acc1 = sum(x["correct"] for x in first_half) / max(len(first_half), 1)
    acc2 = sum(x["correct"] for x in second_half) / max(len(second_half), 1)
    return {
        "label": label,
        "overall_accuracy": acc,
        "first_half_accuracy": acc1,
        "second_half_accuracy": acc2,
        "final_memory_size": log[-1]["memory_size"] if log else 0,
    }


def main():
    tasks = load_tasks(N_TASKS, SEED)
    vectorizer = TfidfVectorizer(max_features=500).fit([t["question"] for t in tasks])

    RESULTS_DIR.mkdir(exist_ok=True)

    print("=== Running STATIC baseline ===")
    static_store = StaticMemoryStore()
    static_log = run_condition(static_store, tasks, vectorizer, "static")

    print("\n=== Running ATMC (proposed) ===")
    atmc_store = AdaptiveTrustStore()
    atmc_log = run_condition(atmc_store, tasks, vectorizer, "atmc")

    results = {
        "n_tasks": N_TASKS,
        "seed": SEED,
        "static": {"log": static_log, "summary": summarize(static_log, "static")},
        "atmc": {"log": atmc_log, "summary": summarize(atmc_log, "atmc")},
    }
    out_path = RESULTS_DIR / f"pilot_{int(time.time())}.json"
    out_path.write_text(json.dumps(results, indent=2))

    print("\n\n===== PILOT SUMMARY =====")
    for cond in ("static", "atmc"):
        s = results[cond]["summary"]
        print(f"{cond:10s}  overall={s['overall_accuracy']:.3f}  "
              f"1st_half={s['first_half_accuracy']:.3f}  "
              f"2nd_half={s['second_half_accuracy']:.3f}  "
              f"final_mem_size={s['final_memory_size']}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
