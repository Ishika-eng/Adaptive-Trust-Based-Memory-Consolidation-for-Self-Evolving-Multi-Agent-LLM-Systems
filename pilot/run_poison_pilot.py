"""Poisoned-memory pilot: directly probes H1 (trust scoring reduces the
persistence and retrieval of incorrect memories) instead of hoping the model
makes organic mistakes on GSM8K -- the accuracy pilot showed the model is
too strong for that (near-ceiling on both conditions).

Setup: seed BOTH stores with the identical set of synthetic, deliberately
WRONG reflections (self-matching a subset of the real questions, so they're
guaranteed to be near-perfectly similar when that same question comes up
live -- simulating "a checker mistake let a bad reflection into memory").
Both stores start from the exact same contaminated state. Then run the same
40-task live sequence on top of that and measure whether trust-aware
retrieval + citation-gated trust updates let AdaptiveTrustStore (ATMC)
recover from the contamination faster than the static baseline, which has
no mechanism to ever downweight or forget a bad memory once written.

Credit assignment: a memory's trust only moves if the model's own "USED:"
citation says it actually relied on that memory for this task (see
llm.extract_used_mask) -- crediting/blaming every retrieved memory for the
overall task outcome regardless of use was a known gap in an earlier version
of this pilot (a poisoned memory that got retrieved but ignored still had
its trust rewarded whenever the model succeeded anyway).
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
SEED = int(os.environ.get("PILOT_SEED", "42"))
POISON_EVERY = 3  # poison every 3rd task -> ~13 poisoned seeds out of 40
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


def make_wrong_answer(gold: str) -> str:
    try:
        g = float(gold.replace(",", ""))
        wrong = g + (7 if g >= 0 else -7)
        return str(int(wrong)) if wrong == int(wrong) else str(wrong)
    except ValueError:
        return gold + "0"  # fallback for non-numeric gold (shouldn't happen for GSM8K)


def build_poison_text(question: str, wrong_answer: str) -> str:
    return (
        f"Q: {question}\n"
        f"A: I worked through the problem and calculated the values step by step. "
        f"#### {wrong_answer}\n"
        f"Outcome: INCORRECT (gold={wrong_answer} was wrong)"
    )


def seed_poison(store, poison_specs, vectorizer, is_atmc: bool):
    for text, embedding in poison_specs:
        # correct=False: an honest checker did label these wrong at write time.
        # The interesting question is whether ongoing operation (retrieval +
        # outcome-driven trust decay) purges them faster than static's
        # never-forget baseline, not whether the initial gate blocks them.
        store.add(text, embedding, correct=False, is_poison=True)


def run_condition(store, tasks, vectorizer, poison_texts: set, label: str):
    log = []
    correct_count = 0
    poison_exposure = 0  # cumulative count of poison memories retrieved into context
    for i, task in enumerate(tasks):
        q_emb = vectorizer.transform([task["question"]]).toarray()[0]
        retrieved = store.retrieve(q_emb, k=TOP_K)
        mem_texts = [m.text for m in retrieved]
        n_poison_retrieved = sum(1 for m in retrieved if m.is_poison)
        poison_exposure += n_poison_retrieved

        raw = llm.solve(task["question"], mem_texts)
        pred = llm.extract_final_number(raw)
        correct = llm.numbers_match(pred, task["gold"])
        correct_count += int(correct)

        used_mask = llm.extract_used_mask(raw, len(retrieved))
        n_poison_used = sum(1 for m, used in zip(retrieved, used_mask) if used and m.is_poison)
        store.update_from_outcome(retrieved, correct, used_mask=used_mask)

        reflection = (
            f"Q: {task['question']}\n"
            f"A: {raw.strip()[:400]}\n"
            f"Outcome: {'CORRECT' if correct else 'INCORRECT'} (gold={task['gold']})"
        )
        store.add(reflection, q_emb, correct)

        poison_remaining = sum(1 for m in store.items if m.is_poison)

        log.append({
            "idx": i,
            "correct": correct,
            "n_poison_retrieved": n_poison_retrieved,
            "n_poison_used": n_poison_used,
            "poison_exposure_cumulative": poison_exposure,
            "poison_remaining_in_store": poison_remaining,
            "memory_size": store.size(),
            "running_acc": correct_count / (i + 1),
        })
        print(f"[{label}] {i+1}/{len(tasks)} correct={correct} poison_hit={n_poison_retrieved>0} "
              f"poison_used={n_poison_used>0} poison_remaining={poison_remaining} "
              f"running_acc={correct_count/(i+1):.3f}")
        time.sleep(4.5)
    return log


def summarize(log, label):
    n = len(log)
    acc = sum(x["correct"] for x in log) / n
    total_poison_exposure = log[-1]["poison_exposure_cumulative"]
    total_poison_used = sum(x["n_poison_used"] for x in log)
    return {
        "label": label,
        "overall_accuracy": acc,
        "total_poison_exposure": total_poison_exposure,
        "total_poison_used": total_poison_used,
        "poison_remaining_final": log[-1]["poison_remaining_in_store"],
        "final_memory_size": log[-1]["memory_size"],
    }


def main():
    tasks = load_tasks(N_TASKS, SEED)
    vectorizer = TfidfVectorizer(max_features=500).fit([t["question"] for t in tasks])

    RESULTS_DIR.mkdir(exist_ok=True)

    poison_specs = []
    poison_texts = set()
    for i, task in enumerate(tasks):
        if i % POISON_EVERY == 0:
            wrong = make_wrong_answer(task["gold"])
            text = build_poison_text(task["question"], wrong)
            emb = vectorizer.transform([task["question"]]).toarray()[0]
            poison_specs.append((text, emb))
            poison_texts.add(text)
    print(f"Seeding {len(poison_specs)} poisoned memories into both stores.\n")

    print("=== Running STATIC baseline (poisoned) ===")
    static_store = StaticMemoryStore()
    seed_poison(static_store, poison_specs, vectorizer, is_atmc=False)
    static_log = run_condition(static_store, tasks, vectorizer, poison_texts, "static")

    print("\n=== Running ATMC (poisoned) ===")
    embed_fn = lambda text: vectorizer.transform([text]).toarray()[0]
    atmc_store = AdaptiveTrustStore(embed_fn=embed_fn, summarize_fn=llm.summarize)
    seed_poison(atmc_store, poison_specs, vectorizer, is_atmc=True)
    atmc_log = run_condition(atmc_store, tasks, vectorizer, poison_texts, "atmc")

    results = {
        "n_tasks": N_TASKS,
        "seed": SEED,
        "n_poison_seeded": len(poison_specs),
        "static": {"log": static_log, "summary": summarize(static_log, "static")},
        "atmc": {"log": atmc_log, "summary": summarize(atmc_log, "atmc")},
    }
    out_path = RESULTS_DIR / f"poison_pilot_seed{SEED}_{int(time.time())}.json"
    out_path.write_text(json.dumps(results, indent=2))

    print("\n\n===== POISON PILOT SUMMARY =====")
    print(f"Poisoned memories seeded per store: {len(poison_specs)}")
    for cond in ("static", "atmc"):
        s = results[cond]["summary"]
        print(f"{cond:10s}  overall_acc={s['overall_accuracy']:.3f}  "
              f"total_poison_exposure={s['total_poison_exposure']}  "
              f"total_poison_used={s['total_poison_used']}  "
              f"poison_remaining_final={s['poison_remaining_final']}/{len(poison_specs)}  "
              f"final_mem_size={s['final_memory_size']}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
