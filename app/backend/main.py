"""Demo API for ATMC (Adaptive Trust-based Memory Consolidation) vs a static
memory baseline. Wraps the same engine used in the pilot experiments
(pilot/memory.py, pilot/stores.py, pilot/llm.py) behind a small REST API so
the exact poison-memory experiment can be driven live, interactively, for a
review/demo instead of only from a CLI log.

Both memory stores persist to a local SQLite file (db.py, app/backend/memory.db)
after every mutation, and are reloaded from it at startup -- a session's
memory now survives a server restart instead of resetting every time.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "pilot"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import db  # noqa: E402
import llm  # noqa: E402
from config import load_config  # noqa: E402
from embeddings import make_embedder  # noqa: E402
from factory import build_atmc, build_static  # noqa: E402
from tasks import build_poison_specs, load_tasks  # noqa: E402

app = FastAPI(title="ATMC Demo API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- config, a fixed question pool, and the embedder (all from configs/default.yaml) ----
CFG = load_config()
llm.configure(CFG)
TASKS = load_tasks(CFG, CFG.app.n_sample_tasks, CFG.app.sample_seed)
EMBEDDER = make_embedder(CFG, corpus=[t["question"] for t in TASKS])


def _embed(text: str):
    return EMBEDDER.embed(text)


# ---- persistent session state (backed by SQLite, see db.py) ----
db.init_db()
STORES = {
    "static": build_static(),
    "atmc": build_atmc(CFG, EMBEDDER, llm.summarize),
}
for _name, _store in STORES.items():
    db.load_store(_name, _store, EMBEDDER.name)

STATE = {
    "poisoned": any(m.is_poison for s in STORES.values() for m in s.items),
    "solved_ids": [],
}


def _save_all():
    for name, store in STORES.items():
        db.save_store(name, store, EMBEDDER.name)


class SolveRequest(BaseModel):
    task_id: int


def _run_store(store, task, use_memory: bool):
    q_emb = _embed(task["question"])
    retrieved = store.retrieve(q_emb, k=CFG.retrieval.top_k) if use_memory else []
    mem_texts = [m.text for m in retrieved]

    raw = llm.solve(task["question"], mem_texts)
    pred = llm.extract_final_number(raw)
    correct = llm.numbers_match(pred, task["gold"])

    used_mask = llm.extract_used_mask(raw, len(retrieved))
    store.update_from_outcome(retrieved, correct, used_mask=used_mask)

    reflection = (
        f"Q: {task['question']}\n"
        f"A: {raw.strip()[:400]}\n"
        f"Outcome: {'CORRECT' if correct else 'INCORRECT'} (gold={task['gold']})"
    )
    store.add(reflection, q_emb, correct)

    return {
        "answer_raw": raw,
        "pred": pred,
        "gold": task["gold"],
        "correct": correct,
        "memory_size": store.size(),
        "retrieved": [
            {
                "text": m.text,
                "trust": round(m.trust, 3),
                "hits": m.hits,
                "is_poison": m.is_poison,
                "used": used,
            }
            for m, used in zip(retrieved, used_mask)
        ],
    }


@app.get("/api/tasks")
def get_tasks():
    return [{"id": i, "question": t["question"]} for i, t in enumerate(TASKS)]


@app.get("/api/state")
def get_state():
    return {
        "poisoned": STATE["poisoned"],
        "solved_ids": STATE["solved_ids"],
        "static_memory_size": STORES["static"].size(),
        "atmc_memory_size": STORES["atmc"].size(),
        "atmc_compressions": STORES["atmc"].n_compressions,
    }


@app.post("/api/solve")
def solve(req: SolveRequest):
    task = TASKS[req.task_id]
    static_result = _run_store(STORES["static"], task, use_memory=True)
    time.sleep(CFG.model.request_interval_s)  # stay under the free-tier rate limit between the two calls
    atmc_result = _run_store(STORES["atmc"], task, use_memory=True)
    STATE["solved_ids"].append(req.task_id)
    _save_all()
    return {"question": task["question"], "static": static_result, "atmc": atmc_result}


@app.post("/api/poison")
def seed_poison():
    if STATE["poisoned"]:
        return {"seeded": 0, "already_poisoned": True}
    specs = build_poison_specs(TASKS, EMBEDDER, CFG.evaluation.poison_every)
    for text, emb in specs:
        STORES["static"].add(text, emb, correct=False, is_poison=True)
        STORES["atmc"].add(text, emb, correct=False, is_poison=True)
    STATE["poisoned"] = True
    _save_all()
    return {"seeded": len(specs), "already_poisoned": False}


@app.post("/api/reset")
def reset():
    STORES["static"] = build_static()
    STORES["atmc"] = build_atmc(CFG, EMBEDDER, llm.summarize)
    STATE["poisoned"] = False
    STATE["solved_ids"] = []
    db.clear_store("static")
    db.clear_store("atmc")
    return {"ok": True}


@app.get("/api/memory/{store_name}")
def get_memory(store_name: str):
    store = STORES.get(store_name)
    if store is None:
        return {"error": "unknown store"}
    return [
        {
            "text": m.text[:300],
            "trust": round(m.trust, 3),
            "hits": m.hits,
            "is_poison": m.is_poison,
            "correct": m.correct,
            "created_at_step": m.created_at_step,
        }
        for m in store.items
    ]


@app.get("/api/results")
def get_results():
    """Aggregated 3-seed poison-pilot results (from pilot/results/), for the
    results dashboard. Loaded fresh each call in case new runs are added."""
    results_dir = ROOT / "pilot" / "results"
    files = sorted(results_dir.glob("poison_pilot_seed*.json"))
    latest_per_seed = {}
    for f in files:
        d = json.load(open(f))
        latest_per_seed[d["seed"]] = d  # later files (higher timestamp) overwrite earlier
    runs = list(latest_per_seed.values())
    if not runs:
        return {"runs": []}

    def agg(cond):
        accs = [r[cond]["summary"]["overall_accuracy"] for r in runs]
        exposures = [r[cond]["summary"]["total_poison_exposure"] for r in runs]
        used = [r[cond]["summary"]["total_poison_used"] for r in runs]
        return {
            "accuracy_mean": sum(accs) / len(accs),
            "exposure_mean": sum(exposures) / len(exposures),
            "used_mean": sum(used) / len(used),
            "runs": {"accuracy": accs, "exposure": exposures, "used": used},
        }

    return {
        "n_seeds": len(runs),
        "seeds": [r["seed"] for r in runs],
        "static": agg("static"),
        "atmc": agg("atmc"),
    }


@app.get("/api/ablation")
def get_ablation():
    """Latest 4-way ablation study result (Full System / w/o Trust /
    w/o Forgetting / w/o Compression), from pilot/run_ablation.py."""
    results_dir = ROOT / "pilot" / "results"
    files = sorted(results_dir.glob("ablation_seed*.json"), key=lambda f: f.stat().st_mtime)
    if not files:
        return {"conditions": []}
    latest = json.load(open(files[-1]))
    conditions = []
    for name, data in latest["results"].items():
        conditions.append({
            "name": name,
            "flags": data["flags"],
            **data["summary"],
        })
    return {
        "seed": latest["seed"],
        "n_tasks": latest["n_tasks"],
        "n_poison_seeded": latest["n_poison_seeded"],
        "conditions": conditions,
    }
