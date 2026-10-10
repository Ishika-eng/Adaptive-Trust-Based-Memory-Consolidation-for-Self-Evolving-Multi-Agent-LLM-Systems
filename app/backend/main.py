"""API for ATMC (Adaptive Trust-based Memory Consolidation).

Two things are served, both running the same `agent.Agent` loop the experiments use:

  1. The side-by-side demo: a static memory and an ATMC memory solve the same math
     problems, so the effect of trust-gated memory can be watched live.
  2. The self-evolving assistant: one ATMC agent per domain (math or code) that
     plans, retrieves, solves, runs its code against real tests in a sandbox,
     reflects on failures, updates trust, and chooses its own next task.

Every memory store -- and the assistant's step history -- is persisted to SQLite
(db.py) after each mutation and reloaded at startup, so a restart loses nothing.
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "pilot"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import db  # noqa: E402
import llm  # noqa: E402
import report  # noqa: E402
import sandbox  # noqa: E402
from agent import Agent, build_agent  # noqa: E402
from config import Config, load_config  # noqa: E402
from domains import get_domain  # noqa: E402
from embeddings import make_embedder  # noqa: E402
from factory import build_atmc, build_static  # noqa: E402

app = FastAPI(title="ATMC API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

CFG = load_config()
llm.configure(CFG)
LOCK = threading.RLock()          # steps mutate shared memory; serialise them

DOMAINS = {name: get_domain(name, CFG) for name in ("math", "code")}
EMBEDDER = make_embedder(CFG, corpus=[t["question"] for t in DOMAINS["math"].load_tasks(CFG, 200, 0)])
db.init_db()


ISOLATION = sandbox.run_python("pass").isolation     # checked once, not on every poll


def _summarize(texts):
    return llm.summarize(texts)


# ---------------------------------------------------------------------------
# 1. side-by-side demo (math): static vs ATMC
# ---------------------------------------------------------------------------
TASKS = DOMAINS["math"].load_tasks(CFG, CFG.app.n_sample_tasks, CFG.app.sample_seed)


def _fixed_cfg(**agent) -> Config:
    d = CFG.to_dict()
    d["agent"] = {**d["agent"], "curriculum": "fixed", "use_planner": False, "use_reflection": False, **agent}
    return Config(d)


def _compare_agent(name: str) -> Agent:
    store = build_static() if name == "static" else build_atmc(CFG, EMBEDDER, _summarize)
    db.load_store(name, store, EMBEDDER.name)
    return build_agent(_fixed_cfg(), store, EMBEDDER, DOMAINS["math"], TASKS)


COMPARE = {name: _compare_agent(name) for name in ("static", "atmc")}
STATE = {"poisoned": any(m.is_poison for a in COMPARE.values() for m in a.store.items), "solved_ids": []}


def _retrieved_json(result) -> list[dict]:
    return [
        {"text": m.text, "trust": round(m.trust, 3), "hits": m.hits, "is_poison": m.is_poison, "used": u}
        for m, u in zip(result.retrieved, result.used_mask)
    ]


def _quota_guard(fn):
    try:
        return fn()
    except llm.QuotaExhausted as e:
        raise HTTPException(429, f"Gemini daily quota exhausted — try again later. {str(e)[:120]}")


def _save_compare():
    for name, a in COMPARE.items():
        db.save_store(name, a.store, EMBEDDER.name)


class SolveRequest(BaseModel):
    task_id: int


@app.get("/api/tasks")
def get_tasks():
    return [{"id": i, "question": t["question"]} for i, t in enumerate(TASKS)]


@app.get("/api/state")
def get_state():
    return {
        "poisoned": STATE["poisoned"],
        "solved_ids": STATE["solved_ids"],
        "static_memory_size": COMPARE["static"].store.size(),
        "atmc_memory_size": COMPARE["atmc"].store.size(),
        "atmc_compressions": COMPARE["atmc"].store.n_compressions,
    }


@app.post("/api/solve")
def solve(req: SolveRequest):
    task = TASKS[req.task_id]
    out = {"question": task["question"]}
    with LOCK:
        for name, a in COMPARE.items():
            r = _quota_guard(lambda a=a: a.step(task))
            out[name] = {
                "answer_raw": r.raw, "pred": r.answer, "gold": task["gold"], "correct": r.correct,
                "memory_size": r.memory_size, "retrieved": _retrieved_json(r),
            }
        STATE["solved_ids"].append(req.task_id)
        _save_compare()
    return out


@app.post("/api/poison")
def seed_poison():
    with LOCK:
        if STATE["poisoned"]:
            return {"seeded": 0, "already_poisoned": True}
        specs = DOMAINS["math"].poison_specs(TASKS, EMBEDDER, CFG.evaluation.poison_every)
        for text, emb in specs:
            for a in COMPARE.values():
                a.store.add(text, emb, correct=False, is_poison=True)
        STATE["poisoned"] = True
        _save_compare()
    return {"seeded": len(specs), "already_poisoned": False}


@app.post("/api/reset")
def reset():
    with LOCK:
        for name in COMPARE:
            db.clear_store(name)
        COMPARE["static"] = build_agent(_fixed_cfg(), build_static(), EMBEDDER, DOMAINS["math"], TASKS)
        COMPARE["atmc"] = build_agent(_fixed_cfg(), build_atmc(CFG, EMBEDDER, _summarize), EMBEDDER,
                                      DOMAINS["math"], TASKS)
        STATE.update(poisoned=False, solved_ids=[])
    return {"ok": True}


# ---------------------------------------------------------------------------
# 2. the self-evolving assistant (one agent per domain)
# ---------------------------------------------------------------------------
def _assistant_cfg() -> Config:
    d = CFG.to_dict()
    d["agent"] = {**d["agent"], **CFG.app.assistant.to_dict()}
    return Config(d)


def _build_assistant(domain_name: str) -> Agent:
    cfg = _assistant_cfg()
    domain = DOMAINS[domain_name]
    pool = domain.load_tasks(cfg, cfg.app.assistant_pool, cfg.app.sample_seed)
    store = build_atmc(cfg, EMBEDDER, _summarize)
    key = f"agent_{domain_name}"
    db.load_store(key, store, EMBEDDER.name)
    agent = build_agent(cfg, store, EMBEDDER, domain, pool, seed=cfg.app.sample_seed)
    agent.history = db.load_history(domain_name) if store.size() else []
    return agent


ASSISTANTS = {name: _build_assistant(name) for name in DOMAINS}


def _assistant(domain: str) -> Agent:
    if domain not in ASSISTANTS:
        raise HTTPException(404, f"unknown domain {domain!r}; use one of {list(ASSISTANTS)}")
    return ASSISTANTS[domain]


def _save_assistant(domain: str):
    a = ASSISTANTS[domain]
    db.save_store(f"agent_{domain}", a.store, EMBEDDER.name)
    db.save_history(domain, a.history)


def _assistant_summary(domain: str) -> dict:
    a = _assistant(domain)
    hist = a.history
    curve, ok = [], 0
    for i, h in enumerate(hist, 1):
        ok += h["correct"]
        curve.append(round(ok / i, 3))
    return {
        "domain": domain,
        "steps": len(hist),
        "passed": sum(h["correct"] for h in hist),
        "pool_size": len(a.pool),
        "remaining": len(a.pool) - len(a.done_ids),
        "memory_size": a.store.size(),
        "n_compressions": a.store.n_compressions,
        "poisoned": any(m.is_poison for m in a.store.items),
        "learning_curve": curve,
        "history": hist[-30:],
        "options": {"planner": a.planner is not None, "reflection": a.reflector is not None,
                    "curriculum": a.cfg.agent.curriculum},
        "isolation": ISOLATION if domain == "code" else None,
    }


class StepRequest(BaseModel):
    domain: str = "code"
    task_id: str | None = None      # omit to let the agent's curriculum choose


@app.get("/api/agent/state")
def agent_state(domain: str = "code"):
    return _assistant_summary(domain)


@app.get("/api/agent/tasks")
def agent_tasks(domain: str = "code"):
    a = _assistant(domain)
    done = a.done_ids
    return [{"id": t["id"], "question": t["question"][:160], "done": t["id"] in done} for t in a.pool]


@app.post("/api/agent/step")
def agent_step(req: StepRequest):
    with LOCK:
        a = _assistant(req.domain)
        chosen_by = "you"
        task = None
        if req.task_id:
            task = next((t for t in a.pool if t["id"] == req.task_id), None)
            if task is None:
                raise HTTPException(404, f"task {req.task_id!r} is not in the pool")
        else:
            chosen_by = a.cfg.agent.curriculum
            try:
                task = a.next_task()
            except StopIteration:
                raise HTTPException(409, "the agent has attempted every task in its pool")
        r = _quota_guard(lambda: a.step(task))
        _save_assistant(req.domain)
    return {
        "chosen_by": chosen_by,
        "task": {"id": task["id"], "question": task["question"]},
        "plan": r.plan,
        "retrieved": _retrieved_json(r),
        "answer": r.answer,
        "verdict": {"correct": r.verdict.correct, "stage": r.verdict.stage, "detail": r.verdict.detail},
        "lesson": r.lesson,
        "note": r.note,
        "poison": {"in_context": r.n_poison_retrieved, "used": r.n_poison_used},
        "cost": {"tokens_in": r.tokens_in, "tokens_out": r.tokens_out, "retrieval_ms": round(r.retrieval_ms, 2)},
        "summary": _assistant_summary(req.domain),
    }


class DomainRequest(BaseModel):
    domain: str = "code"


@app.post("/api/agent/poison")
def agent_poison(req: DomainRequest):
    """Plant plausible-but-wrong past solutions (verified to fail their tests) in the assistant's memory."""
    with LOCK:
        a = _assistant(req.domain)
        if any(m.is_poison for m in a.store.items):
            return {"seeded": 0, "already_poisoned": True}
        specs = a.domain.poison_specs(a.pool[:30], EMBEDDER, CFG.evaluation.poison_every)
        for text, emb in specs:
            a.store.add(text, emb, correct=False, is_poison=True)
        _save_assistant(req.domain)
    return {"seeded": len(specs), "already_poisoned": False}


@app.post("/api/agent/reset")
def agent_reset(req: DomainRequest):
    with LOCK:
        _assistant(req.domain)
        db.clear_store(f"agent_{req.domain}")
        db.clear_history(req.domain)
        ASSISTANTS[req.domain] = _build_assistant(req.domain)
    return {"ok": True}


# ---------------------------------------------------------------------------
# memory inspection and results
# ---------------------------------------------------------------------------
def _store(name: str):
    if name in COMPARE:
        return COMPARE[name].store
    if name in ASSISTANTS:
        return ASSISTANTS[name].store
    return None


@app.get("/api/memory/{store_name}")
def get_memory(store_name: str):
    """store_name: static | atmc (side-by-side demo) | math | code (assistant memories)."""
    store = _store(store_name)
    if store is None:
        raise HTTPException(404, f"unknown store {store_name!r}")
    return [
        {"text": m.text[:300], "trust": round(m.trust, 3), "hits": m.hits, "is_poison": m.is_poison,
         "correct": m.correct, "created_at_step": m.created_at_step}
        for m in store.items
    ]


@app.get("/api/eval")
def eval_report(experiment: str | None = None, domain: str | None = None, ref: str | None = None):
    """Experiment results from pilot/experiment.py, with paired statistics.
    With no arguments, lists which experiments have saved runs."""
    if experiment is None:
        found = sorted({tuple(p.name.split("__")[:2]) for p in report.EVAL_DIR.glob("*__*__*.json")})
        return [{"experiment": e, "domain": d} for e, d in found]
    return report.build_report(experiment, domain or "math", ref)


@app.get("/api/results")
def get_results():
    """Earlier 3-seed poison pilot (pilot/results/*.json), kept for the original dashboard."""
    files = sorted((ROOT / "pilot" / "results").glob("poison_pilot_seed*.json"))
    latest_per_seed = {}
    for f in files:
        d = json.load(open(f))
        latest_per_seed[d["seed"]] = d
    runs = list(latest_per_seed.values())
    if not runs:
        return {"runs": []}

    def agg(cond):
        accs = [r[cond]["summary"]["overall_accuracy"] for r in runs]
        exposures = [r[cond]["summary"]["total_poison_exposure"] for r in runs]
        used = [r[cond]["summary"]["total_poison_used"] for r in runs]
        return {"accuracy_mean": sum(accs) / len(accs), "exposure_mean": sum(exposures) / len(exposures),
                "used_mean": sum(used) / len(used), "runs": {"accuracy": accs, "exposure": exposures, "used": used}}

    return {"n_seeds": len(runs), "seeds": [r["seed"] for r in runs], "static": agg("static"), "atmc": agg("atmc")}


@app.get("/api/ablation")
def get_ablation():
    """Earlier single-seed 4-way ablation (pilot/results/ablation_seed*.json)."""
    files = sorted((ROOT / "pilot" / "results").glob("ablation_seed*.json"), key=lambda f: f.stat().st_mtime)
    if not files:
        return {"conditions": []}
    latest = json.load(open(files[-1]))
    return {
        "seed": latest["seed"], "n_seeds": 1, "n_tasks": latest["n_tasks"], "n_poison_seeded": latest["n_poison_seeded"],
        "conditions": [{"name": n, "flags": d["flags"], **d["summary"]} for n, d in latest["results"].items()],
    }
