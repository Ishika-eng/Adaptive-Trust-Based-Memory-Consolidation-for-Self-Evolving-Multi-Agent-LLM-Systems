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
from autoverify import AutoVerifier  # noqa: E402
from domains.base import Verdict  # noqa: E402
from strategy import BASELINE, STRATEGIES, StrategyBook, label as category_label  # noqa: E402
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
    explain = list(getattr(result, "explain", []) or [])
    return [
        {"text": m.text, "trust": round(m.trust, 3), "hits": m.hits, "is_poison": m.is_poison, "used": u,
         "uid": getattr(m, "uid", ""), "source": getattr(m, "source", "tests"),
         "successes": getattr(m, "successes", 0), "failures": getattr(m, "failures", 0),
         "why": explain[i] if i < len(explain) else None}
        for i, (m, u) in enumerate(zip(result.retrieved, result.used_mask))
    ]


def _quota_guard(fn):
    try:
        return fn()
    except llm.QuotaExhausted as e:
        raise HTTPException(429, f"Gemini daily quota exhausted — try again later. {str(e)[:120]}")
    except HTTPException:
        raise
    except Exception as e:
        # An uncaught error would reach the browser as an opaque "Failed to fetch" (no CORS headers).
        raise HTTPException(502, f"{type(e).__name__}: {str(e)[:300]}")


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
# 3. the chat coding assistant: the user's own questions
# ---------------------------------------------------------------------------
# A user task usually has no tests, so there is no ground truth. With tests, the sandbox
# verifies the code and memory learns at once. Without them the answer is "unverified" and
# memory learns only when the user says whether it worked (the user is the checker).
#
# Messages in one conversation see the last few exchanges (follow-ups). The assistant also
# chooses its own next practice task with the weakness curriculum: a benchmark task close to
# something it got wrong, solved against real tests so memory learns verified outcomes.
def _build_chat_agent() -> Agent:
    cfg = _assistant_cfg()
    store = build_atmc(cfg, EMBEDDER, _summarize)
    db.load_store("agent_chat", store, EMBEDDER.name)
    pool = DOMAINS["code"].load_tasks(cfg, cfg.app.assistant_pool, cfg.app.sample_seed)
    agent = build_agent(cfg, store, EMBEDDER, DOMAINS["code"], pool, seed=cfg.app.sample_seed)
    agent.history = db.load_history("chat") if store.size() else []
    return agent


def _initial_chat() -> dict:
    turns = db.load_turns()
    return {"agent": _build_chat_agent(), "turns": turns, "pending": {}, "learned": {}, "suggestion": None,
            "conv": max([t.get("conversation", 1) for t in turns], default=1)}


VERIFIER = AutoVerifier(DOMAINS["code"], CFG)
_SC = CFG.to_dict().get("strategy", {"enabled": False})
BOOK = StrategyBook(EMBEDDER, min_trials=_SC.get("min_trials", 5), explore=_SC.get("explore", 0.1),
                    rng=__import__("random").Random(CFG.app.sample_seed))
BOOK.events = db.load_events()
CHAT = _initial_chat()
CHAT["learned"] = {}


def _save_book():
    db.save_events(BOOK.events)
MAX_PROMPT_CHARS = 4000
CONTEXT_TURNS = 3


class AskRequest(BaseModel):
    prompt: str
    tests: str = ""
    autocheck: bool = True     # with no tests: generate tests and alternative solutions to check the answer


class DisputeRequest(BaseModel):
    id: str
    comment: str = ""


class FeedbackRequest(BaseModel):
    id: str
    correct: bool
    comment: str = ""


class PractiseRequest(BaseModel):
    task_id: str


def _outcome_line(t: dict) -> str:
    if t["status"] == "passed":
        return "the tests passed"
    if t["status"] == "failed":
        return f"it failed ({t['stage'].replace('_', ' ')}: {t['detail']})"
    fb = t.get("feedback")
    if fb:
        return "the user confirmed it worked" if fb["correct"] else \
            "the user said it did NOT work" + (f": {fb['comment']}" if fb["comment"] else "")
    return "not verified"


def _conversation_context() -> tuple[str, str]:
    """(context text, the earlier user request) from the last few exchanges of this conversation."""
    prev = [t for t in CHAT["turns"] if t.get("conversation") == CHAT["conv"] and not t.get("practice")]
    prev = prev[-CONTEXT_TURNS:]
    if not prev:
        return "", ""
    parts = [f"User: {t['prompt'][:600]}\nYour answer (code):\n```python\n{t['code'][:1500]}\n```\n"
             f"Outcome: {_outcome_line(t)}" for t in prev]
    return "\n\n".join(parts), prev[-1]["prompt"]


def _turn(result, turn_id: str, prompt: str, tests: str = "", practice: bool = False, strategy: dict | None = None) -> dict:
    v, domain = result.verdict, DOMAINS["code"]
    status = "unverified" if not v.verified else ("passed" if v.correct else "failed")
    if v.verified and v.evidence == "generated":
        status = "auto_verified"
    return {
        "id": turn_id, "conversation": CHAT["conv"], "practice": practice,
        "autocheck": result.autocheck.to_json() if result.autocheck else None,
        "strategy": strategy, "filtered": getattr(result, "filtered", 0),
        "prompt": prompt.strip(), "tests": tests.strip(),
        "explanation": domain.explanation(result.raw), "code": result.answer or "",
        "plan": result.plan, "retrieved": _retrieved_json(result),
        "status": status, "stage": v.stage, "detail": v.detail, "output": v.detail if status == "unverified" else "",
        "lesson": result.lesson, "feedback": None,
        "cost": {"tokens_in": result.tokens_in, "tokens_out": result.tokens_out,
                 "retrieval_ms": round(result.retrieval_ms, 2)},
        "memory_size": result.memory_size,
    }


def _save_chat():
    db.save_store("agent_chat", CHAT["agent"].store, EMBEDDER.name)
    db.save_history("chat", CHAT["agent"].history)


def _pick_strategy(ref: str, text: str, mode: str) -> dict | None:
    """Choose how to solve this task and log the attempt (its outcome is filled in once it is known)."""
    if not _SC.get("enabled"):
        return None
    category = BOOK.classify(text)
    name, why = BOOK.choose(category, mode)
    BOOK.start(ref, category, name, mode)
    _save_book()
    return {"name": name, "description": STRATEGIES[name], "category": category,
            "category_label": category_label(category), "reason": why, "mode": mode}


def _resolve_strategy(ref: str, success: bool) -> None:
    if _SC.get("enabled"):
        BOOK.resolve(ref, success)
        _save_book()


@app.post("/api/chat/ask")
def chat_ask(req: AskRequest):
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(400, "Type a coding question first.")
    if len(prompt) > MAX_PROMPT_CHARS or len(req.tests) > MAX_PROMPT_CHARS:
        raise HTTPException(400, f"Keep the question and tests under {MAX_PROMPT_CHARS} characters.")
    with LOCK:
        agent = CHAT["agent"]
        turn_id = f"t{len(CHAT['turns']) + 1}"
        context, previous = _conversation_context()
        task = DOMAINS["code"].user_task(prompt, req.tests, idx=len(CHAT["turns"]) + 1,
                                         context=context, previous=previous)
        strategy = _pick_strategy(turn_id, task["question"], "serve")
        result = _quota_guard(lambda: agent.attempt(task, strategy["name"] if strategy else None))
        if result.verdict.stage == "unverified" and req.autocheck and CFG.autoverify.enabled:
            before = llm.usage_snapshot()
            check = _quota_guard(lambda: VERIFIER.run(task, result.answer))
            after = llm.usage_snapshot()
            result.tokens_in += after["prompt_tokens"] - before["prompt_tokens"]
            result.tokens_out += after["output_tokens"] - before["output_tokens"]
            result.autocheck = check
            if check.status in ("verified", "switched"):
                result.answer = check.answer
                result.verdict = Verdict(True, "ok", "", evidence="generated")
        if result.verdict.verified:
            _quota_guard(lambda: agent.learn(result))
            if result.verdict.evidence == "generated":
                CHAT["learned"][turn_id] = result            # kept so the user can dispute it later
            _resolve_strategy(turn_id, result.verdict.correct)
            _save_chat()
        else:
            CHAT["pending"][turn_id] = result
        turn = _turn(result, turn_id, prompt, req.tests, strategy=strategy)
        turn["followup"] = bool(context)
        CHAT["turns"].append(turn)
        db.save_turn(turn)
    return turn


@app.post("/api/chat/new")
def chat_new():
    """Start a fresh conversation: earlier messages stop being context, but memory is kept."""
    with LOCK:
        CHAT["conv"] += 1
    return {"conversation": CHAT["conv"]}


@app.post("/api/chat/dispute")
def chat_dispute(req: DisputeRequest):
    """The user says an auto-verified answer is actually wrong. Generated tests are weaker evidence than real
    ones, so the user's word overrides them: the notes it relied on are penalised and its own note is retired."""
    with LOCK:
        turn = next((t for t in CHAT["turns"] if t["id"] == req.id), None)
        if turn is None:
            raise HTTPException(404, "unknown message")
        if turn["status"] != "auto_verified":
            raise HTTPException(409, "only auto-verified answers can be disputed")
        result = CHAT["learned"].pop(req.id, None)
        if result is None:
            raise HTTPException(410, "this answer is from before the server restarted and can no longer be disputed")
        agent = CHAT["agent"]
        agent.store.update_from_outcome(result.retrieved, False, used_mask=result.used_mask)
        for m in agent.store.items:
            if m.text == result.note:
                m.correct, m.trust = False, 0.0                 # retired: below the verification gate
        verdict = Verdict(False, "user_rejected", req.comment.strip()[:300])
        agent.store.add(agent.domain.reflection(result.task, result.raw, result.answer, verdict), result.q_emb, False)
        turn.update(status="disputed", feedback={"correct": False, "comment": req.comment.strip()},
                    memory_size=agent.store.size())
        _resolve_strategy(req.id, False)
        _save_chat()
        db.save_turn(turn)
    return turn


@app.post("/api/chat/feedback")
def chat_feedback(req: FeedbackRequest):
    """The user's verdict on an unverified answer -- this is what lets memory learn from it."""
    with LOCK:
        turn = next((t for t in CHAT["turns"] if t["id"] == req.id), None)
        if turn is None:
            raise HTTPException(404, "unknown message")
        if turn["feedback"] is not None or turn["status"] != "unverified":
            raise HTTPException(409, "this answer was already verified")
        result = CHAT["pending"].pop(req.id, None)
        if result is None:
            raise HTTPException(410, "this answer is from before the server restarted and can no longer be graded")
        _quota_guard(lambda: CHAT["agent"].learn(result, correct=req.correct, feedback=req.comment))
        turn.update(feedback={"correct": req.correct, "comment": req.comment.strip()},
                    lesson=result.lesson, memory_size=result.memory_size)
        _resolve_strategy(req.id, req.correct)
        _save_chat()
        db.save_turn(turn)
    return turn


def _suggest(agent: Agent) -> dict | None:
    """The assistant's own choice of what to practise next (weakness curriculum), with the reason."""
    try:
        task = agent.next_task()
    except StopIteration:
        return None
    mode = getattr(agent.curriculum, "last_mode", None)
    note = agent.curriculum.related_failure(task, agent.store, EMBEDDER) if mode == "weakness" else None
    if note is not None:
        title = note.text.split("\n")[0].replace("Task: ", "")[:90]
        reason = f"This is close to a task I got wrong before: \u201c{title}\u201d."
    elif mode == "explore":
        reason = "I'm trying something different so I don't get stuck practising only one area."
    else:
        reason = "I haven't made a mistake yet, so I'm trying something new."
    return {"id": task["id"], "question": task["question"][:400], "reason": reason}


@app.get("/api/chat/suggest")
def chat_suggest(refresh: bool = False):
    with LOCK:
        if refresh or CHAT["suggestion"] is None or CHAT["suggestion"]["id"] in CHAT["agent"].done_ids:
            CHAT["suggestion"] = _suggest(CHAT["agent"])
        return CHAT["suggestion"] or {"id": None}


@app.post("/api/chat/practise")
def chat_practise(req: PractiseRequest):
    """The assistant works on a task it chose itself. Benchmark tasks come with real tests, so the
    result is verified and memory learns from it straight away."""
    with LOCK:
        agent = CHAT["agent"]
        task = next((t for t in agent.pool if t["id"] == req.task_id), None)
        if task is None or task["id"] in agent.done_ids:
            raise HTTPException(404, "that practice task is not available")
        turn_id = f"t{len(CHAT['turns']) + 1}"
        strategy = _pick_strategy(turn_id, task["question"], "practise")
        result = _quota_guard(lambda: agent.attempt(task, strategy["name"] if strategy else None))
        _quota_guard(lambda: agent.learn(result))
        _resolve_strategy(turn_id, result.verdict.correct)
        _save_chat()
        turn = _turn(result, turn_id, task["question"], practice=True, strategy=strategy)
        turn["followup"] = False
        CHAT["turns"].append(turn)
        db.save_turn(turn)
        CHAT["suggestion"] = None
    return turn


@app.get("/api/chat/strategies")
def chat_strategies():
    """The strategy scoreboard: evidence per problem type, who is champion, and the latest attempts."""
    prompts = {t["id"]: t["prompt"] for t in CHAT["turns"]}
    recent = [{"ref": e["ref"], "category": category_label(e["category"]), "strategy": e["strategy"], "mode": e["mode"],
               "success": e["success"], "prompt": prompts.get(e["ref"], "")[:90]} for e in BOOK.events[-20:]][::-1]
    return {"enabled": bool(_SC.get("enabled")), "baseline": BASELINE, "min_trials": BOOK.min_trials,
            "descriptions": STRATEGIES, "categories": BOOK.table(), "events": recent, "total": len(BOOK.events)}


@app.get("/api/chat/history")
def chat_history():
    return CHAT["turns"]


@app.get("/api/chat/summary")
def chat_summary():
    store = CHAT["agent"].store
    turns = [t for t in CHAT["turns"] if not t.get("practice")]
    practice = [t for t in CHAT["turns"] if t.get("practice")]
    good = sum(t["status"] in ("passed", "auto_verified") or (t["feedback"] or {}).get("correct", False) for t in turns)
    return {
        "asked": len(turns), "solved": good,
        "practised": len(practice), "practised_passed": sum(t["status"] == "passed" for t in practice),
        "awaiting_feedback": sum(t["status"] == "unverified" and t["feedback"] is None for t in turns),
        "memory_size": store.size(), "n_compressions": store.n_compressions,
        "conversation": CHAT["conv"], "isolation": ISOLATION,
    }


@app.post("/api/chat/reset")
def chat_reset():
    with LOCK:
        db.clear_store("agent_chat")
        db.clear_history("chat")
        db.clear_turns()
        BOOK.events = []
        db.clear_events()
        CHAT.clear()
        CHAT.update(_initial_chat())
    return {"ok": True}


# ---------------------------------------------------------------------------
# memory inspection and results
# ---------------------------------------------------------------------------
def _store(name: str):
    if name in COMPARE:
        return COMPARE[name].store
    if name in ASSISTANTS:
        return ASSISTANTS[name].store
    if name == "chat":
        return CHAT["agent"].store
    return None


@app.get("/api/memory/{store_name}")
def get_memory(store_name: str):
    """store_name: static | atmc (side-by-side demo) | math | code (assistant memories)."""
    store = _store(store_name)
    if store is None:
        raise HTTPException(404, f"unknown store {store_name!r}")
    return [
        {"uid": getattr(m, "uid", ""), "text": m.text[:1500], "trust": round(m.trust, 3), "hits": m.hits,
         "is_poison": m.is_poison, "correct": m.correct, "created_at_step": m.created_at_step,
         "source": getattr(m, "source", "tests"), "successes": getattr(m, "successes", 0),
         "failures": getattr(m, "failures", 0), "last_used_step": getattr(m, "last_used_step", None),
         "version": getattr(m, "version", 1), "pinned": getattr(m, "pinned", False),
         "gated": store.name == "atmc" and m.trust < store.t_min}
        for m in store.items
    ]


class PinRequest(BaseModel):
    uid: str
    pinned: bool = True


class ImportedNote(BaseModel):
    text: str
    embedding: list[float]
    correct: bool | None = None
    is_poison: bool = False
    successes: int = 0
    failures: int = 0
    version: int = 1


class ImportRequest(BaseModel):
    embedder: str
    items: list[ImportedNote]


MAX_IMPORT_NOTES = 2000


@app.post("/api/memory/chat/pin")
def memory_pin(req: PinRequest):
    with LOCK:
        if not CHAT["agent"].store.set_pinned(req.uid, req.pinned):
            raise HTTPException(404, "no such note")
        _save_chat()
    return {"ok": True}


@app.delete("/api/memory/chat/{uid}")
def memory_delete(uid: str):
    with LOCK:
        if not CHAT["agent"].store.remove(uid):
            raise HTTPException(404, "no such note")
        _save_chat()
    return {"ok": True, "memory_size": CHAT["agent"].store.size()}


@app.get("/api/memory/chat/export")
def memory_export():
    """Everything needed to restore this memory elsewhere (same embedder only: vectors are not portable)."""
    store = CHAT["agent"].store
    return {"format": "atmc-memory/1", "embedder": EMBEDDER.name, "items": [
        {"text": m.text, "embedding": [round(float(x), 6) for x in m.embedding], "correct": m.correct,
         "is_poison": m.is_poison, "trust": round(m.trust, 3), "source": m.source,
         "successes": m.successes, "failures": m.failures, "version": m.version, "pinned": m.pinned}
        for m in store.items]}


@app.post("/api/memory/chat/import")
def memory_import(req: ImportRequest):
    if req.embedder != EMBEDDER.name:
        raise HTTPException(400, f"These notes were made with the embedder {req.embedder!r}; this server uses "
                                 f"{EMBEDDER.name!r}, and their vectors are not compatible.")
    if len(req.items) > MAX_IMPORT_NOTES:
        raise HTTPException(400, f"Import at most {MAX_IMPORT_NOTES} notes at a time.")
    dim = len(EMBEDDER.embed("dimension probe"))
    with LOCK:
        added = skipped = 0
        for n in req.items:
            if len(n.embedding) != dim or not n.text.strip() or len(n.text) > 6000:
                raise HTTPException(400, "Malformed note in the import file.")
            ok = CHAT["agent"].store.import_item(n.text, n.embedding, n.correct, is_poison=n.is_poison,
                                                 successes=max(0, n.successes), failures=max(0, n.failures),
                                                 version=max(1, n.version))
            added += ok
            skipped += not ok
        _save_chat()
    return {"added": added, "skipped_duplicates": skipped, "memory_size": CHAT["agent"].store.size()}


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
