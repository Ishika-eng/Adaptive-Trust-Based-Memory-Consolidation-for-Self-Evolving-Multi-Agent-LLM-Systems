"""Learning Lab: the same stream of hidden-convention questions goes to two assistants -- one with NO memory and one
with the memory configuration that scored best in our experiments -- and every step is returned so the page can show
the model learning. Also lets the page switch between the local model (Ollama) and Gemini.

The "with memory" assistant is exactly the `ladder_trust` condition of docs/experiments (trust on, forgetting and merging
off, checker-verified failure notes at normal trust, no retrieval-frequency boost). Because the model runs at temperature
0 and responses are cached, a stream that was already run (e.g. seed 42) is replayed quickly; the sandbox, the checks and
the memory all still run for real, and each step reports how many model calls were served from the cache.
"""
from __future__ import annotations

import json
import threading
import urllib.request

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import llm
from agent import build_agent
from config import Config, load_config
from domains import get_domain
from factory import build_store

router = APIRouter(prefix="/api")
LOCK = threading.RLock()

LOCAL = {"provider": "ollama", "name": "qwen2.5-coder:7b-instruct", "request_interval_s": 0.0}
CLOUD = {"provider": "gemini"}            # name and pacing come from configs/default.yaml
_state = {"mode": None}                   # "local" | "cloud"; None until chosen


def _cfg(overrides: dict | None = None) -> Config:
    return load_config(overrides=overrides)


def _apply_mode(mode: str) -> None:
    over = dict(LOCAL if mode == "local" else CLOUD)
    cfg = _cfg({"model": over})
    llm.configure(cfg)
    _state["mode"] = mode
    _state["name"] = cfg.model.name


def _ollama_up() -> bool:
    try:
        with urllib.request.urlopen(f"{llm.BASE_URL}/api/tags", timeout=1.5) as r:
            return any(m.get("name", "").startswith("qwen2.5-coder:7b") for m in json.load(r).get("models", []))
    except Exception:
        return False


@router.get("/model")
def get_model():
    if _state["mode"] is None:
        _state["mode"] = "local" if llm.PROVIDER == "ollama" else "cloud"
        _state["name"] = llm.MODEL_NAME
    return {"mode": _state["mode"], "provider": llm.PROVIDER, "name": llm.MODEL_NAME,
            "local_available": _ollama_up(), "local_name": LOCAL["name"]}


class ModelRequest(BaseModel):
    mode: str                              # "local" | "cloud"


@router.post("/model")
def set_model(req: ModelRequest):
    if req.mode not in ("local", "cloud"):
        raise HTTPException(400, "mode must be 'local' or 'cloud'")
    if req.mode == "local" and not _ollama_up():
        raise HTTPException(409, f"The local model {LOCAL['name']} is not available. Start Ollama (`ollama serve`) "
                                 f"and make sure the model is installed (`ollama pull {LOCAL['name']}`).")
    with LOCK:
        _apply_mode(req.mode)
        SESSION.clear()                    # a running lab belongs to the previous model
    return get_model()


# --------------------------------------------------------------------------------------------------------------
class Lab:
    """One run: two agents, one shared task stream."""

    def __init__(self, seed: int, n: int, embedder, summarize):
        base = _cfg().to_dict()
        base["coding"] = {**base["coding"], "datasets": ["conventions"], "note_style": "evidence"}
        base["agent"] = {**base["agent"], "domain": "code", "curriculum": "fixed", "use_planner": False,
                         "use_reflection": False, "evidence_note_trust": True, "credit": "cited"}
        base["retrieval"] = {**base["retrieval"], "hit_boost": False}
        self.cfg = Config(base)
        self.domain = get_domain("code", self.cfg)
        self.seed, self.mode = seed, _state["mode"]
        tasks = self.domain.load_tasks(self.cfg, 110, seed)
        self.stream = tasks[:n]
        self.agents = {}
        for name, kind, pillars in (("no_memory", "none", {}),
                                    ("memory", "atmc", {"enable_forgetting": False, "enable_compression": False})):
            store = build_store(kind, self.cfg, embedder, summarize, **pillars)
            a = build_agent(self.cfg, store, embedder, self.domain, self.stream, seed)
            a.pool = self.stream
            self.agents[name] = a
        self.i = 0
        self.rows: list[dict] = []        # one per finished step
        self.model_label = llm.MODEL_NAME

    @property
    def done(self) -> bool:
        return self.i >= len(self.stream)

    def _agent_view(self, a, r) -> dict:
        expl = list(getattr(r, "explain", []) or [])
        notes = [{"text": m.text, "trust": round(m.trust, 2), "correct": m.correct,
                  "similarity": (expl[k].get("similarity") if k < len(expl) and expl[k] else None)}
                 for k, m in enumerate(r.retrieved)]
        return {"answer": r.answer, "correct": bool(r.verdict.correct), "stage": r.verdict.stage,
                "detail": r.verdict.detail, "notes_used": notes, "note_saved": r.note if r.learned else None,
                "memory_size": a.store.size()}

    def step(self) -> dict:
        task = self.stream[self.i]
        calls0 = llm.usage_snapshot()
        out = {}
        for name, a in self.agents.items():
            r = a.step(task)
            out[name] = self._agent_view(a, r)
        calls1 = llm.usage_snapshot()
        self.i += 1
        row = {"i": self.i, "task_id": task["id"], "family": task.get("family", ""), "role": task.get("role", ""),
               "question": task["question"], "no_memory": out["no_memory"], "memory": out["memory"],
               "model_calls": calls1["calls"] - calls0["calls"],
               "from_cache": calls1["cache_hits"] - calls0["cache_hits"]}
        self.rows.append(row)
        return row

    def summary(self) -> dict:
        def stats(key):
            ok, curve, per = 0, [], {}
            for k, r in enumerate(self.rows, 1):
                ok += r[key]["correct"]
                curve.append(round(ok / k, 4))
                f = per.setdefault(r["family"], [0, 0])
                f[0] += r[key]["correct"]
                f[1] += 1
            return {"passed": ok, "steps": len(self.rows), "curve": curve, "by_family": per}
        rescued = [r["i"] for r in self.rows if r["memory"]["correct"] and not r["no_memory"]["correct"]]
        hurt = [r["i"] for r in self.rows if r["no_memory"]["correct"] and not r["memory"]["correct"]]
        return {"seed": self.seed, "n": len(self.stream), "step": self.i, "done": self.done, "model": self.model_label,
                "mode": self.mode, "no_memory": stats("no_memory"), "memory": stats("memory"),
                "rescued": rescued, "hurt": hurt, "memory_size": self.agents["memory"].store.size(),
                "cache_hits": sum(r["from_cache"] for r in self.rows), "model_calls": sum(r["model_calls"] for r in self.rows)}


class _Session(dict):
    lab: Lab | None = None

    def clear(self):
        self.lab = None


SESSION = _Session()
_deps: dict = {}


def init(embedder, summarize) -> None:
    _deps.update(embedder=embedder, summarize=summarize)


def _guard(fn):
    try:
        return fn()
    except llm.QuotaExhausted as e:
        raise HTTPException(429, f"Gemini daily quota exhausted. {str(e)[:120]}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"{type(e).__name__}: {str(e)[:300]}")


class StartRequest(BaseModel):
    seed: int = 42
    n: int = 110


@router.post("/learn/start")
def start(req: StartRequest):
    if not 5 <= req.n <= 110:
        raise HTTPException(400, "n must be between 5 and 110")
    get_model()
    with LOCK:
        SESSION.lab = _guard(lambda: Lab(req.seed, req.n, _deps["embedder"], _deps["summarize"]))
        return SESSION.lab.summary()


@router.get("/learn/state")
def state():
    lab = SESSION.lab
    return lab.summary() if lab else {"idle": True}


@router.post("/learn/step")
def step():
    with LOCK:
        lab = SESSION.lab
        if lab is None:
            raise HTTPException(409, "Press Start first.")
        if lab.done:
            raise HTTPException(409, "This run has finished all of its questions.")
        row = _guard(lab.step)
        return {"row": row, "summary": lab.summary()}


@router.get("/learn/rows")
def rows(since: int = 0):
    lab = SESSION.lab
    return lab.rows[since:] if lab else []
