"""The agent loop, written once.

    pick task (curriculum) -> plan -> retrieve memories -> solve -> check
        -> credit the memories that were actually used -> reflect on failures
        -> write the new memory -> (forget / compress inside the store)

Every experiment script, the evaluation harness and the demo backend run this
same `Agent.step`, so they cannot drift apart.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import llm
from domains.base import Domain, Verdict


@dataclass
class StepResult:
    task: dict
    plan: str | None
    retrieved: list
    used_mask: list[bool]
    raw: str
    answer: object
    verdict: Verdict
    lesson: str | None
    note: str
    n_poison_retrieved: int
    n_poison_used: int
    retrieval_ms: float
    tokens_in: int
    tokens_out: int
    memory_size: int
    q_emb: object = None
    learned: bool = False

    @property
    def correct(self) -> bool:
        return self.verdict.correct

    def to_log(self) -> dict:
        """JSON-safe summary, small enough to save for every step of every run."""
        return {
            "task_id": self.task["id"],
            "correct": self.verdict.correct,
            "stage": self.verdict.stage,
            "n_retrieved": len(self.retrieved),
            "n_poison_retrieved": self.n_poison_retrieved,
            "n_poison_used": self.n_poison_used,
            "retrieval_ms": round(self.retrieval_ms, 3),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "memory_size": self.memory_size,
            "planned": self.plan is not None,
            "reflected": self.lesson is not None,
        }


@dataclass
class Agent:
    cfg: object
    domain: Domain
    store: object
    embedder: object
    pool: list[dict] = field(default_factory=list)
    planner: object = None
    reflector: object = None
    curriculum: object = None
    rng: random.Random = field(default_factory=lambda: random.Random(0))
    history: list[dict] = field(default_factory=list)

    @property
    def done_ids(self) -> set:
        return {h["task_id"] for h in self.history}

    def next_task(self) -> dict:
        if self.curriculum is None:
            raise ValueError("no curriculum configured and no task given")
        return self.curriculum.choose(self.pool, self.done_ids, self.store, self.embedder, self.rng)

    def attempt(self, task: dict) -> StepResult:
        """Plan, retrieve, solve and check -- without touching memory."""
        before = llm.usage_snapshot()

        plan = self.planner.plan(task) if self.planner else None

        q_emb = self.embedder.embed(task["question"])
        t0 = time.perf_counter()
        retrieved = self.store.retrieve(q_emb, k=self.cfg.retrieval.top_k)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        raw = llm.complete(self.domain.build_prompt(task, [m.text for m in retrieved], plan))
        answer = self.domain.extract_answer(raw, task)
        verdict = self.domain.check(task, answer)
        used_mask = llm.extract_used_mask(raw, len(retrieved))

        after = llm.usage_snapshot()
        return StepResult(
            task=task, plan=plan, retrieved=retrieved, used_mask=used_mask, raw=raw, answer=answer,
            verdict=verdict, lesson=None, note="",
            n_poison_retrieved=sum(m.is_poison for m in retrieved),
            n_poison_used=sum(1 for m, u in zip(retrieved, used_mask) if u and m.is_poison),
            retrieval_ms=retrieval_ms,
            tokens_in=after["prompt_tokens"] - before["prompt_tokens"],
            tokens_out=after["output_tokens"] - before["output_tokens"],
            memory_size=self.store.size(), q_emb=q_emb,
        )

    def learn(self, result: StepResult, correct: bool | None = None, feedback: str = "") -> StepResult:
        """Update memory from an attempt: credit the notes that were used, reflect on a failure and
        store the new note. A task with no ground truth (verdict.verified is False) needs the user's
        own verdict -- `correct` -- before anything is learned from it."""
        if result.learned:
            return result
        verdict = result.verdict
        if not verdict.verified:
            if correct is None:
                raise ValueError("this attempt is unverified: learning needs the user's verdict")
            verdict = Verdict(correct, "ok" if correct else "user_rejected", feedback.strip()[:300])
            result.verdict = verdict
        task, answer = result.task, result.answer
        self.store.update_from_outcome(result.retrieved, verdict.correct, used_mask=result.used_mask)

        lesson = None
        if self.reflector and not verdict.correct:
            before = llm.usage_snapshot()
            lesson = self.reflector.lesson(task, answer, verdict)
            after = llm.usage_snapshot()
            result.tokens_in += after["prompt_tokens"] - before["prompt_tokens"]
            result.tokens_out += after["output_tokens"] - before["output_tokens"]
        note = self.domain.reflection(task, result.raw, answer, verdict)
        if lesson:
            note += f"\nLesson: {lesson}"
        self.store.add(note, result.q_emb, verdict.correct)

        result.lesson, result.note, result.learned = lesson, note, True
        result.memory_size = self.store.size()
        self.history.append(result.to_log())
        return result

    def step(self, task: dict | None = None) -> StepResult:
        """One full benchmark step: attempt, then learn (the checker is ground truth)."""
        result = self.attempt(task or self.next_task())
        return self.learn(result) if result.verdict.verified else result

    def run(self, n: int, on_step=None) -> list[StepResult]:
        results = []
        for _ in range(n):
            try:
                r = self.step()
            except StopIteration:
                break
            results.append(r)
            if on_step:
                on_step(r)
        return results


def build_agent(cfg, store, embedder, domain: Domain, pool: list[dict], seed: int = 0) -> Agent:
    """Assemble an agent from the config's `agent` section."""
    from curriculum import make_curriculum
    from planner import Planner
    from reflection import Reflector

    return Agent(
        cfg=cfg, domain=domain, store=store, embedder=embedder, pool=pool,
        planner=Planner() if cfg.agent.use_planner else None,
        reflector=Reflector() if cfg.agent.use_reflection else None,
        curriculum=make_curriculum(cfg),
        rng=random.Random(seed),
    )
