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
    autocheck: object = None
    explain: list = field(default_factory=list)     # why each retrieved note was chosen
    filtered: int = 0                               # notes skipped because their trust was too low
    cite_status: str = ""                           # how the USED line was read: listed | none | missing | unparseable | no_memories
    stale_marked: int = 0                           # stored solutions proven wrong by the current checker (rule change detected)
    repair: dict | None = None                      # {attempted, passed} when a failed attempt got one evidence-guided retry

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
            "cite_status": self.cite_status,
            "planned": self.plan is not None,
            "reflected": self.lesson is not None,
            "stale_marked": self.stale_marked,
            "repair_attempted": bool(self.repair and self.repair["attempted"]),
            "repair_passed": bool(self.repair and self.repair["passed"]),
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

    def attempt(self, task: dict, strategy: str | None = None) -> StepResult:
        """Plan, retrieve, solve and check -- without touching memory.

        `strategy` (chat assistant only; benchmarks pass None and keep the configured behaviour):
        "plan_first" plans before solving, "direct" skips planning, "revise" skips planning and has the
        model review and correct its own first draft."""
        before = llm.usage_snapshot()

        use_plan = self.planner is not None and strategy in (None, "plan_first")
        plan = self.planner.plan(task) if use_plan else None

        q_emb = self.embedder.embed(task["question"])
        t0 = time.perf_counter()
        retrieved = self.store.retrieve(q_emb, k=self.cfg.retrieval.top_k)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        raw = llm.complete(self.domain.build_prompt(task, [m.text for m in retrieved], plan))
        answer = self.domain.extract_answer(raw, task)
        used_mask = llm.extract_used_mask(raw, len(retrieved))     # which notes the *first* draft relied on
        cite_status = llm.citation_status(raw, len(retrieved))
        if strategy == "revise" and answer:
            raw = llm.complete(self.domain.revise_prompt(task, answer))
            answer = self.domain.extract_answer(raw, task) or answer
        verdict = self.domain.check(task, answer)

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
            explain=list(getattr(self.store, "last_explanations", [])), filtered=getattr(self.store, "last_filtered", 0),
            cite_status=cite_status,
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
            verdict = Verdict(correct, "ok" if correct else "user_rejected", feedback.strip()[:300], evidence="user")
            result.verdict = verdict
        task, answer = result.task, result.answer
        weight = self.cfg.to_dict().get("autoverify", {}).get("evidence_weight", 0.5) if verdict.evidence == "generated" else 1.0
        # credit rule: "cited" (default) updates only notes the model reported using; "all" updates every retrieved
        # note; "asymmetric" rewards only cited notes after a success but blames every retrieved note after a failure
        rule = self.cfg.to_dict()["agent"].get("credit", "cited")
        use_citations = rule == "cited" or (rule == "asymmetric" and verdict.correct)
        mask = result.used_mask if use_citations else None
        self.store.update_from_outcome(result.retrieved, verdict.correct, used_mask=mask, weight=weight)
        if self.cfg.to_dict()["agent"].get("utility_credit", False) and hasattr(self.store, "record_outcome") and verdict.verified:
            self.store.record_outcome(result.retrieved, verdict.correct)        # credit every shown note, cited or not

        if (self.cfg.to_dict()["agent"].get("reverify_stale", False) and not verdict.correct and verdict.verified
                and verdict.evidence == "tests" and hasattr(self.store, "mark_stale")):
            result.stale_marked = self._reverify_stale(task)

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
        claim_ok = bool(self.cfg.to_dict()["agent"].get("evidence_note_trust", False)) and self.domain.note_claim_verified(verdict)
        self.store.add(note, result.q_emb, verdict.correct, source=verdict.evidence, claim_verified=claim_ok)
        self._repair_and_store(result, verdict)

        result.lesson, result.note, result.learned = lesson, note, True
        result.memory_size = self.store.size()
        self.history.append(result.to_log())
        return result

    def _reverify_stale(self, task: dict) -> int:
        """After a failure: run every stored solution for THIS function against the current checker. One that now fails was right
        once and is wrong now (the rule changed): mark it stale. Solutions for other functions cannot be judged and are skipped."""
        radius = self.cfg.retrieval.to_dict().get("stale_radius", 0.75)
        marked = 0
        for m in list(self.store.items):
            if getattr(m, "slot", None) is not None and hasattr(self.store, "solutions_of"):
                codes = self.store.solutions_of(m)             # a rule slot holds several verified solutions
            else:
                one = self.domain.note_solution(m.text)
                codes = [one] if one else []
            if codes and getattr(m, "stale", False) is False and any(self.domain.reverify(task, c) is False for c in codes):
                self.store.mark_stale(m, radius=radius)
                marked += 1
        return marked

    def _repair_and_store(self, result: StepResult, verdict: Verdict) -> None:
        """One fresh attempt given only the checker's evidence. If it passes the tests, its solution is stored as a verified
        worked example (the only way a failure turns into a correct example the model can imitate). The reported pass rate
        always uses the FIRST attempt; this retry only feeds memory."""
        if (not self.cfg.to_dict()["agent"].get("repair", False) or verdict.correct or not verdict.verified
                or verdict.evidence != "tests"):
            return
        acfg = self.cfg.to_dict()["agent"]
        notes = []
        if acfg.get("repair_with_memory", False):
            k = self.cfg.retrieval.top_k
            if acfg.get("signature_repair", False) and hasattr(self.store, "retrieve_for_repair"):
                notes = self.store.retrieve_for_repair(result.q_emb, self.domain.failure_signature(verdict), k=k)
            else:
                notes = self.store.retrieve(result.q_emb, k=k)
        prompt = self.domain.repair_prompt(result.task, verdict, [m.text for m in notes] if notes else None)
        if prompt is None:
            return
        before = llm.usage_snapshot()
        raw = llm.complete(prompt)
        answer = self.domain.extract_answer(raw, result.task)
        second = self.domain.check(result.task, answer)
        after = llm.usage_snapshot()
        result.tokens_in += after["prompt_tokens"] - before["prompt_tokens"]
        result.tokens_out += after["output_tokens"] - before["output_tokens"]
        result.repair = {"attempted": True, "passed": bool(second.correct)}
        if notes and acfg.get("utility_credit", False) and hasattr(self.store, "record_outcome"):
            self.store.record_outcome(notes, second.correct)
        if second.correct and second.verified:
            example = self.domain.worked_example(result.task, answer, verdict)
            if example:
                self.store.add(example, result.q_emb, True, source="tests")

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
