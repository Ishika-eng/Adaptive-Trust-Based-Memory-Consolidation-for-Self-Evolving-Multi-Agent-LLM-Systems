import random

import numpy as np
import pytest

import llm
from agent import Agent, build_agent
from config import load_config
from curriculum import RandomCurriculum, WeaknessCurriculum
from domains import get_domain
from factory import build_atmc, build_store
from planner import Planner
from reflection import Reflector

TASKS = [
    {"id": f"t{i}", "question": q, "gold": str(g)}
    for i, (q, g) in enumerate([
        ("apples basket count total fruit", 12), ("train speed distance hours travelled", 90),
        ("pizza slices shared among friends", 4), ("garden flowers petunias geraniums planted", 140),
    ])
]


def math_llm(correct=True, cite="1"):
    """Fake model: answers any 'Problem: <q>' with the gold answer (or a wrong one)."""
    gold = {t["question"]: t["gold"] for t in TASKS}

    def script(prompt):
        if "Break the following task" in prompt:
            return "1. read\n2. compute"
        if "previous attempt" in prompt:
            return "You misread the quantities; recompute carefully."
        q = prompt.split("Problem: ", 1)[1].strip()
        ans = gold[q] if correct else str(int(gold[q]) + 1)
        used = f"\nUSED: {cite}" if "Past experience" in prompt else ""
        return f"reasoning...\n#### {ans}{used}"
    return script


def make_agent(cfg, embedder, fake_llm, script, kind="atmc", **agent_overrides):
    fake_llm(script)
    cfg = load_config(overrides={"agent": agent_overrides}) if agent_overrides else cfg
    store = build_store(kind, cfg, embedder, lambda t: "SUMMARY")
    return build_agent(cfg, store, embedder, get_domain("math", cfg), TASKS)


def test_step_checks_answer_stores_a_note_and_records_cost(cfg, embedder, fake_llm):
    a = make_agent(cfg, embedder, fake_llm, math_llm(True))
    r = a.step(TASKS[0])
    assert r.correct and r.verdict.stage == "ok"
    assert a.store.size() == 1 and "Outcome: CORRECT" in a.store.items[0].text
    assert r.tokens_in > 0 and r.tokens_out > 0 and r.retrieval_ms >= 0
    assert a.history[0]["task_id"] == "t0" and a.history[0]["correct"] is True


def test_wrong_answer_is_stored_as_a_low_trust_warning(cfg, embedder, fake_llm):
    a = make_agent(cfg, embedder, fake_llm, math_llm(False))
    r = a.step(TASKS[0])
    assert not r.correct and r.verdict.stage == "wrong_answer"
    assert a.store.items[0].trust == pytest.approx(0.35) and "INCORRECT" in a.store.items[0].text


def test_trust_moves_only_for_the_memory_the_model_cited(cfg, embedder, fake_llm):
    a = make_agent(cfg, embedder, fake_llm, math_llm(True, cite="2"))
    for t in TASKS[:3]:
        a.store.add(f"Q: {t['question']}\nOutcome: CORRECT", embedder.embed(t["question"]), True)
    before = [m.trust for m in a.store.items]
    r = a.step(TASKS[3])
    cited = [m for m, u in zip(r.retrieved, r.used_mask) if u]
    uncited = [m for m, u in zip(r.retrieved, r.used_mask) if not u]
    assert len(cited) == 1 and len(uncited) == len(r.retrieved) - 1
    assert cited[0].trust == pytest.approx(0.60)
    assert all(m.trust == pytest.approx(0.55) for m in uncited)
    assert before == [0.55, 0.55, 0.55]


def test_poison_exposure_and_use_are_counted(cfg, embedder, fake_llm):
    a = make_agent(cfg, embedder, fake_llm, math_llm(True, cite="1"))
    a.store.add("Q: apples\nOutcome: INCORRECT", embedder.embed(TASKS[0]["question"]), False, is_poison=True)
    r = a.step(TASKS[0])
    assert r.n_poison_retrieved == 1 and r.n_poison_used == 1


def test_planner_is_off_by_default(cfg, embedder, fake_llm):
    fake = fake_llm(math_llm())
    a = build_agent(cfg, build_store("atmc", cfg, embedder, lambda t: "S"), embedder, get_domain("math", cfg), TASKS)
    r = a.step(TASKS[0])
    assert a.planner is None and r.plan is None
    assert not any("Break the following task" in p for p in fake.prompts) and len(fake.prompts) == 1


def test_planner_when_enabled_sees_no_memories_and_feeds_the_solver(cfg, embedder, fake_llm):
    fake = fake_llm(math_llm())
    pcfg = load_config(overrides={"agent": {"use_planner": True}})
    a = build_agent(pcfg, build_store("atmc", pcfg, embedder, lambda t: "S"), embedder, get_domain("math", pcfg), TASKS)
    a.store.add("SECRET-MEMORY-TEXT", embedder.embed(TASKS[0]["question"]), True)
    r = a.step(TASKS[0])
    plan_prompt = next(p for p in fake.prompts if "Break the following task" in p)
    assert "SECRET-MEMORY-TEXT" not in plan_prompt            # planning is independent of memory
    solve_prompt = next(p for p in fake.prompts if "Problem:" in p)
    assert "SECRET-MEMORY-TEXT" in solve_prompt               # ...but the solver does get the memory
    assert "Plan to follow:\n1. read" in solve_prompt and r.plan == "1. read\n2. compute"


def test_reflection_only_on_failure_and_lesson_is_stored(cfg, embedder, fake_llm):
    ok = make_agent(cfg, embedder, fake_llm, math_llm(True), use_reflection=True)
    r = ok.step(TASKS[0])
    assert r.lesson is None and "Lesson:" not in ok.store.items[0].text

    bad = make_agent(cfg, embedder, fake_llm, math_llm(False), use_reflection=True)
    r = bad.step(TASKS[0])
    assert r.lesson and "Lesson: You misread" in bad.store.items[0].text
    assert bad.store.items[0].trust == pytest.approx(0.35)    # a lesson never buys trust


def test_quota_failure_mid_step_leaves_memory_untouched(cfg, embedder, fake_llm):
    def boom(prompt):
        raise llm.QuotaExhausted("daily quota")
    a = make_agent(cfg, embedder, fake_llm, boom)
    a.store.add("seed", embedder.embed("seed"), True)
    with pytest.raises(llm.QuotaExhausted):
        a.step(TASKS[0])
    assert a.store.size() == 1 and a.history == [] and a.store.items[0].trust == pytest.approx(0.55)


def test_run_with_random_curriculum_never_repeats_and_stops_when_exhausted(cfg, embedder, fake_llm):
    a = make_agent(cfg, embedder, fake_llm, math_llm(True))
    results = a.run(10)                       # pool has 4 tasks
    ids = [r.task["id"] for r in results]
    assert sorted(ids) == sorted(t["id"] for t in TASKS) and len(set(ids)) == 4


def test_weakness_curriculum_targets_the_area_of_past_failures(cfg, embedder):
    store = build_atmc(cfg, embedder, lambda t: "S")
    store.add("note", embedder.embed(TASKS[1]["question"]), correct=False)           # failed on trains
    store.add("poison", embedder.embed(TASKS[2]["question"]), correct=False, is_poison=True)  # ignored
    cur = WeaknessCurriculum(explore=0.0)
    pool = [TASKS[0], TASKS[2], TASKS[3], dict(TASKS[1], id="t1-variant", question=TASKS[1]["question"] + " again")]
    assert cur.choose(pool, set(), store, embedder, random.Random(0))["id"] == "t1-variant"


def test_weakness_curriculum_falls_back_to_random_with_no_failures(cfg, embedder):
    store = build_atmc(cfg, embedder, lambda t: "S")
    t = WeaknessCurriculum(0.0).choose(TASKS, set(), store, embedder, random.Random(1))
    assert t in TASKS


def test_code_domain_end_to_end(cfg, embedder, fake_llm):
    code = get_domain("code", cfg)
    task = code._load("humaneval")[0]
    task = dict(task)
    good = f"```python\n{task['question']}{task['reference']}```"
    bad = "```python\ndef has_close_elements(numbers, threshold):\n    return False\n```"
    for reply, expect, stage in ((good, True, "ok"), (bad, False, "wrong_answer")):
        fake_llm(lambda p, r=reply: r)
        store = build_store("atmc", cfg, embedder, lambda t: "S")
        a = Agent(cfg=cfg, domain=code, store=store, embedder=embedder, pool=[task])
        r = a.step(task)
        assert r.correct is expect and r.verdict.stage == stage
        assert ("CORRECT" if expect else f"INCORRECT ({stage}") in store.items[0].text
