import random

import pytest

from agent import Agent
from config import load_config
from conftest import WordHashEmbedder
from domains import get_domain
from factory import build_atmc
from strategy import BASELINE, CATEGORIES, StrategyBook, wilson


def book(**kw):
    return StrategyBook(WordHashEmbedder(), rng=random.Random(0), **kw)


def record(b, category, strategy, wins, losses):
    for i in range(wins + losses):
        ref = f"{category}-{strategy}-{len(b.events)}"
        b.start(ref, category, strategy, "practise")
        b.resolve(ref, i < wins)


def test_wilson_does_not_reward_a_lucky_small_sample():
    assert wilson(2, 2)[0] < wilson(20, 20)[0]
    assert wilson(0, 0) == (0.0, 1.0)
    lo, hi = wilson(8, 10)
    assert lo < 0.8 < hi


def test_problem_types_come_from_the_nearest_prototype():
    b = book()
    for key, (_, description) in CATEGORIES.items():          # each category's own description maps to itself
        assert b.classify(description) == key
    assert b.classify("zzz qqq xxx nothing related") == "general"


def test_baseline_is_champion_until_there_is_enough_evidence():
    b = book(min_trials=5)
    record(b, "lists", "direct", 3, 0)                       # great, but only 3 trials and no baseline data
    assert b.champion("lists") == (BASELINE, "baseline")
    record(b, "lists", BASELINE, 3, 2)
    assert b.champion("lists") == (BASELINE, "baseline")     # baseline still under min_trials


def test_a_challenger_is_promoted_only_when_clearly_better_and_rolled_back_when_it_worsens():
    b = book(min_trials=5)
    record(b, "lists", BASELINE, 3, 3)                       # baseline: 50%
    record(b, "lists", "revise", 10, 0)                      # challenger: 10/10
    assert b.champion("lists") == ("revise", "promoted")
    record(b, "lists", "revise", 0, 14)                      # later it fails repeatedly -> 10/24
    assert b.champion("lists") == (BASELINE, "baseline")     # automatic rollback, no manual step


def test_a_lucky_streak_below_min_trials_does_not_promote():
    b = book(min_trials=5)
    record(b, "lists", BASELINE, 3, 3)
    record(b, "lists", "revise", 4, 0)
    assert b.champion("lists")[0] == BASELINE


def test_evidence_is_kept_per_problem_type():
    b = book(min_trials=5)
    record(b, "lists", BASELINE, 3, 3)
    record(b, "lists", "revise", 10, 0)
    assert b.champion("lists")[0] == "revise" and b.champion("strings")[0] == BASELINE


def test_practising_explores_the_least_tried_strategy_and_serving_uses_the_champion():
    b = book(explore=0.0)
    record(b, "lists", BASELINE, 4, 0)
    record(b, "lists", "direct", 2, 0)
    s, why = b.choose("lists", "practise")
    assert s == "revise" and "least" in why                  # never tried
    assert b.choose("lists", "serve")[0] == BASELINE


def test_unresolved_attempts_do_not_count_as_evidence():
    b = book()
    b.start("t1", "lists", "direct", "serve")
    assert b.stats("lists")["direct"] == (0, 0)
    b.resolve("t1", True)
    assert b.stats("lists")["direct"] == (1, 1)
    b.resolve("t1", False)                                   # a later dispute overturns it
    assert b.stats("lists")["direct"] == (0, 1)


def test_table_reports_roles_for_the_ui():
    b = book(min_trials=5)
    record(b, "lists", BASELINE, 3, 3)
    record(b, "lists", "revise", 10, 0)
    row = next(r for r in b.table() if r["category"] == "lists")
    roles = {s["strategy"]: s["role"] for s in row["strategies"]}
    assert roles == {"plan_first": "baseline", "direct": "challenger", "revise": "champion"}
    assert set(r["category"] for r in b.table()) <= set(CATEGORIES) | {"general"}


# ---- the agent applies a strategy --------------------------------------------------------------
class _Calls:
    def __init__(self, fake_llm):
        self.prompts = []

        def script(prompt):
            self.prompts.append(prompt)
            if "Break the following task" in prompt:
                return "1. do it"
            if "Draft solution" in prompt:
                return "Fixed.\n```python\ndef add(a, b):\n    return a + b\n```"
            return "Draft.\n```python\ndef add(a, b):\n    return a - b\n```"
        fake_llm(script)


@pytest.fixture
def agent(fake_llm):
    cfg = load_config()
    d = cfg.to_dict(); d["agent"].update(use_planner=True, use_reflection=False, curriculum="fixed"); cfg = type(cfg)(d)
    emb = WordHashEmbedder()
    domain = get_domain("code", cfg)
    from agent import build_agent
    a = build_agent(cfg, build_atmc(cfg, emb, lambda t: "s"), emb, domain, [])
    a.calls = _Calls(fake_llm)
    return a, domain


@pytest.mark.parametrize("strategy,plans,revises", [("plan_first", 1, 0), ("direct", 0, 0), ("revise", 0, 1), (None, 1, 0)])
def test_each_strategy_runs_the_steps_it_promises(agent, strategy, plans, revises):
    a, domain = agent
    a.attempt(domain.user_task("write add(a, b)", "assert add(1, 2) == 3"), strategy)
    prompts = a.calls.prompts
    assert sum("Break the following task" in p for p in prompts) == plans
    assert sum("Draft solution" in p for p in prompts) == revises


def test_revising_can_turn_a_wrong_draft_into_a_passing_answer(agent):
    a, domain = agent
    task = domain.user_task("write add(a, b)", "assert add(1, 2) == 3")
    assert a.attempt(task, "direct").verdict.correct is False       # the draft subtracts
    assert a.attempt(task, "revise").verdict.correct is True        # the review fixes it


def test_the_default_is_not_called_champion_until_something_earns_promotion():
    b = book()
    record(b, "lists", BASELINE, 1, 0)
    row = next(r for r in b.table() if r["category"] == "lists")
    assert {x["strategy"]: x["role"] for x in row["strategies"]}[BASELINE] == "baseline"
