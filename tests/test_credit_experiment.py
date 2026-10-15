"""Experiment 1 (credit assignment): the three credit rules, the harder poison, repeated passes, trust metrics,
and the simulator."""
import json

import pytest

import experiment
from agent import build_agent
from config import load_config
from credit_sim import SimParams, run_trial
from domains import get_domain
from experiment import relabel_as_correct, trust_auc, trust_metrics
from factory import build_store
from test_agent import TASKS, math_llm
from test_experiment import run, setup  # noqa: F401  (pytest fixture reused)


# ---- the three credit rules --------------------------------------------------------------------------------------
def trust_after(rule, correct, cfg, embedder, fake_llm):
    """Two notes are retrieved; the model cites only the first. Returns (trust of cited note, trust of the other)."""
    cfg = load_config(overrides={"agent": {"credit": rule}})
    fake_llm(math_llm(correct, cite="1"))
    store = build_store("atmc", cfg, embedder, lambda t: "S")
    q = embedder.embed(TASKS[0]["question"])
    store.add("note A", q, correct=True)
    store.add("note B", q, correct=True)
    store.items[0].trust, store.items[1].trust = 0.60, 0.50          # A ranks first, so "Past experience 1" is A
    a = build_agent(cfg, store, embedder, get_domain("math", cfg), TASKS)
    a.step(TASKS[0])
    by = {m.text: m.trust for m in store.items if m.text in ("note A", "note B")}
    return round(by["note A"], 3), round(by["note B"], 3)


@pytest.mark.parametrize("rule,correct,expected", [
    ("cited", True, (0.65, 0.50)),            # only the cited note is rewarded
    ("all", True, (0.65, 0.55)),              # every retrieved note is rewarded
    ("asymmetric", True, (0.65, 0.50)),       # reward: cited only
    ("cited", False, (0.45, 0.50)),           # only the cited note is blamed
    ("all", False, (0.45, 0.35)),             # every retrieved note is blamed
    ("asymmetric", False, (0.45, 0.35)),      # blame: everything retrieved
])
def test_credit_rules_update_the_expected_notes(cfg, embedder, fake_llm, rule, correct, expected):
    assert trust_after(rule, correct, cfg, embedder, fake_llm) == expected


def test_unknown_credit_rule_is_rejected():
    with pytest.raises(ValueError, match="agent.credit"):
        load_config(overrides={"agent": {"credit": "everyone"}})
    for ok in ("cited", "all", "asymmetric"):
        assert load_config(overrides={"agent": {"credit": ok}}).agent.credit == ok


# ---- poison recorded as a success, and the trust metrics ------------------------------------------------------------
def test_relabel_turns_a_failure_note_into_a_false_success_without_touching_the_solution():
    code_note = "Task: add\nSolution:\ndef add(a, b):\n    return a - b\nOutcome: INCORRECT (wrong_answer: assertion failed)"
    math_note = "Q: 2+2?\nA: I worked through it. #### 5\nOutcome: INCORRECT (gold=5 was wrong)"
    for note in (code_note, math_note):
        out = relabel_as_correct(note)
        assert out.endswith("Outcome: CORRECT") and "INCORRECT" not in out
        assert out.split("Outcome")[0] == note.split("Outcome")[0]


def test_trust_auc_values():
    assert trust_auc([0.6, 0.7], [0.2, 0.3]) == 1.0
    assert trust_auc([0.2], [0.6]) == 0.0
    assert trust_auc([0.5, 0.5], [0.5]) == 0.5
    assert trust_auc([], [0.5]) is None and trust_auc([0.5], []) is None


def test_trust_metrics_separates_genuine_from_planted(cfg, embedder):
    store = build_store("atmc", cfg, embedder, lambda t: "S")
    q = embedder.embed("x")
    store.add("genuine", q, correct=True)
    store.add("planted", q, correct=True, is_poison=True)
    store.items[1].trust = 0.2                                        # below the 0.30 gate
    m = trust_metrics(store)
    assert m["trust_auc"] == 1.0 and m["poison_blocked_frac"] == 1.0 and m["genuine_blocked_frac"] == 0.0
    assert m["n_poison_remaining"] == 1


# ---- the preset and the harness ----------------------------------------------------------------------------------------
def test_credit_preset_is_the_designed_experiment():
    p = experiment.PRESETS["credit"]
    assert p.poison and p.poison_label == "correct" and p.repeats == 3
    assert {"credit_cited", "credit_all", "credit_asymmetric"} == set(p.conditions)
    assert {c.agent["credit"] for c in p.conditions.values()} == {"cited", "all", "asymmetric"}
    assert experiment.estimate_calls(p, 40, 0, 1, None) > 3 * experiment.estimate_calls(experiment.PRESETS["poison"], 40, 0, 1, None) // 2


def test_run_records_repeated_passes_false_success_poison_and_trust_metrics(setup):
    run(setup, "credit", "credit_cited", n=6)
    d = json.loads((setup[4] / "credit__math__credit_cited__seed1.json").read_text())
    ids = [s["task_id"] for s in d["steps"]]
    assert len(ids) == 18 and ids[:6] == ids[6:12] == ids[12:]          # the same six tasks, three passes, in order
    assert d["repeats"] == 3 and d["poison_label"] == "correct" and d["n"] == 18
    s = d["summary"]
    assert s["n_poison_seeded"] == 2 and s["n_steps"] == 18
    assert {"trust_auc", "poison_trust_final", "genuine_trust_final", "poison_blocked_frac"} <= s.keys()
    assert all("poison_trust_mean" in step for step in d["steps"])


# ---- the simulator (real store, simulated model) -------------------------------------------------------------------------
def test_simulator_is_deterministic_and_pairs_by_seed(cfg):
    p = SimParams(n_tasks=12, repeats=2)
    assert run_trial(p, "cited", 7, cfg) == run_trial(p, "cited", 7, cfg)
    assert run_trial(p, "cited", 7, cfg) != run_trial(p, "all", 7, cfg)


def test_in_the_simulator_credit_all_rewards_ignored_wrong_notes_but_cited_does_not(cfg):
    # the model never copies, never cites a wrong note, never omits the USED line, and is always right
    p = SimParams(n_tasks=12, repeats=3, follow=0.0, follow_other=0.0, spurious=0.0, omit_citation=0.0, base_accuracy=1.0)
    cited, all_ = run_trial(p, "cited", 3, cfg), run_trial(p, "all", 3, cfg)
    assert cited["poison_trust"] == pytest.approx(0.55)                 # ignored notes are never credited
    assert all_["poison_trust"] > 0.55                                  # credit-all pumps them up on every success
    assert cited["trust_auc"] > all_["trust_auc"]


# ---- amendment 1: isolate the credit rule, and measure citation reliability -------------------------------------------
def test_citation_status_mirrors_the_parser():
    import llm
    assert llm.citation_status("answer", 0) == "no_memories"
    assert llm.citation_status("answer only", 3) == "missing"
    assert llm.citation_status("x\nUSED: none", 3) == "none"
    assert llm.citation_status("x\nUSED: 1, 3", 3) == "listed"
    assert llm.citation_status("x\nUSED: 9", 3) == "unparseable"
    assert llm.citation_status("x\nUSED: banana", 3) == "unparseable"


def test_isolated_preset_turns_forgetting_and_merging_off_for_every_rule():
    p = experiment.PRESETS["credit_isolated"]
    assert p.poison_label == "correct" and p.repeats == 3
    for name, cond in p.conditions.items():
        assert cond.pillars == {"enable_forgetting": False, "enable_compression": False}, name
    assert {c.agent["credit"] for c in p.conditions.values()} == {"cited", "all", "asymmetric"}


def test_isolated_run_keeps_every_planted_note_to_the_end_and_logs_citation_reliability(setup):
    run(setup, "credit_isolated", "iso_cited", n=6)
    d = json.loads((setup[4] / "credit_isolated__math__iso_cited__seed1.json").read_text())
    s = d["summary"]
    assert s["n_compressions"] == 0 and s["n_poison_remaining"] == s["n_poison_seeded"] == 2     # nothing merged or pruned away
    assert {"cite_listed_frac", "cite_missing_frac"} <= s.keys()
    assert all("cite_status" in step for step in d["steps"])
