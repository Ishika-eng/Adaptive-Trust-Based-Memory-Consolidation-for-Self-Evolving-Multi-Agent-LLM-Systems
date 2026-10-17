"""One harness for every experiment, any domain.

    python3 pilot/experiment.py poison    --domain math --seeds 42,123,456
    python3 pilot/experiment.py ablation  --domain code
    python3 pilot/experiment.py baselines --domain code --n 40
    python3 pilot/experiment.py agent     --domain code --heldout 20
    python3 pilot/experiment.py --list

A *preset* is a set of named *conditions* (memory system + agent options).
Every (condition, seed) run is saved on its own to results/eval/, and skipped
if its file already exists -- so a long experiment can be stopped by the daily
API quota and simply re-run tomorrow. Everything runs through agent.Agent.

After training on n tasks, `--heldout m` scores m unseen tasks with memory frozen
(no learning, no hit counting): "does the memory built help on new problems?"
-- the fair way to evaluate things like the curriculum, which changes *which*
tasks are practised and so cannot be judged by in-run pass rate.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import llm
from agent import Agent, build_agent
from config import ROOT, Config, load_config
from domains import get_domain
from domains.base import Verdict
from embeddings import make_embedder
from factory import build_store

OUT_DIR = ROOT / "pilot" / "results" / "eval"


@dataclass
class Condition:
    kind: str                                   # none | static | reflexion | atmc
    pillars: dict = field(default_factory=dict)  # enable_trust / enable_forgetting / enable_compression
    agent: dict = field(default_factory=dict)    # overrides for cfg.agent (planner, reflection, curriculum)
    retrieval: dict = field(default_factory=dict)  # overrides for cfg.retrieval (e.g. min_similarity)
    composite: dict = field(default_factory=dict)  # overrides for cfg.composite (ranking weights; must still sum to 1)
    compression: dict = field(default_factory=dict)  # overrides for cfg.compression (e.g. evidence_merge)


@dataclass
class Preset:
    conditions: dict[str, Condition]
    poison: bool = False
    heldout: int = 0
    pool_factor: int = 1       # train pool = n * pool_factor (bigger when the agent chooses its own tasks)
    poison_label: str = "failed"   # how planted notes are recorded: "failed" (labelled INCORRECT) or "correct" (a wrong
                                   # solution recorded as a success -- the harder, realistic false-positive case)
    repeats: int = 1           # passes over the same tasks, so planted notes are retrieved more than once
    verifier_noise: float = 0.0   # P(a wrong answer is recorded as correct): a fooled auto-verifier or a mistaken thumbs-up.
                                  # Reported pass rates still use the TRUE result; only what the agent LEARNS from is noisy.


PRESETS: dict[str, Preset] = {
    "poison": Preset({"static": Condition("static"), "atmc": Condition("atmc")}, poison=True),
    "ablation": Preset({
        "full_system": Condition("atmc"),
        "wo_trust": Condition("atmc", {"enable_trust": False}),
        "wo_forgetting": Condition("atmc", {"enable_forgetting": False}),
        "wo_compression": Condition("atmc", {"enable_compression": False}),
    }, poison=True),
    "baselines": Preset({
        "no_memory": Condition("none"), "static": Condition("static"),
        "reflexion": Condition("reflexion"), "atmc": Condition("atmc"),
    }, heldout=0),
    "baselines_poison": Preset({
        "no_memory": Condition("none"), "static": Condition("static"),
        "reflexion": Condition("reflexion"), "atmc": Condition("atmc"),
        # relevance floor: only inject notes at least this similar to the task
        "static_minsim": Condition("static", retrieval={"min_similarity": 0.75}),
        "atmc_minsim": Condition("atmc", retrieval={"min_similarity": 0.75}),
    }, poison=True),
    # Experiment 1 (docs/experiments/EXP1_credit_assignment.md): does crediting only the notes the model cited
    # separate good notes from bad ones better than crediting every retrieved note (the MemRL-style rule)?
    "credit": Preset({
        "credit_cited": Condition("atmc", agent={"credit": "cited"}),
        "credit_all": Condition("atmc", agent={"credit": "all"}),
        "credit_asymmetric": Condition("atmc", agent={"credit": "asymmetric"}),    # exploratory, not pre-registered
    }, poison=True, poison_label="correct", repeats=3),
    # Experiment 1, version 2 (amendment 1): forgetting and merging OFF, so every planted note keeps its identity to the
    # end and the trust metrics are well defined. Version 1 (above) is kept: merging made its metrics ill-posed.
    "credit_isolated": Preset({
        "iso_cited": Condition("atmc", {"enable_forgetting": False, "enable_compression": False}, {"credit": "cited"}),
        "iso_all": Condition("atmc", {"enable_forgetting": False, "enable_compression": False}, {"credit": "all"}),
        "iso_asymmetric": Condition("atmc", {"enable_forgetting": False, "enable_compression": False}, {"credit": "asymmetric"}),
    }, poison=True, poison_label="correct", repeats=3),
    # Local-model learning experiment (docs/experiments/LOCAL_MODEL_PILOT.md). Run with --dataset conventions
    # --provider ollama --note-style evidence. Does the agent get better at hidden project conventions as it works?
    "learning": Preset({
        "no_memory": Condition("none"), "static": Condition("static"),
        "atmc_no_trust": Condition("atmc", {"enable_trust": False}),
        "atmc_no_merge": Condition("atmc", {"enable_compression": False}),
        "atmc": Condition("atmc"),
    }),
    # same, but 25% of wrong answers are recorded as correct: can the agent still learn safely when its checker is fooled?
    "learning_noisy": Preset({
        "no_memory": Condition("none"), "static": Condition("static"),
        "atmc_no_trust": Condition("atmc", {"enable_trust": False}),
        "atmc_no_merge": Condition("atmc", {"enable_compression": False}),
        "atmc": Condition("atmc"),
    }, verifier_noise=0.25),
    # EXPLORATORY, post hoc: why does full ATMC trail static memory in "learning"? Tested by changing how notes are ranked
    # and whether they decay. Any setting chosen from this must be re-tested on families it was not tuned on.
    "learning_diag": Preset({
        "atmc_simrank": Condition("atmc", composite={"w_trust": 0.0, "w_importance": 1.0, "w_decay": 0.0}),
        "atmc_noforget": Condition("atmc", {"enable_forgetting": False}),
        "atmc_simrank_noforget": Condition("atmc", {"enable_forgetting": False}, composite={"w_trust": 0.0, "w_importance": 1.0, "w_decay": 0.0}),
    }),
    # EXPLORATORY, round 2: is the retrieval-frequency boost (importance = sim * (1 + log(1 + hits))) what hurts?
    # atmc_pure has every pillar and the boost off, so it should behave like static memory (a pipeline sanity check).
    "learning_diag2": Preset({
        "atmc_pure": Condition("atmc", {"enable_trust": False, "enable_forgetting": False, "enable_compression": False},
                               retrieval={"hit_boost": False}),
        "atmc_trust_only": Condition("atmc", {"enable_forgetting": False, "enable_compression": False}, retrieval={"hit_boost": False}),
        "atmc_nohits": Condition("atmc", retrieval={"hit_boost": False}),
    }),
    # EXPLORATORY, round 3: does letting verified-evidence failure notes enter at normal trust fix ATMC? (hypothesis from
    # round 2: trust-weighted ranking pushes failure notes -- the ones that carry the rule -- below success notes)
    "learning_fix": Preset({
        "atmc_ev": Condition("atmc", agent={"evidence_note_trust": True}),
        "atmc_ev_nohits": Condition("atmc", agent={"evidence_note_trust": True}, retrieval={"hit_boost": False}),
    }),
    # Ablation ladder (not tuning): add ATMC's pillars to plain memory one at a time, with verified-evidence trust and no hit
    # boost. Rung 3 (all pillars) is "atmc_ev_nohits" in learning_fix. static = no pillars at all.
    "learning_ladder": Preset({
        "ladder_trust": Condition("atmc", {"enable_forgetting": False, "enable_compression": False},
                                  agent={"evidence_note_trust": True}, retrieval={"hit_boost": False}),
        "ladder_trust_forget": Condition("atmc", {"enable_compression": False},
                                         agent={"evidence_note_trust": True}, retrieval={"hit_boost": False}),
    }),
    # PRE-REGISTERED safety test (rule fixed before the runs): does trust-only ATMC beat static memory by >= 3 points, 95% CI
    # above zero, when the checker records wrong answers as correct? static vs ladder_trust at 25% and 50% noise.
    "safety_25": Preset({
        "static": Condition("static"),
        "ladder_trust": Condition("atmc", {"enable_forgetting": False, "enable_compression": False},
                                  agent={"evidence_note_trust": True}, retrieval={"hit_boost": False}),
    }, verifier_noise=0.25),
    "safety_50": Preset({
        "static": Condition("static"),
        "ladder_trust": Condition("atmc", {"enable_forgetting": False, "enable_compression": False},
                                  agent={"evidence_note_trust": True}, retrieval={"hit_boost": False}),
    }, verifier_noise=0.50),
    # Step 1 (planned before running): Qwen cites a note in only ~12 pct of steps, so the 'cited' credit rule gives trust
    # almost no signal. 'asymmetric' blames every retrieved note after a failure, with no citation needed. Success rule, fixed
    # in advance: >= +3 points over static memory (95 pct CI above 0) at 25 and 50 pct noise, and no worse than -3 when clean.
    "asym_clean": Preset({"asym_trust": Condition("atmc", {"enable_forgetting": False, "enable_compression": False},
                                 agent={"evidence_note_trust": True, "credit": "asymmetric"}, retrieval={"hit_boost": False})}),
    "asym_25": Preset({"asym_trust": Condition("atmc", {"enable_forgetting": False, "enable_compression": False},
                                 agent={"evidence_note_trust": True, "credit": "asymmetric"}, retrieval={"hit_boost": False})}, verifier_noise=0.25),
    "asym_50": Preset({"asym_trust": Condition("atmc", {"enable_forgetting": False, "enable_compression": False},
                                 agent={"evidence_note_trust": True, "credit": "asymmetric"}, retrieval={"hit_boost": False})}, verifier_noise=0.50),
    # ATMC v2 (planned before running): trust-only ATMC + evidence consolidation into rule cards. threshold 0.75 was chosen from
    # cluster purity of the task embeddings alone (83 pct same-family), not from any pass rate. Compare against static memory.
    "learning_v2": Preset({
        "atmc_v2": Condition("atmc", {"enable_forgetting": False, "enable_compression": False}, agent={"evidence_note_trust": True},
                             retrieval={"hit_boost": False}, compression={"evidence_merge": True}),
        "atmc_v2_notrust": Condition("atmc", {"enable_trust": False, "enable_forgetting": False, "enable_compression": False},
                                     agent={"evidence_note_trust": True}, retrieval={"hit_boost": False}, compression={"evidence_merge": True}),
    }),
    # ATMC v3 = verified playbook (design from ACE / ExpeL / Voyager / Dynamic Cheatsheet, see docs/experiments/ATMC_V3.md).
    # Written before the run: atmc_v3 should beat static_repair by >= 3 points on first-attempt pass rate (95 pct CI above 0).
    "learning_v3": Preset({
        "static_repair": Condition("static", agent={"repair": True}),
        "atmc_v3_nocards": Condition("atmc", {"enable_forgetting": False, "enable_compression": False},
                                     agent={"evidence_note_trust": True, "repair": True}, retrieval={"hit_boost": False}),
        "atmc_v3": Condition("atmc", {"enable_forgetting": False, "enable_compression": False},
                             agent={"evidence_note_trust": True, "repair": True}, retrieval={"hit_boost": False},
                             compression={"evidence_merge": True}),
    }),
    # ATMC v4 (docs/experiments/ATMC_V4.md, prediction fixed before running): learned utility ranking + failure-signature retrieval
    # for the retry + a retry that uses memory. static_rm is the control (plain memory with the same retry-with-memory).
    "learning_v4": Preset({
        "static_rm": Condition("static", agent={"repair": True, "repair_with_memory": True}),
        "atmc_v4_simonly": Condition("atmc", {"enable_forgetting": False, "enable_compression": False},
                                     agent={"evidence_note_trust": True, "repair": True, "repair_with_memory": True},
                                     retrieval={"hit_boost": False}, compression={"evidence_merge": True}),
        "atmc_v4": Condition("atmc", {"enable_forgetting": False, "enable_compression": False},
                             agent={"evidence_note_trust": True, "repair": True, "repair_with_memory": True,
                                    "signature_repair": True, "utility_credit": True},
                             retrieval={"hit_boost": False, "utility_rank": True}, compression={"evidence_merge": True}),
    }),
    # Rule-change (drift) stream, docs/experiments/DRIFT.md: 29 questions asked under an old rule, then again under the opposite rule.
    # Run with --dataset conventions_drift --n 58.
    "drift": Preset({
        "no_memory": Condition("none"),
        "static_rm": Condition("static", agent={"repair": True, "repair_with_memory": True}),
        "atmc_v4": Condition("atmc", {"enable_forgetting": False, "enable_compression": False},
                             agent={"evidence_note_trust": True, "repair": True, "repair_with_memory": True,
                                    "signature_repair": True, "utility_credit": True},
                             retrieval={"hit_boost": False, "utility_rank": True}, compression={"evidence_merge": True}),
    }),
    # Abstention (docs/experiments/DRIFT.md, part 2): only validated notes are ever shown. static_val is the plain-memory control.
    "learning_v5": Preset({
        "static_val": Condition("static", agent={"repair": True, "repair_with_memory": True},
                                retrieval={"show_kinds": ["example", "solved"]}),
        "atmc_v5": Condition("atmc", {"enable_forgetting": False, "enable_compression": False},
                             agent={"evidence_note_trust": True, "repair": True, "repair_with_memory": True,
                                    "signature_repair": True, "utility_credit": True},
                             retrieval={"hit_boost": False, "utility_rank": True, "show_kinds": ["example", "solved", "card"]},
                             compression={"evidence_merge": True}),
    }),
    # Re-verification detector (docs/experiments/DRIFT.md part 3, rules fixed before running).
    # static_rv: plain memory that drops the one note proven stale. atmc_v6: ATMC v4 + proven-stale marking that demotes the whole older
    # neighbourhood. Dev stream = --dataset conventions_drift --n 58; final, run once = drift2_final with --dataset conventions_drift2 --n 40.
    "drift_v6": Preset({"static_rv": Condition("static", agent={"repair": True, "repair_with_memory": True, "reverify_stale": True}), "atmc_v6": Condition("atmc", {"enable_forgetting": False, "enable_compression": False},
                             agent={"evidence_note_trust": True, "repair": True, "repair_with_memory": True,
                                    "signature_repair": True, "utility_credit": True, "reverify_stale": True},
                             retrieval={"hit_boost": False, "utility_rank": True}, compression={"evidence_merge": True})}),
    "drift2_final": Preset({
        "no_memory": Condition("none"),
        "static_rm": Condition("static", agent={"repair": True, "repair_with_memory": True}),
        "atmc_v4": Condition("atmc", {"enable_forgetting": False, "enable_compression": False},
                             agent={"evidence_note_trust": True, "repair": True, "repair_with_memory": True,
                                    "signature_repair": True, "utility_credit": True},
                             retrieval={"hit_boost": False, "utility_rank": True}, compression={"evidence_merge": True}),
        "static_rv": Condition("static", agent={"repair": True, "repair_with_memory": True, "reverify_stale": True}), "atmc_v6": Condition("atmc", {"enable_forgetting": False, "enable_compression": False},
                             agent={"evidence_note_trust": True, "repair": True, "repair_with_memory": True,
                                    "signature_repair": True, "utility_credit": True, "reverify_stale": True},
                             retrieval={"hit_boost": False, "utility_rank": True}, compression={"evidence_merge": True}),
    }),
    # Rule memory v7 (docs/experiments/RULE_MEMORY.md; rules fixed before running). Clean stream: --dataset conventions --n 110.
    # Dev drift stream: drift_v7 with --dataset conventions_drift --n 58. Unseen stream, run once if the dev rules hold: drift2_v7 with
    # --dataset conventions_drift2 --n 40.
    "learning_v7": Preset({"rule_mem": Condition("atmc", {"enable_forgetting": False, "enable_compression": False},
                            agent={"evidence_note_trust": True, "repair": True, "repair_with_memory": True, "reverify_stale": True},
                            retrieval={"hit_boost": False}, compression={"rule_memory": True})}),
    "drift_v7": Preset({"rule_mem": Condition("atmc", {"enable_forgetting": False, "enable_compression": False},
                            agent={"evidence_note_trust": True, "repair": True, "repair_with_memory": True, "reverify_stale": True},
                            retrieval={"hit_boost": False}, compression={"rule_memory": True})}),
    "drift2_v7": Preset({"rule_mem": Condition("atmc", {"enable_forgetting": False, "enable_compression": False},
                            agent={"evidence_note_trust": True, "repair": True, "repair_with_memory": True, "reverify_stale": True},
                            retrieval={"hit_boost": False}, compression={"rule_memory": True})}),
    # Rule memory v8, Phase 1 (docs/experiments/RULE_MEMORY_V8.md): design seeds 7, 11, 19, clean stream (--dataset conventions --n 110).
    "rm_design": Preset({
        "static_rm": Condition("static", agent={"repair": True, "repair_with_memory": True}),
        "rule_mem": Condition("atmc", {"enable_forgetting": False, "enable_compression": False}, agent={"evidence_note_trust": True, "repair": True, "repair_with_memory": True, "reverify_stale": True}, retrieval={"hit_boost": False}, compression={"rule_memory": True}),
        "rm_short": Condition("atmc", {"enable_forgetting": False, "enable_compression": False}, agent={"evidence_note_trust": True, "repair": True, "repair_with_memory": True, "reverify_stale": True}, retrieval={"hit_boost": False},
                              compression={"rule_memory": True, "rule_show_evidence": 1, "rule_show_solutions": 1}),
        "rm_evidence_slots": Condition("atmc", {"enable_forgetting": False, "enable_compression": False}, agent={"evidence_note_trust": True, "repair": True, "repair_with_memory": True, "reverify_stale": True}, retrieval={"hit_boost": False},
                                       compression={"rule_memory": True, "rule_mode": "evidence_only"}),
        "rm_k2": Condition("atmc", {"enable_forgetting": False, "enable_compression": False}, agent={"evidence_note_trust": True, "repair": True, "repair_with_memory": True, "reverify_stale": True}, retrieval={"hit_boost": False, "top_k": 2}, compression={"rule_memory": True}),
    }),
    # Stronger drift evidence (docs/experiments/DRIFT_V3.md; fixed before any run). Recurring stream: --dataset conventions_drift3 --n 147;
    # transfer stream: --dataset conventions_drift4 --n 49. Seeds 101 202 303.
    "drift_v3": Preset({
        "no_memory": Condition("none"),
        "static_rm": Condition("static", agent={"repair": True, "repair_with_memory": True}),
        "static_rv": Condition("static", agent={"repair": True, "repair_with_memory": True, "reverify_stale": True}),
        "static_lww": Condition("static", agent={"repair": True, "repair_with_memory": True}, retrieval={"last_write_wins": True}),
        "atmc_v6": Condition("atmc", {"enable_forgetting": False, "enable_compression": False},
                             agent={"evidence_note_trust": True, "repair": True, "repair_with_memory": True,
                                    "signature_repair": True, "utility_credit": True, "reverify_stale": True},
                             retrieval={"hit_boost": False, "utility_rank": True}, compression={"evidence_merge": True}),
        "rule_mem": Condition("atmc", {"enable_forgetting": False, "enable_compression": False},
                              agent={"evidence_note_trust": True, "repair": True, "repair_with_memory": True, "reverify_stale": True},
                              retrieval={"hit_boost": False}, compression={"rule_memory": True}),
    }),
    "agent": Preset({
        "atmc": Condition("atmc"),
        "atmc_planner": Condition("atmc", agent={"use_planner": True}),
        "atmc_reflection": Condition("atmc", agent={"use_reflection": True}),
        "atmc_planner_reflection": Condition("atmc", agent={"use_planner": True, "use_reflection": True}),
        "atmc_random_choice": Condition("atmc", agent={"curriculum": "random"}),
        "atmc_weakness_choice": Condition("atmc", agent={"curriculum": "weakness"}),
    }, heldout=20, pool_factor=3),
}


class NoisyChecker:
    """Wraps a domain so that, with probability p, a WRONG answer is reported as correct (a false success). The true
    result of the latest check is kept in `last_true_correct` so the experiment can still score honestly."""

    def __init__(self, domain, p: float, seed: int):
        self._d, self._p, self._rng = domain, p, random.Random(f"noise-{seed}")
        self.last_true_correct = None

    def __getattr__(self, name):
        return getattr(self._d, name)

    def check(self, task, answer):
        v = self._d.check(task, answer)
        self.last_true_correct = v.correct
        if not v.correct and v.verified and v.stage in ("wrong_answer", "runtime_error") and self._rng.random() < self._p:
            return Verdict(True, "ok", "false success (simulated checker error)")
        return v


def condition_cfg(cfg, cond: Condition) -> Config:
    """Experiments follow the pool in a fixed order unless the condition says otherwise."""
    d = cfg.to_dict()
    d["agent"] = {**d["agent"], "curriculum": "fixed", **cond.agent}
    d["retrieval"] = {**d["retrieval"], **cond.retrieval}
    d["composite"] = {**d["composite"], **cond.composite}
    d["compression"] = {**d["compression"], **cond.compression}
    return Config(d)


_OUTCOME_LINE = re.compile(r"^Outcome:.*$", re.MULTILINE)


def relabel_as_correct(text: str) -> str:
    """A planted note as a *false success*: same wrong solution, but its outcome line now says CORRECT."""
    return _OUTCOME_LINE.sub("Outcome: CORRECT", text)


def trust_auc(genuine: list[float], poison: list[float]) -> float | None:
    """P(a random genuine note is trusted more than a random planted one); ties count half. 0.5 = no separation."""
    if not genuine or not poison:
        return None
    wins = sum((g > p) + 0.5 * (g == p) for g in genuine for p in poison)
    return wins / (len(genuine) * len(poison))


def _frac(steps: list[dict], status: str) -> float:
    """Share of steps (that had retrieved notes) whose USED line was read as `status`."""
    with_notes = [x for x in steps if x.get("cite_status") not in (None, "", "no_memories")]
    return sum(x["cite_status"] == status for x in with_notes) / len(with_notes) if with_notes else 0.0


def trust_metrics(store) -> dict:
    """How well did the store's trust scores separate genuine notes from planted ones at the end of the run?"""
    items = getattr(store, "items", [])
    gate = getattr(store, "t_min", 0.0)
    genuine = [m.trust for m in items if not m.is_poison and m.correct is True]
    poison = [m.trust for m in items if m.is_poison]
    return {
        "trust_auc": trust_auc(genuine, poison),
        "genuine_trust_final": sum(genuine) / len(genuine) if genuine else None,
        "poison_trust_final": sum(poison) / len(poison) if poison else None,
        "genuine_blocked_frac": sum(t < gate for t in genuine) / len(genuine) if genuine else None,
        "poison_blocked_frac": sum(t < gate for t in poison) / len(poison) if poison else None,
        "n_poison_remaining": len(poison),
    }


def _store_codes(store, domain, m) -> list[str]:
    if getattr(m, "slot", None) is not None and hasattr(store, "solutions_of"):
        return list(store.solutions_of(m))
    one = domain.note_solution(m.text)
    return [one] if one else []


def _is_visible(store, m) -> bool:
    """Would the memory still offer this note to the model? (plain memory shows everything it holds)"""
    if getattr(m, "stale", False):
        return False
    t_min = getattr(store, "t_min", None)
    if t_min is not None and getattr(store, "enable_trust", False) and m.trust < t_min:
        return False
    return bool(m.text) if getattr(m, "slot", None) is not None else True


def hygiene_audit(store, domain, phase_tasks: list[dict]) -> dict:
    """Re-run every stored verified solution against the CURRENT checker (the task of the same function in the phase that just ended).
    wrongly_shown: fails now but is still offered. wrongly_hidden: passes now but is not offered (premature retirement)."""
    import re as _re
    by_ep = {t["entry_point"]: t for t in phase_tasks if t.get("entry_point")}
    n = ws = wh = 0
    for m in getattr(store, "items", []):
        for code in _store_codes(store, domain, m):
            ep = _re.search(r"def (\w+)\(", code)
            task = by_ep.get(ep.group(1)) if ep else None
            ok = domain.reverify(task, code) if task else None
            if ok is None:
                continue
            n += 1
            vis = _is_visible(store, m)
            ws += (not ok) and vis
            wh += ok and (not vis)
    return {"audited": n, "wrongly_shown": ws, "wrongly_hidden": wh}


def evaluate_frozen(agent: Agent, task: dict) -> bool:
    """Score one unseen task with memory read-only."""
    saved = [(m, m.hits) for m in agent.store.items]
    try:
        plan = agent.planner.plan(task) if agent.planner else None
        retrieved = agent.store.retrieve(agent.embedder.embed(task["question"]), k=agent.cfg.retrieval.top_k)
        raw = llm.complete(agent.domain.build_prompt(task, [m.text for m in retrieved], plan))
        return agent.domain.check(task, agent.domain.extract_answer(raw, task)).correct
    finally:
        for m, h in saved:
            m.hits = h


def run_one(exp: str, cname: str, cond: Condition, preset: Preset, seed: int, cfg, embedder, domain, n: int,
            heldout: int, force: bool) -> str:
    path = OUT_DIR / f"{exp}__{domain.name}__{cname}__seed{seed}.json"
    if path.exists() and not force:
        return "skipped (already done)"

    ccfg = condition_cfg(cfg, cond)
    pool_n = n * preset.pool_factor
    tasks = domain.load_tasks(cfg, pool_n + heldout, seed)
    train_pool, held = tasks[:pool_n], tasks[pool_n:pool_n + heldout]
    sequence = train_pool[:n]

    store = build_store(cond.kind, ccfg, embedder, llm.summarize, **cond.pillars)
    n_poison = 0
    if preset.poison:
        as_correct = preset.poison_label == "correct"
        for text, emb in domain.poison_specs(sequence, embedder, cfg.evaluation.poison_every):
            store.add(relabel_as_correct(text) if as_correct else text, emb, correct=as_correct, is_poison=True)
            n_poison += 1

    if preset.verifier_noise:
        domain = NoisyChecker(domain, preset.verifier_noise, seed)
    agent = build_agent(ccfg, store, embedder, domain, train_pool, seed)
    if ccfg.agent.curriculum == "fixed":
        agent.pool = sequence                      # identical sequence for every fixed-order condition
    usage0, t0 = llm.usage_snapshot(), time.time()
    steps, audits = [], {}
    if preset.repeats > 1:                         # same tasks again and again: pass-major order, explicit tasks
        schedule = sequence * preset.repeats
    else:
        schedule = [None] * n                      # let the agent's own curriculum choose
    n_steps = len(schedule)
    for i, task in enumerate(schedule):
        r = agent.step(task)
        log = r.to_log()
        log["phase"] = r.task.get("phase")
        if preset.verifier_noise:                  # score on the truth; keep what the agent was told
            log["recorded_correct"] = log["correct"]
            log["correct"] = bool(domain.last_true_correct)
        poison_now = [m.trust for m in getattr(store, "items", []) if m.is_poison]
        log["poison_trust_mean"] = round(sum(poison_now) / len(poison_now), 4) if poison_now else None
        steps.append(log)
        nxt = sequence[i + 1].get("phase") if i + 1 < len(sequence) else None
        if log["phase"] and nxt != log["phase"]:           # a phase just ended: audit what the memory would still show
            audits[log["phase"]] = hygiene_audit(store, domain, [t for t in sequence if t.get("phase") == log["phase"]])
        r_ok = log["correct"]
        print(f"  [{cname} s{seed}] {i+1}/{n_steps} {'PASS' if r_ok else 'fail'} ({r.verdict.stage}) "
              f"poison_in_ctx={r.n_poison_retrieved} mem={r.memory_size}", flush=True)

    held_correct, held_detail = None, []
    if heldout:
        held_detail = [{"task_id": t["id"], "correct": evaluate_frozen(agent, t)} for t in held]
        held_correct = sum(h["correct"] for h in held_detail) / len(held_detail)
        print(f"  [{cname} s{seed}] held-out pass rate {held_correct:.2f} on {len(held)} unseen tasks", flush=True)

    usage1 = llm.usage_snapshot()
    warm = cfg.retrieval.top_k          # first tasks before any genuine note can exist (cold start)
    ok = [s["correct"] for s in steps]
    N = len(steps)
    summary = {
        "pass_rate": sum(ok) / N,
        "pass_rate_first_half": sum(ok[: N // 2]) / max(N // 2, 1),
        "pass_rate_second_half": sum(ok[N // 2:]) / max(N - N // 2, 1),
        "heldout_pass_rate": held_correct,
        "poison_exposure": sum(s["n_poison_retrieved"] for s in steps),
        "poison_exposure_after_warmup": sum(s["n_poison_retrieved"] for s in steps[warm:]),
        "poison_used": sum(s["n_poison_used"] for s in steps),
        "n_poison_seeded": n_poison,
        "final_memory_size": steps[-1]["memory_size"],
        "n_compressions": getattr(store, "n_compressions", 0),
        "n_poison_laundered": getattr(store, "n_poison_laundered", 0),
        "llm_calls": usage1["calls"] - usage0["calls"],
        "api_calls": usage1["api_calls"] - usage0["api_calls"],
        "cache_hits": usage1["cache_hits"] - usage0["cache_hits"],
        "tokens_in": usage1["prompt_tokens"] - usage0["prompt_tokens"],
        "tokens_out": usage1["output_tokens"] - usage0["output_tokens"],
        "mean_retrieval_ms": sum(s["retrieval_ms"] for s in steps) / N,
        "n_steps": N,
        "hygiene": audits,
        "repair_attempted": sum(s.get("repair_attempted", False) for s in steps),
        "repair_passed": sum(s.get("repair_passed", False) for s in steps),
        "pass_rate_with_retry": (sum(ok) + sum(s.get("repair_passed", False) for s in steps)) / N,
        "cite_listed_frac": _frac(steps, "listed"), "cite_none_frac": _frac(steps, "none"),
        "cite_missing_frac": _frac(steps, "missing") + _frac(steps, "unparseable"),
        **trust_metrics(store),
        "wall_seconds": round(time.time() - t0, 1),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "experiment": exp, "domain": domain.name, "condition": cname, "seed": seed, "n": N, "heldout": heldout,
        "poison_label": preset.poison_label, "repeats": preset.repeats,
        "embedder": embedder.name, "condition_spec": {"kind": cond.kind, "pillars": cond.pillars, "agent": cond.agent,
                                            "retrieval": cond.retrieval},
        "config": ccfg.to_dict(), "summary": summary, "steps": steps, "heldout_steps": held_detail,
    }, indent=1))
    return f"pass {summary['pass_rate']:.2f}, poison {summary['poison_exposure']}, {summary['llm_calls']} calls"


def estimate_calls(preset: Preset, n: int, heldout: int, seeds: int, only) -> int:
    total = 0
    for name, c in preset.conditions.items():
        if only and name not in only:
            continue
        per_task = 1 + (1 if c.agent.get("use_planner") else 0) + (0.3 if c.agent.get("use_reflection") else 0)
        compress = (n + 14) // 8 if c.kind == "atmc" and c.pillars.get("enable_compression", True) else 0
        steps = n * preset.repeats
        compress = (steps + 14) // 8 if c.kind == "atmc" and c.pillars.get("enable_compression", True) else 0
        total += seeds * (steps * per_task + compress + heldout * (2 if c.agent.get("use_planner") else 1))
    return int(total)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset", nargs="?", choices=PRESETS)
    ap.add_argument("--domain", choices=["math", "code"], default="math")
    ap.add_argument("--seeds", default=None, help="comma-separated; default = evaluation.seeds from the config")
    ap.add_argument("--n", type=int, default=None, help="training tasks per run (default: evaluation.n_tasks)")
    ap.add_argument("--heldout", type=int, default=None, help="unseen tasks scored with frozen memory")
    ap.add_argument("--only", default=None, help="comma-separated condition names")
    ap.add_argument("--embedding", default=None, help="fastembed | gemini | tfidf")
    ap.add_argument("--force", action="store_true", help="re-run even if a result file exists")
    ap.add_argument("--out", default=None, help="results folder (default pilot/results/eval); use a scratch folder for smoke tests")
    ap.add_argument("--note-style", choices=["full", "evidence"], default=None, help="how failed attempts are stored (coding.note_style)")
    ap.add_argument("--dataset", default=None, help="coding task pool, comma-separated: humaneval,mbpp,conventions")
    ap.add_argument("--provider", choices=["gemini", "ollama"], default=None, help="model provider (default: from the config)")
    ap.add_argument("--model", default=None, help="model name for the provider, e.g. qwen2.5-coder:7b-instruct")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.out:
        global OUT_DIR
        OUT_DIR = Path(args.out).resolve()

    if args.list or not args.preset:
        for name, p in PRESETS.items():
            print(f"{name:18s} conditions: {', '.join(p.conditions)}  poison={p.poison} heldout={p.heldout}")
        return

    overrides = {"embedding": {"backend": args.embedding}} if args.embedding else {}
    if args.provider or args.model:
        overrides["model"] = {k: v for k, v in (("provider", args.provider), ("name", args.model)) if v}
        if args.provider == "ollama":
            overrides["model"]["request_interval_s"] = 0.0     # no rate limit on a local model
    if args.note_style:
        overrides.setdefault("coding", {})["note_style"] = args.note_style
    if args.dataset:
        overrides.setdefault("coding", {})["datasets"] = args.dataset.split(",")
    overrides = overrides or None
    cfg = load_config(overrides=overrides)
    llm.configure(cfg)
    preset = PRESETS[args.preset]
    n = args.n or cfg.evaluation.n_tasks
    heldout = args.heldout if args.heldout is not None else preset.heldout
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else cfg.evaluation.seeds
    only = set(args.only.split(",")) if args.only else None
    domain = get_domain(args.domain, cfg)
    embedder = make_embedder(cfg, corpus=[t["question"] for t in domain.load_tasks(cfg, 200, 0)])

    todo = [(c, s) for s in seeds for c in preset.conditions if not only or c in only]   # seed-major: a quota stop leaves whole seeds
    est = estimate_calls(preset, n, heldout, len(seeds), only)
    print(f"{args.preset} / {args.domain}: {len(todo)} runs, n={n}, heldout={heldout}, seeds={seeds}, "
          f"embedder={embedder.name}\n≈{est} LLM calls (free tier: 500/day for this model)\n", flush=True)

    done = skipped = 0
    try:
        for cname, seed in todo:
            print(f"== {cname} seed {seed}", flush=True)
            msg = run_one(args.preset, cname, preset.conditions[cname], preset, seed, cfg, embedder, domain, n,
                          heldout, args.force)
            print(f"   -> {msg}\n", flush=True)
            skipped += msg.startswith("skipped")
            done += not msg.startswith("skipped")
    except llm.QuotaExhausted as e:
        print(f"\nDAILY API QUOTA EXHAUSTED after {done} new run(s) ({skipped} skipped). "
              f"Completed runs are saved; re-run the same command later to continue.\n{str(e)[:160]}")
        sys.exit(3)
    print(f"finished: {done} new run(s), {skipped} already done. Results in {OUT_DIR}")


if __name__ == "__main__":
    main()
