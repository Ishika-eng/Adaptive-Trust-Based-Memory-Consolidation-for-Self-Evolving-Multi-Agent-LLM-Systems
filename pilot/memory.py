"""Core memory math: trust update, importance, adaptive decay, composite retrieval score.

Implements the formulas from the project README (Section 2), kept as plain
functions so they're easy to unit-test independently of the LLM pipeline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class MemoryItem:
    text: str            # the stored reflection (problem + reasoning summary)
    embedding: list       # TF-IDF vector (dense) for similarity
    trust: float = 0.5    # T
    hits: int = 0         # retrieval count, for importance
    created_at_step: int = 0  # task-step index at creation (not wall-clock: see decay note)
    correct: bool = None  # ground-truth verified outcome at creation time
    is_poison: bool = False  # synthetic-contamination marker, used only by the poison pilot


# ---- Trust update (README 2.1) ----
def update_trust(T: float, R_checker: int, eta: float = 0.05, delta: float = 0.15) -> float:
    """T_{t+1} = clip(T_t + eta*R - (1-R)*delta, 0, 1), delta > eta so bad memories
    lose trust faster than good ones gain it. R_checker in {+1, -1}."""
    assert delta > eta
    if R_checker > 0:
        new_T = T + eta * R_checker
    else:
        new_T = T - delta  # (1 - R_checker) term collapses to this for R in {-1,+1}
    return min(1.0, max(0.0, new_T))


# ---- Importance (README 2.2) ----
def importance(cos_sim: float, hits: int) -> float:
    """I(m) = cos(e_q, e_m) * (1 + log(1 + hits(m)))"""
    return cos_sim * (1 + math.log(1 + hits))


# ---- Adaptive decay (README 2.3) ----
# Note: delta_t is measured in task-steps (memories elapsed since creation),
# not wall-clock time -- decay should track the agent's interaction history,
# not how fast the script happens to run.
def decay_rate(T: float, I: float, lambda_0: float = 0.1, alpha: float = 1.0, beta: float = 1.0) -> float:
    return lambda_0 / (1 + alpha * T + beta * I)


def retention_factor(T: float, I: float, delta_t: float, **decay_kwargs) -> float:
    lam = decay_rate(T, I, **decay_kwargs)
    return math.exp(-lam * delta_t)


# ---- Composite retrieval score (README 2.4) ----
def composite_score(T: float, I: float, F: float, w1: float = 0.4, w2: float = 0.4, w3: float = 0.2) -> float:
    assert abs((w1 + w2 + w3) - 1.0) < 1e-6
    return w1 * T + w2 * I + w3 * F
