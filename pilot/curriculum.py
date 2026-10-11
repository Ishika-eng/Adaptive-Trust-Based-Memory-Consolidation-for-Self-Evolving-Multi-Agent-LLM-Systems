"""Curriculum: the agent chooses its own next task instead of being fed a fixed list.

- fixed:     the pool in order (controlled experiments).
- random:    uniform over tasks not yet attempted (the control).
- weakness:  practise where past failures cluster. Each unattempted task is scored by
             its similarity to the agent's own *failure* memories; the highest wins.
             With probability `explore` it picks at random instead, so it cannot
             lock onto one area. Poisoned notes are excluded -- they are planted,
             not experienced.
"""
from __future__ import annotations

import random

import numpy as np


class Curriculum:
    def choose(self, pool, done_ids, store, embedder, rng: random.Random) -> dict:
        raise NotImplementedError


class FixedOrderCurriculum(Curriculum):
    """Follow the pool in order -- identical task sequence for every condition,
    which is what a controlled comparison needs."""

    def choose(self, pool, done_ids, store, embedder, rng):
        for t in pool:
            if t["id"] not in done_ids:
                return t
        raise StopIteration("task pool exhausted")


class RandomCurriculum(Curriculum):
    def choose(self, pool, done_ids, store, embedder, rng):
        remaining = [t for t in pool if t["id"] not in done_ids]
        if not remaining:
            raise StopIteration("task pool exhausted")
        return rng.choice(remaining)


class WeaknessCurriculum(Curriculum):
    def __init__(self, explore: float = 0.2):
        self.explore = explore
        self.last_mode = None      # "cold" (no failures yet) | "explore" (random) | "weakness" -- why the last pick

    def choose(self, pool, done_ids, store, embedder, rng):
        remaining = [t for t in pool if t["id"] not in done_ids]
        if not remaining:
            raise StopIteration("task pool exhausted")
        failures = [m for m in store.items if m.correct is False and not m.is_poison]
        if not failures:
            self.last_mode = "cold"
            return rng.choice(remaining)
        if rng.random() < self.explore:
            self.last_mode = "explore"
            return rng.choice(remaining)
        self.last_mode = "weakness"
        fail_vecs = np.array([m.embedding for m in failures])
        scores = [float(np.max(fail_vecs @ embedder.embed(t["question"]))) for t in remaining]
        return remaining[int(np.argmax(scores))]

    @staticmethod
    def related_failure(task, store, embedder):
        """The agent's own failure note closest to `task` (why the weakness curriculum picks it), or None."""
        failures = [m for m in store.items if m.correct is False and not m.is_poison]
        if not failures:
            return None
        q = embedder.embed(task["question"])
        return max(failures, key=lambda m: float(np.dot(m.embedding, q)))


def make_curriculum(cfg) -> Curriculum:
    if cfg.agent.curriculum == "fixed":
        return FixedOrderCurriculum()
    if cfg.agent.curriculum == "weakness":
        return WeaknessCurriculum(cfg.agent.curriculum_explore)
    return RandomCurriculum()
