"""Strategy evolution: the assistant learns *how* to solve each kind of problem, not just what it remembers.

Champion / challenger, per problem type:
  - There is a stable default (the baseline). It is never removed.
  - Other strategies are challengers. A challenger becomes champion for a problem type only when it has
    enough trials and the lower confidence bound of its success rate reaches the baseline's own rate
    (a Wilson bound, so a lucky 2-for-2 does not count).
  - The champion is recomputed from the evidence every time, so rollback is automatic: if a promoted
    strategy's record worsens, it simply stops qualifying and the baseline takes over again.
  - Exploration happens while *practising* (benchmark tasks with real tests, outcomes verified), where the
    least-tried strategy is used; live user questions use the champion, with a small chance of exploring.

Problem types come from nearest-prototype matching with the same embedder as memory -- no extra LLM call.
"""
from __future__ import annotations

import math
import random

import numpy as np

STRATEGIES = {
    "plan_first": "Plan the approach first (without looking at memory), then write the code",
    "direct": "Write the code straight away, with no separate planning step",
    "revise": "Write a draft, then review it for bugs and missed edge cases and fix it",
}
BASELINE = "plan_first"

CATEGORIES = {
    "strings": ("Strings and text", "Manipulate or check strings, text, characters, palindromes, words and substrings"),
    "lists": ("Lists and arrays", "Process lists, arrays, tuples and matrices: sums, filters, rotations, maximum and minimum"),
    "numbers": ("Numbers and math", "Number theory and arithmetic: primes, factors, gcd, digits, bits, powers and sequences"),
    "search_sort": ("Searching and sorting", "Sort, search, merge, rank or order elements: binary search, sorting, intervals"),
    "recursion_dp": ("Recursion and dynamic programming", "Recursion, dynamic programming, combinatorics, paths, subsequences and counting ways"),
    "structures": ("Data structures", "Dictionaries, sets, stacks, queues, trees, graphs, frequency counting and grouping"),
}
GENERAL = "general"
MIN_SIMILARITY = 0.45


def label(category: str) -> str:
    return CATEGORIES[category][0] if category in CATEGORIES else "General"


def wilson(wins: int, n: int, z: float = 1.28) -> tuple[float, float]:
    """~80% two-sided Wilson interval for a success rate; (0, 1) when there is no data."""
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


class StrategyBook:
    def __init__(self, embedder, min_trials: int = 5, explore: float = 0.1, rng: random.Random | None = None):
        self.embedder = embedder
        self.min_trials = min_trials
        self.explore = explore
        self.rng = rng or random.Random(0)
        self.events: list[dict] = []          # {ref, category, strategy, mode, success (None until known)}
        self._protos = None

    # ---- problem type -------------------------------------------------
    def classify(self, text: str) -> str:
        if self._protos is None:
            self._protos = {k: np.asarray(self.embedder.embed(desc)) for k, (_, desc) in CATEGORIES.items()}
        q = np.asarray(self.embedder.embed(text))
        q = q / (np.linalg.norm(q) or 1.0)
        best, best_sim = GENERAL, MIN_SIMILARITY
        for k, v in self._protos.items():
            sim = float(np.dot(q, v / (np.linalg.norm(v) or 1.0)))
            if sim > best_sim:
                best, best_sim = k, sim
        return best

    # ---- evidence -----------------------------------------------------
    def stats(self, category: str) -> dict[str, tuple[int, int]]:
        out = {s: [0, 0] for s in STRATEGIES}
        for e in self.events:
            if e["category"] == category and e["success"] is not None:
                out[e["strategy"]][1] += 1
                out[e["strategy"]][0] += int(e["success"])
        return {s: (w, n) for s, (w, n) in out.items()}

    def champion(self, category: str) -> tuple[str, str]:
        """(strategy, role) where role is "baseline" or "promoted"."""
        st = self.stats(category)
        base_w, base_n = st[BASELINE]
        if base_n < self.min_trials:
            return BASELINE, "baseline"
        base_rate = base_w / base_n
        best, best_lo = BASELINE, -1.0
        for s, (w, n) in st.items():
            if s == BASELINE or n < self.min_trials:
                continue
            lo = wilson(w, n)[0]
            if lo >= base_rate and lo > best_lo:
                best, best_lo = s, lo
        return (best, "promoted") if best != BASELINE else (BASELINE, "baseline")

    # ---- choosing -----------------------------------------------------
    def _least_tried(self, category: str) -> str:
        st = self.stats(category)
        fewest = min(n for _, n in st.values())
        return self.rng.choice([s for s, (_, n) in st.items() if n == fewest])

    def choose(self, category: str, mode: str = "serve") -> tuple[str, str]:
        """(strategy, reason). mode "practise" explores; "serve" uses the champion (rarely exploring)."""
        if mode == "practise":
            s = self._least_tried(category)
            return s, "practising: trying the approach I've tested least on this kind of problem"
        champ, role = self.champion(category)
        if self.rng.random() < self.explore:
            s = self._least_tried(category)
            if s != champ:
                return s, "occasionally trying a less-tested approach"
        why = "proven best for this kind of problem" if role == "promoted" else "my default approach"
        return champ, why

    # ---- recording ----------------------------------------------------
    def start(self, ref: str, category: str, strategy: str, mode: str) -> None:
        self.events.append({"ref": ref, "category": category, "strategy": strategy, "mode": mode, "success": None})

    def resolve(self, ref: str, success: bool) -> None:
        for e in self.events:
            if e["ref"] == ref:
                e["success"] = bool(success)

    # ---- reporting ----------------------------------------------------
    def table(self) -> list[dict]:
        rows = []
        for cat in [*CATEGORIES, GENERAL]:
            st = self.stats(cat)
            if not any(n for _, n in st.values()) and cat == GENERAL:
                continue
            champ, role = self.champion(cat)
            rows.append({
                "category": cat, "label": label(cat), "champion": champ, "role": role,
                "strategies": [{
                    "strategy": s, "trials": n, "wins": w, "rate": (w / n if n else None),
                    "lower": wilson(w, n)[0] if n else None,
                    # "champion" is earned by evidence; the default is just the default
                    "role": ("champion" if (s == champ and role == "promoted") else "baseline" if s == BASELINE else "challenger"),
                    "enough_evidence": n >= self.min_trials,
                } for s, (w, n) in st.items()],
            })
        return rows
