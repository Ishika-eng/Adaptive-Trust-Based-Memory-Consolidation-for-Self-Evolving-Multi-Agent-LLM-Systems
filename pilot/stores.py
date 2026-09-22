"""Two memory stores compared in the pilot:

- StaticMemoryStore: baseline. Stores every reflection forever, retrieves by
  pure cosine similarity (+ mild recency), no trust weighting, no forgetting.
- AdaptiveTrustStore (ATMC -- Adaptive Trust-based Memory Consolidation): the
  proposed method. Trust-weighted composite retrieval, a hard verification
  gate before promotion, and adaptive decay-based pruning.

Note on naming: an unrelated prior-art paper (arXiv 2606.25161) already holds
the name "TrustMem" for a different mechanism (RL-trained memory-transition
verification). This project uses ATMC instead to avoid the collision.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from memory import MemoryItem, update_trust, importance, retention_factor, composite_score


class StaticMemoryStore:
    """Baseline: no trust scoring, no forgetting. Everything that's ever
    generated gets stored and is retrievable forever, ranked by similarity."""

    name = "static"

    def __init__(self):
        self.items: list[MemoryItem] = []

    def add(self, text: str, embedding, correct: bool, is_poison: bool = False):
        # unconditional storage -- no verification gate
        self.items.append(MemoryItem(text=text, embedding=embedding, correct=correct, is_poison=is_poison))

    def retrieve(self, query_embedding, k: int = 3):
        if not self.items:
            return []
        sims = cosine_similarity([query_embedding], [m.embedding for m in self.items])[0]
        order = np.argsort(-sims)[:k]
        chosen = [self.items[i] for i in order]
        for m in chosen:
            m.hits += 1
        return chosen

    def update_from_outcome(self, retrieved_items, task_correct: bool, used_mask=None):
        pass  # baseline has no trust concept -- nothing to update

    def size(self):
        return len(self.items)


class AdaptiveTrustStore:
    """Proposed method (ATMC): hard verification gate, trust update, importance-
    weighted composite retrieval, adaptive decay-based pruning."""

    name = "atmc"

    def __init__(self, t_min: float = 0.30, eta: float = 0.05, delta: float = 0.15,
                 lambda_0: float = 0.1, alpha: float = 1.0, beta: float = 1.0,
                 w1: float = 0.4, w2: float = 0.4, w3: float = 0.2, prune_every: int = 10):
        self.items: list[MemoryItem] = []
        self.t_min = t_min
        self.eta, self.delta = eta, delta
        self.decay_kwargs = dict(lambda_0=lambda_0, alpha=alpha, beta=beta)
        self.w1, self.w2, self.w3 = w1, w2, w3
        self.prune_every = prune_every
        self._n_added = 0
        self._step = 0  # task-step counter; decay Delta t is measured in steps

    def add(self, text: str, embedding, correct: bool, is_poison: bool = False):
        # Hard verification gate (README Section 5): only checker-verified
        # reflections enter the bank; the initial trust reflects that check.
        self._step += 1
        R = 1 if correct else -1
        T0 = update_trust(0.5, R, self.eta, self.delta)
        if T0 < self.t_min:
            return  # discarded at the gate -- never promoted
        self.items.append(MemoryItem(text=text, embedding=embedding, trust=T0,
                                      correct=correct, created_at_step=self._step, is_poison=is_poison))
        self._n_added += 1
        if self._n_added % self.prune_every == 0:
            self._prune()

    def _prune(self):
        kept = []
        for m in self.items:
            dt = max(self._step - m.created_at_step, 0)
            I = importance(1.0, m.hits)  # self-importance proxy at prune time
            F = retention_factor(m.trust, I, dt, **self.decay_kwargs)
            if m.trust >= self.t_min and F > 0.05:
                kept.append(m)
        self.items = kept

    def retrieve(self, query_embedding, k: int = 3):
        if not self.items:
            return []
        embs = [m.embedding for m in self.items]
        sims = cosine_similarity([query_embedding], embs)[0]
        scores = []
        for m, sim in zip(self.items, sims):
            if m.trust < self.t_min:
                scores.append(-1e9)  # trust-based filtering
                continue
            I = importance(sim, m.hits)
            dt = max(self._step - m.created_at_step, 0)
            F = retention_factor(m.trust, I, dt, **self.decay_kwargs)
            # Trust gates importance rather than just adding to it: a barely-
            # passing-the-gate memory (trust near t_min) shouldn't be able to
            # buy its way into top-k purely by looking highly relevant. Without
            # this, a poisoned memory that happens to near-duplicate the
            # current question can out-rank genuinely trustworthy memories
            # (observed in the poison pilot: ATMC exposed far fewer poisoned
            # memories overall, but the ones that got through were used by the
            # model at a much higher rate than static's, because they were
            # concentrated in these high-relevance/low-trust cases).
            gated_I = I * m.trust
            scores.append(composite_score(m.trust, gated_I, F, self.w1, self.w2, self.w3))
        order = np.argsort(-np.array(scores))[:k]
        chosen = [self.items[i] for i in order if scores[i] > -1e8]
        for m in chosen:
            m.hits += 1
        return chosen

    def update_from_outcome(self, retrieved_items, task_correct: bool, used_mask=None):
        # Trust reflects "reliability... based on downstream task performance"
        # (README 2.1) -- but only for memories that actually contributed to
        # that outcome. Crediting/blaming every retrieved memory for the
        # overall task result (regardless of whether it was ever used) is a
        # credit-assignment gap: a poisoned memory that gets retrieved but
        # ignored still gets rewarded whenever the model succeeds anyway
        # despite it, which is exactly backwards. `used_mask[i]` -- the
        # model's own citation of which retrieved items it relied on -- gates
        # the update so an ignored memory's trust holds steady instead.
        R = 1 if task_correct else -1
        if used_mask is None:
            used_mask = [True] * len(retrieved_items)
        for m, used in zip(retrieved_items, used_mask):
            if used:
                m.trust = update_trust(m.trust, R, self.eta, self.delta)

    def size(self):
        return len(self.items)
