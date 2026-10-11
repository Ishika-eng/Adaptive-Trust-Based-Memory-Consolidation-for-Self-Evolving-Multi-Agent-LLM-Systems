"""Memory stores.

- StaticMemoryStore: baseline. Stores every reflection forever, retrieves by
  pure cosine similarity, no trust weighting, no forgetting, no compression.
- AdaptiveTrustStore (ATMC -- Adaptive Trust-based Memory Consolidation): the
  proposed method, covering all four pillars from the README's Section 8
  ablation table -- Trust, Importance, Adaptive Forgetting, and Memory
  Compression. Each of the first three is independently toggleable
  (`enable_trust`, `enable_forgetting`, `enable_compression`) so the exact
  4-row ablation matrix (Full System / w/o Trust / w/o Forgetting /
  w/o Compression) can be constructed from this one class -- see
  the `ablation` preset of `experiment.py`. Importance is never toggled off, matching the README's
  own table (it has no "w/o Importance" row).

Note on naming: an unrelated prior-art paper (arXiv 2606.25161) already holds
the name "TrustMem" for a different mechanism (RL-trained memory-transition
verification). This project uses ATMC instead to avoid the collision.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from memory import MemoryItem, update_trust, importance, retention_factor, composite_score


class StaticMemoryStore:
    """Baseline: no trust scoring, no forgetting, no compression. Everything
    that's ever generated gets stored and is retrievable forever, ranked by
    similarity."""

    name = "static"

    def __init__(self, min_similarity: float = 0.0):
        self.items: list[MemoryItem] = []
        self.min_similarity = min_similarity     # 0 = always return the top-k, however dissimilar

    def add(self, text: str, embedding, correct: bool, is_poison: bool = False, source: str = "tests"):
        # unconditional storage -- no verification gate
        self.items.append(MemoryItem(text=text, embedding=embedding, correct=correct, is_poison=is_poison, source=source))

    def retrieve(self, query_embedding, k: int = 3):
        if not self.items:
            return []
        sims = cosine_similarity([query_embedding], [m.embedding for m in self.items])[0]
        order = [i for i in np.argsort(-sims)[:k] if sims[i] >= self.min_similarity]
        chosen = [self.items[i] for i in order]
        for m in chosen:
            m.hits += 1
        return chosen

    def update_from_outcome(self, retrieved_items, task_correct: bool, used_mask=None, weight: float = 1.0):
        pass  # baseline has no trust concept -- nothing to update

    def size(self):
        return len(self.items)


class NoMemoryStore:
    """Control: the agent never remembers anything."""

    name = "none"

    def __init__(self):
        self.items: list[MemoryItem] = []

    def add(self, text, embedding, correct, is_poison: bool = False, source: str = "tests"):
        pass

    def retrieve(self, query_embedding, k: int = 3):
        return []

    def update_from_outcome(self, retrieved_items, task_correct: bool, used_mask=None, weight: float = 1.0):
        pass

    def size(self):
        return 0


class ReflexionStyleStore:
    """Reflexion-style baseline: a small buffer of the most recent *failure*
    reflections, retrieved by recency alone (no similarity, no trust, no decay).
    Successes are not stored -- verbal reflection on failure is the whole idea."""

    name = "reflexion"

    def __init__(self, buffer_size: int = 3):
        self.buffer_size = buffer_size
        self.items: list[MemoryItem] = []

    def add(self, text, embedding, correct, is_poison: bool = False, source: str = "tests"):
        if correct and not is_poison:
            return
        self.items.append(MemoryItem(text=text, embedding=embedding, correct=correct, is_poison=is_poison, source=source))
        self.items = self.items[-self.buffer_size:]

    def retrieve(self, query_embedding, k: int = 3):
        chosen = self.items[-k:]
        for m in chosen:
            m.hits += 1
        return list(reversed(chosen))

    def update_from_outcome(self, retrieved_items, task_correct: bool, used_mask=None, weight: float = 1.0):
        pass

    def size(self):
        return len(self.items)


class AdaptiveTrustStore:
    """ATMC: hard verification gate, trust update, importance-weighted
    composite retrieval, adaptive decay-based pruning, and memory
    compression -- all four independently toggleable for ablation."""

    name = "atmc"

    def __init__(self, t_min: float = 0.30, eta: float = 0.05, delta: float = 0.15,
                 lambda_0: float = 0.1, alpha: float = 1.0, beta: float = 1.0,
                 w1: float = 0.4, w2: float = 0.4, w3: float = 0.2, prune_every: int = 10,
                 enable_trust: bool = True, enable_forgetting: bool = True,
                 enable_compression: bool = True,
                 compress_every: int = 8, compress_batch_size: int = 3,
                 compress_min_age_steps: int = 5,
                 initial_trust: float = 0.5, prune_min_retention: float = 0.05,
                 embed_fn=None, summarize_fn=None, label: str | None = None,
                 min_similarity: float = 0.0):
        self.items: list[MemoryItem] = []
        self.min_similarity = min_similarity     # relevance floor: never inject a note this dissimilar to the task
        self.initial_trust = initial_trust
        self.prune_min_retention = prune_min_retention
        self.t_min = t_min
        self.eta, self.delta = eta, delta
        self.decay_kwargs = dict(lambda_0=lambda_0, alpha=alpha, beta=beta)
        self.w1, self.w2, self.w3 = w1, w2, w3
        self.prune_every = prune_every
        self._n_added = 0
        self._step = 0  # task-step counter; decay Delta t is measured in steps

        self.enable_trust = enable_trust
        self.enable_forgetting = enable_forgetting
        self.enable_compression = enable_compression
        self.compress_every = compress_every
        self.compress_batch_size = compress_batch_size
        self.compress_min_age_steps = compress_min_age_steps
        self.embed_fn = embed_fn          # text -> embedding, required if enable_compression
        self.summarize_fn = summarize_fn  # list[str] -> str, required if enable_compression
        self.name = label or ("atmc" if (enable_trust and enable_forgetting and enable_compression) else "atmc-ablation")

        # bookkeeping surfaced for evaluation (e.g. did compression ever merge
        # a poisoned memory into an otherwise-clean summary -- see below)
        self.n_compressions = 0
        self.n_poison_laundered = 0

    def _new_uid(self) -> str:
        self._uid_counter = max(getattr(self, "_uid_counter", 0), max((int(m.uid[1:]) for m in self.items if m.uid[1:].isdigit()), default=0)) + 1
        return f"m{self._uid_counter}"

    def add(self, text: str, embedding, correct: bool, is_poison: bool = False, source: str = "tests"):
        self._step += 1
        if self.enable_trust:
            # Hard verification gate (README Section 5): only checker-verified
            # reflections enter the bank; the initial trust reflects that check.
            R = 1 if correct else -1
            T0 = update_trust(self.initial_trust, R, self.eta, self.delta)
            if T0 < self.t_min:
                return  # discarded at the gate -- never promoted
        else:
            T0 = 1.0  # trust pillar off: nothing is ever gated or down-weighted
        self.items.append(MemoryItem(text=text, embedding=embedding, trust=T0, correct=correct,
                                      created_at_step=self._step, is_poison=is_poison, uid=self._new_uid(),
                                      source="planted" if is_poison else source))
        self._n_added += 1

        if self.enable_forgetting and self._n_added % self.prune_every == 0:
            self._prune()
        if self.enable_compression and self._n_added % self.compress_every == 0:
            self._compress()

    def _prune(self):
        kept = []
        for m in self.items:
            dt = max(self._step - m.created_at_step, 0)
            I = importance(1.0, m.hits)  # self-importance proxy at prune time
            F = retention_factor(m.trust, I, dt, **self.decay_kwargs)
            if m.pinned or (m.trust >= self.t_min and F > self.prune_min_retention):
                kept.append(m)
        self.items = kept

    def _compress(self):
        """Memory Compression (README Section 8.4 / 8.1): periodically merge
        old, rarely-retrieved memories into a single shorter summary, instead
        of keeping every raw reflection verbatim forever. This is the fourth
        ATMC pillar -- previously unimplemented.

        Candidates: memories old enough to be stable (created_at_step at
        least `compress_min_age_steps` in the past) and with the fewest hits
        (least "important" by retrieval frequency) -- these are the ones
        contributing the most token/storage cost for the least retrieval
        value. Trust and poison status are deliberately NOT filtered on here:
        compression is a storage optimization, not a verification step, so a
        low-trust or poisoned memory can still be a compression candidate.
        That's worth tracking, not hiding -- see `n_poison_laundered` below.
        """
        if self.embed_fn is None or self.summarize_fn is None:
            raise RuntimeError("enable_compression=True requires embed_fn and summarize_fn")

        candidates = [m for m in self.items if not m.pinned and (self._step - m.created_at_step) >= self.compress_min_age_steps]
        if len(candidates) < self.compress_batch_size:
            return

        candidates.sort(key=lambda m: m.hits)
        batch = candidates[: self.compress_batch_size]
        batch_ids = set(id(m) for m in batch)

        summary_text = self.summarize_fn([m.text for m in batch])
        merged_embedding = self.embed_fn(summary_text)
        merged_trust = sum(m.trust for m in batch) / len(batch)
        merged_hits = sum(m.hits for m in batch)
        merged_is_poison = any(m.is_poison for m in batch)

        if merged_is_poison and not all(m.is_poison for m in batch):
            # a poisoned memory got folded into a summary alongside clean
            # ones -- the resulting single memory still carries the poison
            # flag (so pilot metrics stay honest), but this is exactly the
            # failure mode worth reporting: compression can obscure which
            # specific claim in a merged note is the untrustworthy one.
            self.n_poison_laundered += 1

        merged = MemoryItem(
            text=summary_text, embedding=merged_embedding, trust=merged_trust,
            hits=merged_hits, created_at_step=max(m.created_at_step for m in batch),
            correct=None, is_poison=merged_is_poison, uid=self._new_uid(), source="compressed",
            successes=sum(m.successes for m in batch), failures=sum(m.failures for m in batch),
            version=1 + max(m.version for m in batch),
        )
        self.items = [m for m in self.items if id(m) not in batch_ids] + [merged]
        self.n_compressions += 1

    def retrieve(self, query_embedding, k: int = 3):
        self.last_explanations, self.last_filtered = [], 0
        if not self.items:
            return []
        embs = [m.embedding for m in self.items]
        sims = cosine_similarity([query_embedding], embs)[0]
        scores, detail = [], []
        for m, sim in zip(self.items, sims):
            if self.enable_trust and m.trust < self.t_min:
                scores.append(-1e9)  # trust-based filtering
                detail.append(None)
                self.last_filtered += 1
                continue
            if sim < self.min_similarity:
                scores.append(-1e9)  # relevance floor
                detail.append(None)
                continue
            I = importance(sim, m.hits)
            if self.enable_forgetting:
                dt = max(self._step - m.created_at_step, 0)
                F = retention_factor(m.trust, I, dt, **self.decay_kwargs)
            else:
                F = 1.0  # forgetting pillar off: memories never decay
            # Trust gates importance rather than just adding to it: a barely-
            # passing-the-gate memory (trust near t_min) shouldn't be able to
            # buy its way into top-k purely by looking highly relevant. Without
            # this, a poisoned memory that happens to near-duplicate the
            # current question can out-rank genuinely trustworthy memories.
            gated_I = I * m.trust if self.enable_trust else I
            score = composite_score(m.trust, gated_I, F, self.w1, self.w2, self.w3)
            scores.append(score)
            detail.append({"similarity": round(float(sim), 3), "importance": round(float(I), 3),
                           "retention": round(float(F), 3), "score": round(float(score), 3)})
        order = np.argsort(-np.array(scores))[:k]
        chosen = [self.items[i] for i in order if scores[i] > -1e8]
        self.last_explanations = [{**detail[i], "rank": r + 1} for r, i in enumerate(order) if scores[i] > -1e8]
        for m in chosen:
            m.hits += 1
        return chosen

    # ---- note management (the Memory page) ----------------------------
    def find(self, uid: str):
        return next((m for m in self.items if m.uid == uid), None)

    def remove(self, uid: str) -> bool:
        before = len(self.items)
        self.items = [m for m in self.items if m.uid != uid]
        return len(self.items) < before

    def set_pinned(self, uid: str, pinned: bool) -> bool:
        m = self.find(uid)
        if m is None:
            return False
        m.pinned = bool(pinned)
        return True

    def import_item(self, text: str, embedding, correct, **meta) -> bool:
        """Add a note from outside (an export file). It does NOT keep the trust it arrived with: it restarts at
        the initial level and must earn more, so an imported file cannot smuggle in a highly trusted wrong note.
        Returns False for a duplicate."""
        if any(m.text == text for m in self.items):
            return False
        self._step += 1
        self.items.append(MemoryItem(text=text, embedding=embedding, trust=self.initial_trust, correct=correct,
                                      created_at_step=self._step, is_poison=bool(meta.get("is_poison")),
                                      uid=self._new_uid(), source="imported",
                                      successes=int(meta.get("successes", 0)), failures=int(meta.get("failures", 0)),
                                      version=int(meta.get("version", 1))))
        self._n_added += 1
        return True

    def update_from_outcome(self, retrieved_items, task_correct: bool, used_mask=None, weight: float = 1.0):
        # Trust reflects "reliability... based on downstream task performance"
        # (README 2.1) -- but only for memories that actually contributed to
        # that outcome. Crediting/blaming every retrieved memory for the
        # overall task result (regardless of whether it was ever used) is a
        # credit-assignment gap: a poisoned memory that gets retrieved but
        # ignored still gets rewarded whenever the model succeeds anyway
        # despite it, which is exactly backwards. `used_mask[i]` -- the
        # model's own citation of which retrieved items it relied on -- gates
        # the update so an ignored memory's trust holds steady instead.
        if used_mask is None:
            used_mask = [True] * len(retrieved_items)
        for m, used in zip(retrieved_items, used_mask):     # evidence counts are kept whatever the trust setting
            if used:
                m.successes += int(task_correct)
                m.failures += int(not task_correct)
                m.last_used_step = self._step
        if not self.enable_trust:
            return  # trust pillar off: nothing to update
        R = 1 if task_correct else -1
        for m, used in zip(retrieved_items, used_mask):
            if used:
                m.trust = update_trust(m.trust, R, self.eta * weight, self.delta * weight)   # weaker evidence moves trust less

    def size(self):
        return len(self.items)
