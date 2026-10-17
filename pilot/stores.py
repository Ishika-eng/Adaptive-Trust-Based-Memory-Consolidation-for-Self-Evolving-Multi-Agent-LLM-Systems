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

import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from memory import MemoryItem, update_trust, importance, retention_factor, composite_score


def _allowed(text: str, show_kinds) -> bool:
    """Abstention by kind: with show_kinds=None every note may be shown; otherwise only notes of those kinds (example, solved,
    card, evidence). Notes of any other kind are never injected into a prompt -- the assistant 'abstains' from them."""
    if not show_kinds:
        return True
    from consolidate import note_kind
    return note_kind(text) in show_kinds


class StaticMemoryStore:
    """Baseline: no trust scoring, no forgetting, no compression. Everything
    that's ever generated gets stored and is retrievable forever, ranked by
    similarity."""

    name = "static"

    def __init__(self, min_similarity: float = 0.0, show_kinds=None, last_write_wins: bool = False):
        self.items: list[MemoryItem] = []
        self.min_similarity = min_similarity     # 0 = always return the top-k, however dissimilar
        self.show_kinds = tuple(show_kinds) if show_kinds else None
        self.last_write_wins = last_write_wins   # rival for rule changes: a new verified solution for F deletes every older note about F

    def add(self, text: str, embedding, correct: bool, is_poison: bool = False, source: str = "tests", claim_verified: bool = False):
        if self.last_write_wins and correct and "Outcome: CORRECT" in text:
            m_ = re.search(r"def (\w+)\(", text)
            if m_:
                key = m_.group(1) + "("
                self.items = [m for m in self.items if key not in m.text]
        # otherwise unconditional storage -- no verification gate
        self.items.append(MemoryItem(text=text, embedding=embedding, correct=correct, is_poison=is_poison, source=source))

    def mark_stale(self, note, **_):
        """Plain memory's only response to a note proven wrong: drop that one note (supersession)."""
        self.items = [m for m in self.items if m is not note]
        return 0

    def retrieve(self, query_embedding, k: int = 3):
        if not self.items:
            return []
        sims = cosine_similarity([query_embedding], [m.embedding for m in self.items])[0]
        if self.show_kinds:
            sims = np.where([_allowed(m.text, self.show_kinds) for m in self.items], sims, -np.inf)
        order = [i for i in np.argsort(-sims)[:k] if sims[i] >= self.min_similarity and np.isfinite(sims[i])]
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

    def add(self, text, embedding, correct, is_poison: bool = False, source: str = "tests", claim_verified: bool = False):
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

    def add(self, text, embedding, correct, is_poison: bool = False, source: str = "tests", claim_verified: bool = False):
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
                 min_similarity: float = 0.0, hit_boost: bool = True,
                 evidence_merge: bool = False, evidence_threshold: float = 0.75, evidence_max_lines: int = 6,
                 utility_rank: bool = False, show_kinds=None,
                 rule_memory: bool = False, rule_threshold: float = 0.75, rule_max_evidence: int = 6, rule_max_solutions: int = 2,
                 rule_mode: str = "full", rule_show_evidence=None, rule_show_solutions=None):
        self.items: list[MemoryItem] = []
        self.show_kinds = tuple(show_kinds) if show_kinds else None
        self.rule_memory = rule_memory                 # one slot per rule, overwritten when the rule changes (consolidate.render_slot)
        self.rule_threshold, self.rule_max_evidence, self.rule_max_solutions = rule_threshold, rule_max_evidence, rule_max_solutions
        self.rule_mode = rule_mode                     # 'full': evidence and solutions both live in slots; 'evidence_only': only evidence is slotted
        self.rule_show = (rule_show_evidence, rule_show_solutions) if (rule_show_evidence or rule_show_solutions) else None  # how much of a slot a prompt shows
        self.utility_rank = utility_rank           # rank by similarity x learned utility (see record_outcome), not the composite score
        self.kind_stats: dict[str, list[int]] = {}  # note kind -> [answers that passed, answers it was shown for]
        self.evidence_merge = evidence_merge       # fold similar failure-evidence notes into one rule card (consolidate.py)
        self.evidence_threshold, self.evidence_max_lines = evidence_threshold, evidence_max_lines
        self.n_cards_built = 0
        self.hit_boost = hit_boost               # False: importance ignores how often a note was retrieved (no lock-in)
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

    def add(self, text: str, embedding, correct: bool, is_poison: bool = False, source: str = "tests", claim_verified: bool = False):
        """claim_verified: the note only states something the checker verified (e.g. "this call returned X where Y was
        expected"), so it enters with the trust of a verified note even though the attempt it describes FAILED."""
        self._step += 1
        if self.enable_trust:
            # Hard verification gate (README Section 5): only checker-verified
            # reflections enter the bank; the initial trust reflects that check.
            R = 1 if (correct or claim_verified) else -1
            T0 = update_trust(self.initial_trust, R, self.eta, self.delta)
            if T0 < self.t_min:
                return  # discarded at the gate -- never promoted
        else:
            T0 = 1.0  # trust pillar off: nothing is ever gated or down-weighted
        if self.rule_memory and self._slot_add(text, embedding, T0, claim_verified):
            return
        if self.evidence_merge and claim_verified and self._fold_into_card(text, embedding, T0):
            return
        self.items.append(MemoryItem(text=text, embedding=embedding, trust=T0, correct=correct,
                                      created_at_step=self._step, is_poison=is_poison, uid=self._new_uid(),
                                      source="planted" if is_poison else source))
        self._n_added += 1

        if self.enable_forgetting and self._n_added % self.prune_every == 0:
            self._prune()
        if self.enable_compression and self._n_added % self.compress_every == 0:
            self._compress()

    # ---- learned utility (ATMC v4): which KIND of note actually helps, and which notes within a kind ----
    def _kind_rate(self, kind: str) -> float:
        w, n = self.kind_stats.get(kind, (0, 0))
        return (w + 1) / (n + 2)

    def utility(self, m) -> float:
        from consolidate import note_kind
        shown, wins = getattr(m, "u_shown", 0), getattr(m, "u_wins", 0)
        return (wins + 2 * self._kind_rate(note_kind(m.text))) / (shown + 2)

    def record_outcome(self, shown_notes, passed: bool) -> None:
        """Credit EVERY note that was in the prompt with the outcome of the answer it was shown for (no citation needed)."""
        from consolidate import note_kind
        for m in shown_notes:
            m.u_shown = getattr(m, "u_shown", 0) + 1
            m.u_wins = getattr(m, "u_wins", 0) + int(passed)
            ks = self.kind_stats.setdefault(note_kind(m.text), [0, 0])
            ks[0] += int(passed)
            ks[1] += 1

    def retrieve_for_repair(self, query_embedding, signatures, k: int = 3):
        """Notes for the retry: ranked like retrieve(), plus a bonus for notes about the same kind of failure, so a rule seen in one
        function reaches a differently-worded one."""
        from consolidate import note_signatures
        self.last_explanations, self.last_filtered = [], 0
        if not self.items:
            return []
        sims = cosine_similarity([query_embedding], [m.embedding for m in self.items])[0]
        scores = []
        for m, sim in zip(self.items, sims):
            if not _allowed(m.text, self.show_kinds) or (self.enable_trust and m.trust < self.t_min):
                scores.append(-1e9); continue
            bonus = 0.3 if signatures and (note_signatures(m.text) & set(signatures)) else 0.0
            scores.append(float(sim) * (0.5 + self.utility(m)) + bonus)
        order = np.argsort(-np.array(scores))[:k]
        chosen = [self.items[i] for i in order if scores[i] > -1e8]
        for m in chosen:
            m.hits += 1
        return chosen

    # ---- rule memory (v7): one slot per rule, bounded, overwritten when the rule changes ----
    @staticmethod
    def _cos(a, b) -> float:
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        return float(a @ b / ((np.linalg.norm(a) or 1.0) * (np.linalg.norm(b) or 1.0)))

    def _slot_match(self, embedding, sigs=None):
        best, best_sim = None, -1.0
        for m in self.items:
            slot = getattr(m, "slot", None)
            if slot is None or m.pinned:
                continue
            if sigs and slot["sigs"] and not (slot["sigs"] & sigs):
                continue                                   # evidence of a different kind of failure is a different rule
            sim = self._cos(embedding, m.embedding)
            if sim > best_sim:
                best, best_sim = m, sim
        return best if best is not None and best_sim >= self.rule_threshold else None

    def _slot_render(self, m) -> None:
        from consolidate import render_slot
        sl = m.slot
        m.text = render_slot(sl["evidence"], sl["solutions"], sl["n"])
        m.version += 1

    def _slot_new(self, embedding, T0):
        e = np.asarray(embedding, dtype=float)
        m = MemoryItem(text="", embedding=e / (np.linalg.norm(e) or 1.0), trust=T0, correct=None, created_at_step=self._step,
                       uid=self._new_uid(), source="rule_memory")
        m.slot = {"evidence": [], "solutions": [], "ev_emb": [], "sol_emb": [], "sigs": set(), "n": 0}
        m.stale = False
        self.items.append(m)
        return m

    def _slot_touch(self, m, embedding, T0) -> None:
        """Absorb one more task into the slot: move its centroid, refound it if it had been demoted, keep the best trust."""
        sl = m.slot
        if m.trust < self.t_min:                           # an obsolete (demoted) slot: the new evidence starts it afresh
            sl.update(evidence=[], solutions=[], ev_emb=[], sol_emb=[], sigs=set(), n=0)
            m.trust, m.created_at_step, m.stale = T0, self._step, False
            m.embedding = np.asarray(embedding, dtype=float) / (np.linalg.norm(embedding) or 1.0)
        else:
            w = sl["n"] / (sl["n"] + 1)
            e = np.asarray(m.embedding, dtype=float) * w + np.asarray(embedding, dtype=float) / (np.linalg.norm(embedding) or 1.0) * (1 - w)
            m.embedding = e / (np.linalg.norm(e) or 1.0)
            m.trust = max(m.trust, T0)
        sl["n"] += 1

    def _slot_add(self, text, embedding, T0, claim_verified) -> bool:
        from consolidate import _pairs_from_note, extract_solution, is_evidence_note, note_signatures
        if claim_verified and is_evidence_note(text):
            task, check = _pairs_from_note(text)[0]
            sigs = note_signatures(text)
            m = self._slot_match(embedding, sigs) or self._slot_new(embedding, T0)
            self._slot_touch(m, embedding, T0)
            sl = m.slot
            if all(check != c for _, c in sl["evidence"]):
                sl["evidence"] = (sl["evidence"] + [(task, check)])[-self.rule_max_evidence:]
                sl["ev_emb"] = (sl["ev_emb"] + [np.asarray(embedding, dtype=float)])[-self.rule_max_evidence:]
            sl["sigs"] |= sigs
        else:
            sol = extract_solution(text)
            if sol is None or self.rule_mode == "evidence_only":
                return False                                # in 'evidence_only' mode solutions stay ordinary per-question notes
            task, code = sol
            m = self._slot_match(embedding) or self._slot_new(embedding, T0)
            self._slot_touch(m, embedding, T0)
            sl = m.slot
            keep = [i for i, s_ in enumerate(sl["solutions"]) if s_[0] != task]
            sl["solutions"] = ([sl["solutions"][i] for i in keep] + [(task, code)])[-self.rule_max_solutions:]
            sl["sol_emb"] = ([sl["sol_emb"][i] for i in keep] + [np.asarray(embedding, dtype=float)])[-self.rule_max_solutions:]
        self._slot_render(m)
        return True

    def _slot_view(self, m, query_embedding) -> None:
        """Show only the part of a slot nearest to the question (rule_show = how many evidence lines, how many solutions)."""
        from consolidate import render_slot
        sl, (n_ev, n_sol) = m.slot, self.rule_show
        def nearest(items, embs, n):
            if not n or not items:
                return []
            order = sorted(range(len(items)), key=lambda i: -self._cos(query_embedding, embs[i]))[:n]
            return [items[i] for i in sorted(order)]
        m.text = render_slot(nearest(sl["evidence"], sl["ev_emb"], n_ev), nearest(sl["solutions"], sl["sol_emb"], n_sol), sl["n"])

    def solutions_of(self, m) -> list[str]:
        slot = getattr(m, "slot", None)
        return [code for _, code in slot["solutions"]] if slot is not None else []

    def mark_stale(self, note, radius: float = 0.75, penalty: float = 0.35) -> int:
        """A stored solution was PROVEN wrong by the current checker (a rule changed). Its trust drops to zero, and every older note
        about similar tasks (cosine >= radius) loses `penalty` of trust at once, so the whole obsolete neighbourhood falls below the
        trust gate together instead of one note per failure. Notes created after the stale one are left alone (they may be new-rule)."""
        if getattr(note, "slot", None) is not None:        # a rule slot: its rule changed, so the slot is emptied and gated
            born = note.created_at_step
            hit = 0
            for m in self.items:
                if m is note or m.pinned or m.created_at_step > born:
                    continue
                if self._cos(note.embedding, m.embedding) >= radius:
                    m.trust = max(0.0, m.trust - penalty)
                    hit += 1
            note.slot.update(evidence=[], solutions=[], ev_emb=[], sol_emb=[], sigs=set(), n=0)
            note.trust, note.stale, note.created_at_step, note.text = 0.0, False, self._step, ""
            self.n_stale_marked = getattr(self, "n_stale_marked", 0) + 1
            self.n_stale_propagated = getattr(self, "n_stale_propagated", 0) + hit
            return hit
        note.trust, note.stale = 0.0, True
        ref = np.asarray(note.embedding, dtype=float)
        ref = ref / (np.linalg.norm(ref) or 1.0)
        hit = 0
        for m in self.items:
            if m is note or m.pinned or m.created_at_step > note.created_at_step:
                continue
            e = np.asarray(m.embedding, dtype=float)
            if float(ref @ e / (np.linalg.norm(e) or 1.0)) >= radius:
                m.trust = max(0.0, m.trust - penalty)
                hit += 1
        self.n_stale_marked = getattr(self, "n_stale_marked", 0) + 1
        self.n_stale_propagated = getattr(self, "n_stale_propagated", 0) + hit
        return hit

    def _fold_into_card(self, text: str, embedding, trust: float) -> bool:
        """Merge a new failure-evidence note with the most similar earlier evidence note or rule card (cosine above the
        threshold). Returns True when it was merged, False when it should be stored as a note of its own."""
        from consolidate import build_card, is_evidence_note
        if not is_evidence_note(text):
            return False
        cand = [(i, m) for i, m in enumerate(self.items) if not m.pinned and not m.is_poison and is_evidence_note(m.text)
                or (not m.pinned and m.text.startswith("Rule evidence from"))]
        if not cand:
            return False
        sims = cosine_similarity([embedding], [m.embedding for _, m in cand])[0]
        k = int(np.argmax(sims))
        if sims[k] < self.evidence_threshold:
            return False
        i, m = cand[k]
        members = getattr(m, "n_members", 1)
        w = members / (members + 1)
        merged_emb = np.asarray(m.embedding) * w + np.asarray(embedding) * (1 - w)
        merged_emb = merged_emb / (np.linalg.norm(merged_emb) or 1.0)
        card = MemoryItem(text=build_card([m.text, text], self.evidence_max_lines), embedding=merged_emb,
                          trust=max(m.trust, trust), hits=m.hits, created_at_step=self._step, correct=False,
                          uid=self._new_uid(), source="rule_card", successes=m.successes, failures=m.failures,
                          version=m.version + 1)
        card.n_members = members + 1
        self.items[i] = card
        self.n_cards_built += 1
        return True

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
            if not _allowed(m.text, self.show_kinds):
                scores.append(-1e9)  # abstain from this kind of note
                detail.append(None)
                continue
            if self.enable_trust and m.trust < self.t_min:
                scores.append(-1e9)  # trust-based filtering
                detail.append(None)
                self.last_filtered += 1
                continue
            if sim < self.min_similarity:
                scores.append(-1e9)  # relevance floor
                detail.append(None)
                continue
            if self.utility_rank:
                scores.append(float(sim) * (0.5 + self.utility(m)))
                detail.append({"similarity": round(float(sim), 3), "utility": round(self.utility(m), 3)})
                continue
            I = importance(sim, m.hits if self.hit_boost else 0)
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
        if self.rule_memory and self.rule_show:
            for m in chosen:
                if getattr(m, "slot", None) is not None:
                    self._slot_view(m, query_embedding)
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
