# 03 · The memory core

Files: `pilot/memory.py` (65 lines), `pilot/stores.py` (335 lines). This is the heart of the project and it is **pure Python plus numpy**: no LLM, no network. Build and test it first, completely, before touching the rest.

## What you are building and why

Four interchangeable memory systems that the rest of the code treats identically (same four methods):

| Store | What it does | Role in experiments |
|---|---|---|
| `NoMemoryStore` | Remembers nothing | Control |
| `StaticMemoryStore` | Stores everything forever, retrieves by similarity | The "remember everything" baseline |
| `ReflexionStyleStore` | Keeps only the 3 most recent *failure* notes, by recency | A Reflexion-style baseline |
| `AdaptiveTrustStore` | **ATMC** | The proposal |

The shared interface (this is the contract the agent depends on):

```python
store.add(text, embedding, correct, is_poison=False, source="tests")
store.retrieve(query_embedding, k=3) -> list[MemoryItem]
store.update_from_outcome(retrieved_items, task_correct, used_mask=None, weight=1.0)
store.size() -> int
store.items            # list[MemoryItem]
```

## Step 1 — `MemoryItem` (`memory.py`)

A `@dataclass` with these fields (defaults in parentheses):

| Field | Meaning |
|---|---|
| `text` | The stored note |
| `embedding` | Vector of the *question* (not the note text) — retrieval compares questions |
| `trust` (0.5) | `T` |
| `hits` (0) | Times retrieved; feeds importance |
| `created_at_step` (0) | Store step index at creation; decay is measured in *steps*, not seconds |
| `correct` (None) | Verified outcome when created; `None` for merged summaries |
| `is_poison` (False) | Marks planted notes, for experiments only |
| `uid` ("") | Stable id, e.g. `m17`, assigned by the store |
| `source` ("tests") | Who vouched: `tests`, `user`, `generated`, `planted`, `compressed`, `imported` |
| `successes`, `failures` (0) | Times *relied on* in a verified-correct / wrong answer |
| `last_used_step` (None) | When it was last relied on |
| `version` (1) | Bumped when notes are merged |
| `pinned` (False) | Protected from forgetting and merging |

The last six fields were added late (chapter 09). Start with the first seven.

## Step 2 — The four formulas (`memory.py`)

Write each as a small pure function so you can test it in isolation.

**Trust update** — rewards are small, penalties are big, and the result is clamped to [0, 1]:

```python
def update_trust(T, R_checker, eta=0.05, delta=0.15):
    assert delta > eta
    new_T = T + eta if R_checker > 0 else T - delta
    return min(1.0, max(0.0, new_T))
```

With the defaults: success 0.50 → 0.55; failure 0.50 → 0.35.

**Importance** — similarity, boosted by how often a note was retrieved (log, so it saturates):

```python
def importance(cos_sim, hits):
    return cos_sim * (1 + math.log(1 + hits))
```

**Adaptive decay** — trusted and important notes decay more slowly:

```python
def decay_rate(T, I, lambda_0=0.1, alpha=1.0, beta=1.0):
    return lambda_0 / (1 + alpha * T + beta * I)

def retention_factor(T, I, delta_t, **kw):
    return math.exp(-decay_rate(T, I, **kw) * delta_t)      # F in (0, 1]
```

**Composite retrieval score**:

```python
def composite_score(T, I, F, w1=0.4, w2=0.4, w3=0.2):
    assert abs(w1 + w2 + w3 - 1) < 1e-6
    return w1 * T + w2 * I + w3 * F
```

### Worked example (computed with the real functions)

A genuine note: trust 0.55, similarity 0.80 to the query. A planted wrong note: trust 0.35, similarity **0.95** (it is a near-copy of the task).

| | `I` | `I·T` (trust-gated) | `F` (new note) | `S = 0.4T + 0.4·I·T + 0.2F` |
|---|---|---|---|---|
| Genuine | 0.800 | 0.440 | 1.00 | **0.596** |
| Planted | 0.950 | 0.332 | 1.00 | 0.473 |

The planted note is *more similar* yet ranks lower, because trust both adds to the score directly and **multiplies importance**. That multiplication is a deliberate design decision (see the comment in `retrieve`): without it, a poisoned note that near-duplicates the question could out-rank genuinely trusted ones.

### Worked example: how long until an unused note is forgotten?

Pruning removes a note when `F ≤ 0.05`. For trust 0.55 and the "self-importance proxy" the pruner uses (`importance(1.0, hits=0) = 1.0`): `λ = 0.1/(1+0.55+1.0) = 0.0392`, so `F ≤ 0.05` needs `Δt ≥ ln(20)/0.0392 ≈ 76` steps. Our evaluation runs are 40 tasks, so **forgetting never deletes anything in a normal run** — it only lowers a note's rank. That is a measured finding, not a bug; it is why "forgetting" showed a small, inconsistent effect in the ablation.

## Step 3 — The baselines (`stores.py`)

`NoMemoryStore`: every method is a no-op or returns `[]`.

`StaticMemoryStore`:
- `add` appends unconditionally.
- `retrieve` computes `cosine_similarity` against **all** notes, takes the top-k indices with `np.argsort(-sims)`, filters by `min_similarity` (0 = off), increments `hits`, returns them.
- `update_from_outcome` does nothing (no trust concept).

`ReflexionStyleStore(buffer_size=3)`: stores only failures (or poison), keeps the last `buffer_size`, retrieves by recency, ignores the query.

## Step 4 — `AdaptiveTrustStore` (ATMC)

### Constructor

It takes the thresholds and weights from config (via `factory.build_atmc`), three **independently switchable pillars** (`enable_trust`, `enable_forgetting`, `enable_compression` — this is what makes the ablation possible), and two callbacks used by merging: `embed_fn` (text→vector) and `summarize_fn` (list of texts→one summary). `_step` counts additions; `_n_added` counts stored notes.

### `add` — the verification gate

```python
def add(self, text, embedding, correct, is_poison=False, source="tests"):
    self._step += 1
    if self.enable_trust:
        R = 1 if correct else -1
        T0 = update_trust(self.initial_trust, R, self.eta, self.delta)   # 0.55 or 0.35
        if T0 < self.t_min:        # would only happen if you configure initial trust low
            return                  # discarded at the gate
    else:
        T0 = 1.0                    # trust off: every note equally trusted
    self.items.append(MemoryItem(...trust=T0, uid=self._new_uid(),
                                  source="planted" if is_poison else source))
    self._n_added += 1
    if self.enable_forgetting  and self._n_added % self.prune_every    == 0: self._prune()
    if self.enable_compression and self._n_added % self.compress_every == 0: self._compress()
```

Important consequence: **a failed attempt is stored, with trust 0.35**, not discarded. It is useful as a warning ("this approach failed") but ranks below genuine successes (0.55). A note is blocked from retrieval only when its trust falls below 0.30.

### `retrieve` — trust-gated ranking with explanations

1. Compute cosine similarity of the query to every note (`sklearn.metrics.pairwise.cosine_similarity`; embeddings are L2-normalised so this is a dot product).
2. For each note: if `enable_trust` and `trust < t_min` → score `-1e9` (excluded) and count it in `last_filtered`; if `similarity < min_similarity` → excluded; otherwise compute `I`, `F` (or `F=1` if forgetting is off), `gated_I = I*T` (or `I` if trust is off), and `S`.
3. Take the top-k by score, drop any with the sentinel score, increment `hits`, and record `last_explanations` (rank, similarity, importance, retention, score) so the UI can show *why* each note was used.

Cost is O(number of notes) per query. That is fine up to thousands of notes; an approximate-nearest-neighbour index would be the upgrade (chapter 14).

### `update_from_outcome` — citation-conditioned credit

```python
for m, used in zip(retrieved_items, used_mask):     # evidence counts always
    if used:
        m.successes += int(task_correct); m.failures += int(not task_correct)
        m.last_used_step = self._step
if not self.enable_trust: return
R = 1 if task_correct else -1
for m, used in zip(retrieved_items, used_mask):
    if used:
        m.trust = update_trust(m.trust, R, self.eta * weight, self.delta * weight)
```

The `used_mask` comes from the model's own `USED: 1,3` line (chapter 04). **Why not update every retrieved note?** Early in the project we did, and it was wrong: a wrong note that was retrieved but *ignored* would be rewarded whenever the model succeeded anyway — exactly backwards. The archived results in `pilot/results/archive_pre_credit_fix/` are from before this fix. `weight` scales both reward and penalty (0.5 for auto-check evidence).

### `_prune` and `_compress`

Prune (every 10 additions): keep a note if it is pinned, or `trust ≥ t_min` **and** `F > 0.05`.

Compress (every 8 additions):
1. Candidates: not pinned and at least `compress_min_age_steps` (5) old. Need at least `compress_batch_size` (3).
2. Sort by `hits` ascending (stable → oldest first on ties); take the first 3.
3. `summary = summarize_fn([texts])` (one LLM call); `embedding = embed_fn(summary)`.
4. The merged note: trust = **mean** of the parents, hits = sum, `created_at_step` = the newest parent's, `correct=None`, `source="compressed"`, `successes`/`failures` = sums, `version = 1 + max(parent versions)`, `is_poison = any(parent poison)`.
5. Replace the three parents with the merged note; `n_compressions += 1`.
6. If a poisoned note was merged with clean ones, `n_poison_laundered += 1`. Compression is a *storage* step, not a verification step, so it does not filter on trust — and we deliberately **count** the case where it hides a bad note inside a summary.

### Management methods (added late)

`find(uid)`, `remove(uid)`, `set_pinned(uid, flag)`, `import_item(...)`. **`import_item` resets trust to `initial_trust`** and marks the source `imported`, so an import file cannot smuggle in a highly trusted wrong note; duplicates (same text) are skipped.

## Pitfalls actually hit

- **Decay measured in wall-clock time** would make results depend on how fast the script ran. Measure `Δt` in task steps.
- **`hits` increments on retrieval**, which changes importance. The held-out evaluation (chapter 07) therefore saves and restores `hits` so scoring unseen tasks does not teach the memory.
- **Embedding the note text instead of the question** would make notes match on their own wording. We embed the *question*, so a note is found when a similar question arrives.
- **Gate arithmetic**: with `initial=0.5, δ=0.15`, a failed note starts at 0.35, *above* the 0.30 gate. If you set the gate at 0.40 failures would be discarded instead of stored. Decide this consciously.
- **Claiming forgetting "purged" bad notes.** We once said this; the log showed the notes had actually been absorbed by merging. Verify mechanisms from logs, not intuition.

## Tests to write (all in `tests/test_stores.py`, `tests/test_memory_management.py`)

- trust: success → 0.55, failure → 0.35; clamping; `delta ≤ eta` raises.
- gate: low-trust note excluded from `retrieve` and counted in `last_filtered`.
- ranking: a more-similar planted note ranks below a trusted genuine note (the worked example above).
- credit: an unused note's trust does not change.
- weight: half weight moves trust half as much (reward and penalty).
- prune does **not** fire inside a short run (`test_prune_does_not_fire_inside_a_short_run`); after aging a note by 200 steps it does.
- compress: lineage (`version`, summed counts), pinned notes excluded, poison-laundering counter.
- import: trust reset, duplicate skipped.

## Checkpoint

`pilot/.venv/bin/python -m pytest tests/test_stores.py tests/test_memory_management.py -q` passes, and you can reproduce the two worked examples above by calling the functions by hand.
