# 06 · The agent loop

Files: `pilot/agent.py` (187), `pilot/planner.py` (19), `pilot/reflection.py` (24), `pilot/curriculum.py` (78).

## What you are building and why

**One loop, written once.** Experiments, the benchmark lab, the chat assistant and the tests all run the same `Agent` class. If they used different loops, results from experiments would not describe the app. The loop:

```
choose task → [plan] → retrieve memories → solve → check
            → credit the memories that were used → [reflect on failures]
            → store the new note → (store forgets/merges internally)
```

## 6.1 The data carried between stages — `StepResult`

A dataclass holding everything about one attempt: `task`, `plan`, `retrieved` (the `MemoryItem`s), `used_mask`, `raw` (full model reply), `answer`, `verdict`, `lesson`, `note`, `n_poison_retrieved`, `n_poison_used`, `retrieval_ms`, `tokens_in/out`, `memory_size`, plus later additions `q_emb` (the question vector, needed to store the note), `learned` (so learning happens once), `autocheck`, `explain` (why each note was chosen) and `filtered` (notes skipped for low trust). `to_log()` gives a small JSON-safe summary saved for every step of every run.

## 6.2 Splitting `step` into `attempt` and `learn`

The first version had a single `step()`. The chat assistant forced a split: when a user's answer has **no ground truth yet**, you must produce the answer *without* touching memory, and learn later when the user decides.

### `attempt(task, strategy=None)` — never modifies memory

1. Snapshot token usage (`llm.usage_snapshot()`).
2. **Plan**: only if a planner is configured *and* the strategy is `None` or `"plan_first"`. The planner sees the task only, never memory (so a poisoned note cannot bend the plan; planning and memory are independent failure domains).
3. **Embed** the question; time the **retrieval** (`store.retrieve(q_emb, k=top_k)`).
4. **Prompt and solve**: `domain.build_prompt(task, [m.text…], plan)` → `llm.complete` → `domain.extract_answer`.
5. `used_mask = extract_used_mask(raw, n_retrieved)` — computed from the **first** draft (which notes the model relied on).
6. **Strategy "revise"**: a second LLM call asks the model to review its own draft (`domain.revise_prompt`) and the improved code replaces the answer.
7. `verdict = domain.check(task, answer)`.
8. Return a `StepResult` with token deltas.

### `learn(result, correct=None, feedback="")` — the only place memory changes

1. If already learned, return (idempotent).
2. If `verdict.verified` is False (no tests): require the user's `correct`, then build `Verdict(correct, "ok"|"user_rejected", feedback[:300], evidence="user")`. Without it, raise `ValueError`.
3. **Credit**: `store.update_from_outcome(retrieved, verdict.correct, used_mask, weight)` with `weight = autoverify.evidence_weight (0.5)` if the evidence is `generated`, else 1.
4. **Reflect** on failures only: `Reflector.lesson(task, answer, verdict)` (one extra LLM call), with its tokens added to the result.
5. **Write the note**: `domain.reflection(task, raw, answer, verdict)` plus `"\nLesson: …"` if any → `store.add(note, q_emb, verdict.correct, source=verdict.evidence)`.
6. Mark learned, refresh `memory_size`, append `to_log()` to `history`.

### `step(task=None)` = benchmark mode

```python
result = self.attempt(task or self.next_task())
return self.learn(result) if result.verdict.verified else result
```

### `run(n)` and `next_task()`

`next_task()` delegates to the curriculum with `pool`, `done_ids` (derived from `history`, so a restart that reloads history keeps the curriculum consistent), the store, the embedder and the RNG. `run(n)` loops `step` until the pool is exhausted (`StopIteration`).

### `build_agent(cfg, store, embedder, domain, pool, seed)`

Assembles an `Agent` from the config's `agent` section: a `Planner()` if `use_planner`, a `Reflector()` if `use_reflection`, `make_curriculum(cfg)`, and `random.Random(seed)`.

## 6.3 The planner (`planner.py`)

One prompt: "Break the following task into a short numbered plan (at most 5 steps) describing HOW to solve it. Do NOT solve it: give no final answer and write no code." The plan is inserted as `Plan to follow: …`. Costs one extra LLM call per task.

## 6.4 The reflector (`reflection.py`)

After a **failure** it asks for "the most likely cause of the failure and what to do differently next time" in 2–3 sentences, given the task (900 chars), the attempt (900 chars) and what the checker reported. The lesson is an *unverified claim*, so it is stored inside a note whose trust is still set by the objective checker: **a wrong lesson loses trust like any other bad note.**

## 6.5 The curriculum (`curriculum.py`)

The agent chooses its own next task. Three implementations behind `choose(pool, done_ids, store, embedder, rng)`:

- `FixedOrderCurriculum`: the pool in order. Used in experiments so every condition sees the *same* sequence.
- `RandomCurriculum`: uniform over unattempted tasks (the control).
- `WeaknessCurriculum(explore=0.2)`: practise where failures cluster.
  1. Failure notes = items with `correct is False` and **not** poison (planted notes were not experienced).
  2. No failures yet → pick randomly (`last_mode="cold"`).
  3. With probability `explore` pick randomly (`"explore"`), so it cannot fixate.
  4. Otherwise score each unattempted task by its **maximum dot product with any failure-note embedding** and pick the highest (`"weakness"`).
  5. `related_failure(task, store, embedder)` returns the failure note closest to a task — used to *explain* a suggestion ("close to a task I got wrong before: …"). `last_mode` exists so the explanation is only shown when the pick really came from the weakness path.

Raise `StopIteration` when the pool is exhausted.

## Pitfalls

- **Learning twice.** Without the `learned` flag, a retry would store duplicate notes.
- **Mutating memory in `attempt`.** Retrieval increments `hits` (that is intended); but notes must only be *added* in `learn`.
- **Curriculum without ids.** Tasks need unique `id`s (chapter 05).
- **A "test" that proves nothing.** The first planner test passed even if the planner was never called; it was replaced by two tests that count planner calls and assert the plan text reaches the prompt. Test behaviour, not existence.

## Tests (`tests/test_agent.py`, 12)

A scripted fake LLM (chapter 11) answers math problems with the gold or a wrong answer and records every prompt. The tests cover: a step checks the answer, stores a note and records cost; a wrong answer is stored as a low-trust warning (0.35, text contains `INCORRECT`); **trust moves only for the memory the model cited**; poison exposure and reliance are counted; the planner is off by default and, when on, **sees no memories and feeds the solver**; reflection runs **only on failure** and the lesson is stored; a **quota failure mid-step leaves memory untouched**; the random curriculum never repeats and stops when exhausted; the weakness curriculum targets the area of past failures and falls back to random with no failures; and a code-domain end-to-end step.

## Checkpoint

`pilot/.venv/bin/python -m pytest tests/test_agent.py -q` passes. You can construct a store with `build_store("atmc", cfg, embedder, summarize_fn)`, an agent with `build_agent(...)`, and call `agent.step(task)` on a tiny hand-made task list using a fake model.
