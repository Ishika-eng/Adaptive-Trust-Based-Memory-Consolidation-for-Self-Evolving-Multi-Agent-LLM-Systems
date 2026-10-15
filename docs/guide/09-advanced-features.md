# 09 · Verification, strategies and memory management

Files: `pilot/autoverify.py` (159), `pilot/strategy.py` (149), plus the evidence fields in `memory.py`/`stores.py` and the routes in `main.py`. Tests: `test_autoverify.py` (11), `test_strategy.py` (12), `test_memory_management.py` (9).

These three features were added *after* the core worked (commits after 11 Oct). Each answers a specific weakness found in the earlier system.

---

## 9.1 Auto-verification — checking answers that come without tests

### The problem

A user's own question usually has no tests, so the answer is *unverified* and, by our own rule, memory cannot learn from it until the user clicks "It worked / It didn't". That is safe but makes the assistant depend on the user for everything. Can it verify some answers itself — **without being fooled by its own mistakes**? A test written by the same model can be wrong.

### The algorithm (`AutoVerifier.run`)

Input: the task and the assistant's first answer. Output: an `AutoCheck` with `status` ∈ `verified | switched | failed | inconclusive | underspecified | unavailable`.

1. Find the function name in the answer with `ast` (`function_name`). None → `unavailable`.
2. **Ask for tests** (one LLM call): "Write up to 8 Python `assert` statements that test a function named `X` … cover normal cases and edge cases … Only assert behaviour the request clearly specifies. If the request is too vague to define one clear correct behaviour, output only the single word UNDERSPECIFIED." For follow-ups the conversation context is included (a follow-up alone looks vague).
   - Reply is `UNDERSPECIFIED` and no assert parses → status `underspecified`.
   - `parse_tests` keeps only single-line, syntactically valid `assert` statements, de-duplicated, at most 8. None → `inconclusive`.
3. **Write independent alternatives** (up to `candidates-1` = 2 more LLM calls): the normal task prompt plus "Name the main function `X`" plus a *variant instruction* ("use a different approach…", "different style…", "simplest implementation…"). Duplicates of an existing candidate are dropped — identical code is not independent evidence.
4. **Run every test against every candidate** in the sandbox, **one sandbox run per candidate** (`run_tests`): the program is the candidate code followed by

   ```python
   import json as __j
   __T = [... the assert strings ...]
   __R = []
   for __t in __T:
       try:    exec(__t, globals()); __R.append(True)
       except BaseException: __R.append(False)
   print("__ATMC_RESULT__" + __j.dumps(__R))
   ```

   The result is parsed from the marker line. A crash, a timeout or a missing marker → every test `False` (so calling `sys.exit(0)` cannot fake success).
5. **Consensus.** A test is **reliable** only if **at least two** candidates pass it. A mistaken test that nothing agrees with cannot condemn a correct answer.
6. **Enough evidence?** Require at least 2 candidates and `n_reliable ≥ max(2, ceil(n_tests/2))`. Otherwise `inconclusive`. (Without this, two *wrong* solutions can agree on one trivial test such as `add(0,0)==0` and the system would "switch" to a wrong answer. A unit test now covers exactly this.)
7. **Decide.** If the first answer passes all reliable tests → `verified`. Else if any alternative passes all → `switched` (that candidate becomes the answer). Else `failed`.

### Worked example

Tests (3): `add(1,2)==3`, `add(0,4)==4`, `add(2,2)==5` ← the last one is wrong. Candidates: `a+b`, `b+a`, `sum([a,b])`. Candidates pass tests 1 and 2 (3 of 3) and fail test 3 (0 of 3). So reliable tests = {1, 2}; test 3 is discarded. The answer passes both → `verified`, despite the bad generated test.

### How the result is used (in the backend)

- `verified` / `switched` → the verdict becomes `Verdict(True, "ok", evidence="generated")`, the turn is `auto_verified`, and memory learns **at half strength** (`autoverify.evidence_weight = 0.5`: both reward and penalty are halved).
- anything else → the answer stays `unverified`, shown with the reason, and waits for the user.
- The user can **dispute** an auto-verified answer (chapter 08).

### What it does *not* do

It measures consistency with the request *as the model understood it*, not the user's unstated intent. A vague request ("clean up a string") was originally auto-"verified" because all three solutions guessed the same meaning; that is why `UNDERSPECIFIED` exists. It costs about 3 extra model calls per question and has only been exercised on a handful of requests, not evaluated statistically.

### Bugs found by testing it with the real model

| Symptom | Cause | Fix |
|---|---|---|
| Vague request "verified" | Consensus ≠ intent | The test-writing prompt may answer `UNDERSPECIFIED` |
| Wrong switch to a wrong answer | Two wrong candidates agreed on one test | Require a body of reliable tests |
| Clear request flagged underspecified | Follow-up lacked its conversation in the test prompt | Include the context in the tests prompt |

---

## 9.2 Strategy evolution — learning *how* to solve, per problem type

### The idea

Memory is one thing that can improve. Another is the **way of working**. Three strategies exist:

| Strategy | What it does |
|---|---|
| `plan_first` (the stable default) | Planner writes a plan (without seeing memory), then solve |
| `direct` | Solve immediately, no plan |
| `revise` | Solve, then ask the model to review its own draft for bugs/edge cases and output a corrected version |

They are implemented in `Agent.attempt(task, strategy)` (chapter 06). `strategy.py` decides **which to use for which kind of problem**, based on evidence.

### Problem types without an extra LLM call (`StrategyBook.classify`)

Six prototypes (strings, lists, numbers, search/sort, recursion/DP, data structures), each a one-sentence description embedded once. A question is assigned to the prototype with the highest cosine similarity if it exceeds `MIN_SIMILARITY = 0.45`, else `general`. On 300 real tasks this gave a sensible spread (it is a heuristic: "roman numeral" lands in *strings*).

### Evidence

`events`: a list of `{ref, category, strategy, mode, success}` where `success` is `None` until the outcome is known. Only resolved events count. `stats(category)` → wins/trials per strategy. `resolve(ref, success)` can be called again later: a user dispute turns a recorded win into a loss.

### The champion rule (`champion`)

Default: `plan_first`. A challenger replaces it for a category only if **all** hold:

- the default has at least `min_trials` (5) resolved attempts, and so does the challenger;
- the **lower Wilson bound** of the challenger's success rate (z = 1.28, about an 80% interval) is at least the default's *observed* rate; the best qualifying lower bound wins.

Why a *lower bound*: a lucky 2/2 has lower bound 0.55, a solid 10/10 has 0.86, 20/20 has 0.92. Small samples are penalised automatically.

**Rollback is automatic** because the champion is recomputed from evidence every time. Example (tested): default 3/6 (rate 0.50); challenger 10/10 → lower bound 0.86 ≥ 0.50 → *promoted*. Later it fails 14 times in a row → 10/24 (lower bound 0.30 < 0.50) → the default is champion again, no manual step.

### Choosing (`choose(category, mode)`)

- `mode="practise"` → the **least-tried** strategy (explore). Practising is the right time to experiment: tasks have real tests, so outcomes are verified and nothing a user sees is at stake.
- `mode="serve"` (a live question) → the champion, except with probability `explore` (0.1) it tries the least-tried one.

### Recording lifecycle in the backend

`_pick_strategy` classifies, chooses, and `BOOK.start(...)`s an event (unresolved) before the attempt. It is `resolve`d when the outcome is known: immediately for tested answers, after user feedback for unverified ones, to a loss on dispute. Events are persisted (`strategy_events`).

### Honest status

Mechanism implemented and unit-tested; exercised in practice on a handful of attempts. **Not shown to improve accuracy** (the base model is at ceiling, samples per type are small).

---

## 9.3 Memory management and evidence

### Why

A memory you cannot see or fix is a liability. We added evidence to each note and controls to the UI.

### Evidence per note

`source` (who verified it), `successes`/`failures` (times relied on in a verified-correct/wrong answer), `last_used_step`, `version` (bumped on merge), `pinned`. They are updated in `update_from_outcome` and by merging; persisted in `memories` columns added by the in-place migration (chapter 08).

### Controls

| Action | Behaviour |
|---|---|
| **Pin** | Exempt from pruning and from being merged; the user vouches for keeping it |
| **Delete** | Removes the note |
| **Export** | JSON `{"format": "atmc-memory/1", "embedder": …, "items": [{text, embedding, correct, is_poison, trust, source, successes, failures, version, pinned}]}` |
| **Import** | Rejected if the embedder differs, >2000 notes, wrong vector size, empty/oversized text. **Imported notes restart at initial trust** and are labelled `imported`, so a file cannot carry in a highly trusted wrong note. Duplicates skipped |

### Retrieval explanations

`AdaptiveTrustStore.retrieve` records, for each chosen note, `rank`, `similarity`, `importance`, `retention` and `score`, and counts how many notes were skipped for low trust. The UI shows "Why this note: ranked #1 · similarity 0.82 · trust 0.55 · worked 2×, misled 0× before · score 0.71" and "2 note(s) were skipped because their trust is below the threshold". Making the memory legible is part of making it trustworthy.

## Tests worth copying

- Auto-verify: wrong test cannot condemn a right answer; two wrong solutions agreeing on one test is *inconclusive*; identical alternatives give *inconclusive*; a vague request is *underspecified*; a follow-up's tests prompt contains the conversation; `sys.exit(0)` cannot fake success.
- Strategy: promotion only with enough evidence and a safe margin; **rollback**; per-type evidence; practise explores, serve exploits; unresolved attempts do not count; a dispute flips a win to a loss; each strategy really runs the steps it promises (counts planner and revision calls).
- Memory: pinned note survives pruning and merging; merged lineage; import resets trust; duplicates skipped; evidence persists through the DB; an old database is upgraded.

## Checkpoint

`pilot/.venv/bin/python -m pytest tests/test_autoverify.py tests/test_strategy.py tests/test_memory_management.py -q` passes, and you can explain each "reliable test" example above with a pencil.
