# 14 · Rebuild plan and extensions

## 14.1 A day-by-day plan to build it yourself

Roughly twelve focused days for one developer. Each day ends with a **checkpoint you can run**; do not start the next day until it passes. The order is deliberate: pure logic first, then I/O, then the web, then the clever features.

| Day | Build | Chapter | Checkpoint |
|---|---|---|---|
| 1 | Repo, venv, datasets, `.env`, one successful Gemini call | 02 | `import fastembed, google.genai`; a script that prints a model reply |
| 2 | `memory.py` formulas, `MemoryItem`, the four stores | 03 | Unit tests pass; reproduce the worked examples by hand |
| 3 | Config + validation, embeddings, `llm.complete` (pacing, retry, quota error, `USED` parser), factory | 04 | Two paraphrases embed close together; a retry test; the cache test |
| 3 (evening) | **A 100-line pilot script**: GSM8K, 40 tasks, static vs ATMC with planted wrong answers | – | You see static show planted notes far more than ATMC. If you do not, stop and debug before building more |
| 4 | `Domain` interface, math domain, coding domain, **sandbox**, poison builder | 05 | Reference solutions pass; poison fails its tests; the sandbox kills a loop and blocks the network |
| 5 | `Agent.attempt/learn/step`, planner, reflector, curricula | 06 | Agent tests with a fake model pass |
| 6 | `experiment.py` + `report.py`; run a tiny real experiment | 07 | A result JSON appears; `report.py` prints a table with a McNemar p |
| 7 | FastAPI app, SQLite persistence, `_quota_guard` | 08 | API tests pass; `curl` an ask |
| 8 | Frontend shell, API client, chat page (basic) | 10 | You ask a question in the browser and see code + a status |
| 9 | Unverified answers + feedback; conversation context; practice and suggestions | 08 | Feedback test; a follow-up prompt contains the earlier code |
| 10 | Auto-verification | 09 | The three "fool the verifier" tests pass |
| 11 | Strategy evolution + memory management (evidence, pin, import/export, migration) | 09 | Rollback test; migration test on an old DB |
| 12 | Remaining pages, design system, README; run the full experiments in quota-sized batches | 10, 12 | Results page shows your tables; `pytest` is green |

**Tip.** Write the test *before or with* each module. The fake-model and fake-embedder fixtures (chapter 11) let you test almost everything without spending quota.

## 14.2 What I would do differently (so you can start better)

1. **Define the threat model on day 1.** Decide whether planted wrong notes are recorded as *failed* or as *correct*, and say so. We planted failed ones, which is the easier case, and only realised it late. Build the harder attack in from the start.
2. **Build the verifier before the memory** if you can. Everything depends on the checker being right.
3. **Keep one results regime.** Change embeddings, temperature, or the credit rule → archive old results in a new folder and never mix.
4. **Add a larger-headroom benchmark early** (harder tasks or a weaker model). At a ~90% baseline accuracy gains are invisible, which cost us the ability to show benefit.
5. **Persist pending attempts** in the chat backend (note text + embedding), not just in memory.
6. **Decide on enough seeds up front** and budget API quota for them (or use a paid key).
7. **Use relative imports / a real package** once the structure settles, instead of `sys.path` insertion.

## 14.3 Extension exercises (graded by difficulty)

**Easy**
1. Add a `min_trials` setting per strategy category and show it in the UI (hint: `StrategyBook.min_trials`).
2. Add a "Why was this blocked?" tooltip on gated notes in the Memory page (the data is already returned as `gated`).
3. Plot trust over time for one note from `chat` turns.

**Medium**
4. **Realistic poison.** Insert wrong notes recorded as *correct* (a false positive) and re-run `baselines_poison`. Hypothesis: static memory now loses accuracy; ATMC loses trust after one or two cited failures and recovers. This is the most valuable experiment missing from the project.
5. **LLM-as-judge baseline.** Ask the model whether each retrieved note looks relevant and reliable, drop the ones it distrusts, and compare to ATMC — a sceptic's first question.
6. **A second model.** Introduce a provider interface in `llm.py` (`complete(prompt)` is the only function the rest of the code uses). Add a local open-weight model via Ollama; re-run the poison experiment to show the result is not specific to Gemini.
7. **Memory simulator.** Simulate streams of tasks with notes of known quality, a noisy verifier and a model that follows bad notes with probability *p*; run thousands of trials to get tight confidence intervals and a dose-response curve (static degrades with more bad notes, ATMC stays flat). Calibrate *p* from real runs.
8. **Persist pending/learned attempts** so feedback and dispute survive a restart.

**Hard**
9. **Learn the trust weights.** Log, for every retrieved note, its features (similarity, trust, hits, age, source, successes, failures) and whether the answer using it was correct; train a small classifier and compare its ranking (AUC) to the hand-set `0.4/0.4/0.2` formula.
10. **Scale retrieval.** Replace brute-force cosine with an approximate-nearest-neighbour index (HNSW/FAISS), and make saves incremental instead of delete-and-rewrite. Benchmark with 100 → 100,000 synthetic notes.
11. **Multi-user.** Per-user memories, authentication, a job queue for model calls, and a worker pool for the sandbox.
12. **Hard sandbox.** Run generated code in a container with no network and a read-only filesystem.
13. **Multi-agent for real.** Two agents (coder, reviewer) sharing one trust-scored memory, where a note written by one is used by the other; measure whether one agent's mistakes can poison the other and whether trust prevents it.
14. **Evolve the workflow with a guard.** Extend strategy evolution so candidate workflow changes are evaluated on a fixed benchmark and accepted only if they beat the stable version without regressions (with rollback). The design is already in `strategy.py`; the missing piece is automatic *generation* of candidates.

## 14.4 How to know you have rebuilt it

- A stranger can clone your repo, follow your README, and get a running assistant in under 15 minutes.
- Your test suite is green without network access.
- Your results table is regenerated from saved files, not typed.
- You can state, in one paragraph, what your system proves and what it does not.

That last one is the real test of understanding. For this project it is:

> *An agent that writes its own memory should treat that memory as untrusted input that must earn its influence. Scoring each note by verified outcomes, and only for notes the model actually cited, kept planted wrong notes out of the model's context roughly tenfold, with trust scoring as the decisive component. It did not measurably raise accuracy, because the base model already solves about 90% of these tasks; and the planted notes were recorded as failures, so the stronger "wrong note recorded as correct" attack remains untested.*
