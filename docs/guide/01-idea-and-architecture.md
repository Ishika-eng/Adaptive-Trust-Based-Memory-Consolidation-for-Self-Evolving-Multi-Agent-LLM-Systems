# 01 · The idea and the architecture

## The problem in plain words

Imagine a student who keeps a notebook of every problem they solved and every mistake they made. That is useful, but only if the notebook is *accurate*. If it contains a wrong solution written down as if it were right, the student will trust it next time and repeat the mistake, now with more confidence.

AI agents that "learn from experience" have exactly this problem. After each task they write a note about what happened and read relevant notes back on later tasks. The notes are written by a language model, which makes mistakes, hallucinates, and can be tricked. Nothing checks whether a note deserves to be trusted.

**ATMC (Adaptive Trust-based Memory Consolidation)** is our answer: treat every note as something that must *earn* trust.

## The thesis, in four rules

1. **A note's trust starts from a verified outcome.** Tests passed → higher starting trust. Tests failed → lower.
2. **Trust moves with evidence.** When the model *says it relied on a note* and the answer is verified correct, the note's trust rises a little. When it relied on a note and the answer is wrong, trust falls a lot (bad news travels faster than good: penalty 0.15, reward 0.05).
3. **Low-trust notes are never retrieved.** Below a threshold (0.30) a note is invisible to the model.
4. **Memory stays small.** Unused notes fade, and similar old notes are merged into one summary.

Everything else (the assistant, the sandbox, the experiments) exists to build this memory into a working system and to measure whether it helps.

## What "self-evolving" means here — and what it does not

Be precise; this is where projects overclaim.

| Evolves from experience | Does **not** change |
|---|---|
| The memory (trust, which notes exist, merges) | The language model's weights |
| Which practice task the assistant picks next | The prompts and tools (they are fixed by us) |
| Which solving strategy it prefers per problem type | The code of the agent |

So it is **memory-level and strategy-selection-level** evolution. It is training-free. We did **not** show that it makes the model more accurate (the model already solves about 90% of the benchmark tasks alone); we showed it keeps planted wrong notes out of the model's context.

## The system in one picture

```
 ┌─────────── Browser (React) ───────────┐
 │ Assistant · Memory · Evolution · ...  │
 └──────────────────┬────────────────────┘
                    │ HTTP/JSON
 ┌──────────────────▼────────────────────┐
 │ FastAPI backend  (app/backend/main.py)│
 │   one lock, SQLite persistence        │
 └───────┬───────────────────────┬───────┘
         │                       │
 ┌───────▼────────┐      ┌───────▼──────────────────────────────┐
 │ Agent loop     │      │ ATMC memory (pilot/stores.py)        │
 │ plan→retrieve  │◄────►│ trust · importance · forgetting ·    │
 │ →solve→check   │      │ merging · evidence · pin             │
 │ →learn         │      └──────────────────────────────────────┘
 └──┬─────────┬───┘
    │         │
 ┌──▼───┐  ┌──▼─────────────┐   ┌─────────────────────┐
 │ LLM  │  │ Sandbox        │   │ Local embedder      │
 │Gemini│  │ runs the code  │   │ (BGE-small, ONNX)   │
 └──────┘  └────────────────┘   └─────────────────────┘
```

Only the LLM is an external service. Embeddings, the sandbox, memory and persistence all run on your machine.

## One question, end to end

This is the path of a single chat message. Keep it in mind; every later chapter builds one box of it.

1. **Browser** `POST /api/chat/ask {prompt, tests?, autocheck}` (`Assistant.jsx` → `api.js`).
2. **Backend** (`chat_ask` in `main.py`) takes the global lock, builds the task. If this message continues an earlier conversation, the last three exchanges become *context* (`_conversation_context`).
3. **Strategy** is chosen for this problem type (`StrategyBook.classify` + `choose`): *plan first*, *direct*, or *write then review*.
4. **Agent.attempt** (`agent.py`):
   1. (if plan-first) the **planner** writes a plan *without seeing memory*;
   2. the question is **embedded**;
   3. the store **retrieves** the top-3 notes by composite score (trust-gated);
   4. the **prompt** is built from memories + plan + question; the **LLM** answers; the reply ends with a `USED: 1,3` line saying which notes it relied on;
   5. the code is extracted and **checked** in the sandbox (real tests if given).
5. If the user gave no tests, the result is *unverified*. If auto-check is on, **AutoVerifier** asks the model for edge-case tests and two alternative solutions, runs them all in the sandbox, and applies a consensus rule (chapter 09).
6. If the outcome is now *verified* (tests or auto-check), **Agent.learn** runs: trust is updated for the notes that were used, a **reflection** (lesson) is written for failures, and a **new note** is stored. If it stays unverified, nothing is learned until the user clicks "It worked / It didn't".
7. The backend **saves** memory, history and strategy evidence to SQLite and returns a JSON "turn".
8. The browser renders the answer: explanation, highlighted code, status pill, feedback buttons, "how I solved this".

## Module dependency map (who may import whom)

Keep imports flowing in one direction; it keeps tests simple.

```
memory.py                      (pure math, imports nothing of ours)
   ▲
stores.py                      (uses memory.py, numpy, sklearn)
   ▲
factory.py ── config.py        (builds stores from config)
   ▲
domains/*  ── sandbox.py ── llm.py ── embeddings.py
   ▲
agent.py  ── planner.py, reflection.py, curriculum.py
   ▲
autoverify.py, strategy.py     (use llm/sandbox/embeddings, not the agent)
   ▲
experiment.py, report.py       (batch tools)         app/backend/main.py (web)
```

Rules that paid off: the **math has no I/O** (so it is trivially unit-tested), the **agent never knows which UI or experiment is driving it**, and **one `Agent.step` serves experiments and the app alike** so they cannot drift apart.

## Vocabulary

| Term | Meaning |
|---|---|
| Note / memory | One stored record: task, attempted solution, outcome, optional lesson, trust, evidence |
| Trust `T` | Score in [0,1] for how much to rely on a note |
| Gate `T_min` | Below this trust (0.30) a note is never retrieved |
| Verdict | Result of checking an attempt: correct?, stage, detail, evidence source, verified? |
| Evidence | Where a verdict came from: `tests`, `user`, `generated` (auto-check, weaker) |
| Poison | A deliberately planted wrong note (for experiments) |
| Exposure | Planted notes that appeared in the model's context |
| Reliance | Planted notes the model *cited* in `USED:` |
| Warm-up | The first 3 tasks, when planted notes are the only notes in memory |
| Condition | One configuration in an experiment (e.g. `static`, `atmc`, `wo_trust`) |
| Seed | Controls which tasks are sampled; same seed = same tasks for every condition |

## Checkpoint

You should be able to explain, without code: (a) why an unverified memory is a risk, (b) the four rules, (c) what does and does not evolve, and (d) the eight steps of one question. If not, re-read before moving on; the rest is implementation of exactly this.
