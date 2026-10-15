# Building ATMC from scratch — a developer's guide

This guide explains how the ATMC project was built, in the order it was built, in enough detail that a developer who has never seen it can rebuild it. Every claim about the code was checked against the repository at the time of writing; file names, function names and numbers are real.

## What you will have built

A **self-improving coding assistant** whose memory does not blindly trust itself:

- You ask a Python question in a chat page. The assistant plans, remembers, writes code, **runs it in a sandbox**, and checks it.
- It keeps a notebook of what worked and what failed. Every note has a **trust score** that moves with verified outcomes. Notes it cannot trust are never shown to the model again.
- It forgets unused notes, merges similar ones, picks its own practice tasks, verifies answers that come without tests, and tracks which solving strategy works for each kind of problem.
- Around it sits an evaluation harness (poison experiments, ablations, baselines, paired statistics) and a research paper generated from the results.

The language model (Gemini) is used through its API and is **never trained**. Everything that makes this project *ours* is the layer around it.

## How to read this guide

Read the chapters in order the first time. Each chapter has the same shape:

1. **What you are building and why**
2. **Build steps** (what to write, in what order)
3. **Key code explained**
4. **Pitfalls** (mistakes actually made while building it)
5. **Checkpoint** (how to know you got it right before moving on)

| # | Chapter | You build |
|---|---|---|
| 01 | [The idea and the architecture](01-idea-and-architecture.md) | The mental model; no code yet |
| 02 | [Environment and project skeleton](02-environment-and-skeleton.md) | Repo, virtualenv, datasets, API key |
| 03 | [The memory core](03-memory-core.md) | `memory.py`, `stores.py`: trust, importance, forgetting, merging |
| 04 | [Plumbing: config, embeddings, the LLM wrapper](04-config-embeddings-llm.md) | `config.py`, `embeddings.py`, `llm.py`, `factory.py` |
| 05 | [Tasks, checkers and the sandbox](05-domains-and-sandbox.md) | `domains/`, `sandbox.py`, poison construction |
| 06 | [The agent loop](06-agent-loop.md) | `agent.py`, planner, reflector, curriculum |
| 07 | [Experiments and statistics](07-experiments-and-statistics.md) | `experiment.py`, `report.py`, how to read results |
| 08 | [The backend and the chat assistant](08-backend.md) | `app/backend/main.py`, `db.py` |
| 09 | [Verification, strategies and memory management](09-advanced-features.md) | `autoverify.py`, `strategy.py`, pin/import/export |
| 10 | [The frontend](10-frontend.md) | React app, design system, every page |
| 11 | [Testing](11-testing.md) | The 148-case test suite and how to write more |
| 12 | [Results and the paper](12-results-and-paper.md) | Running the experiments, the numbers, generating the paper |
| 13 | [Lessons and mistakes](13-lessons-and-mistakes.md) | The honest log of what went wrong and what it taught |
| 14 | [Rebuild plan and extensions](14-rebuild-plan-and-extensions.md) | A day-by-day plan, exercises, next steps |

## Prerequisites

- **Python 3.11 or newer** (the project was developed on 3.14). Comfortable with dataclasses, `subprocess`, and `pytest`.
- **Node 20 or newer** and basic React (hooks, `useState`/`useEffect`).
- **A Gemini API key** (free tier works; read chapter 02 about its limits).
- **macOS is best** because the sandbox uses `sandbox-exec`. Linux works with weaker isolation (resource limits only).
- Basic ideas: cosine similarity, embeddings, REST APIs, SQLite.

## Repository map

```
.
├── configs/default.yaml        every tunable in one file
├── pilot/                      the engine (all Python, no web code)
│   ├── memory.py               formulas + MemoryItem
│   ├── stores.py               static / none / Reflexion-style / ATMC memories
│   ├── embeddings.py           text -> vectors
│   ├── llm.py                  Gemini wrapper: retries, pacing, cache, parsers
│   ├── config.py, factory.py   config loading; builds stores from config
│   ├── tasks.py                GSM8K loading + math poison
│   ├── domains/                what differs per task type (math, code)
│   ├── sandbox.py              safe execution of generated code
│   ├── agent.py                the loop: attempt -> learn
│   ├── planner.py, reflection.py, curriculum.py
│   ├── autoverify.py           generated tests + consensus
│   ├── strategy.py             champion/challenger strategy selection
│   ├── experiment.py, report.py   evaluation harness + statistics
│   ├── data/                   gsm8k_test.jsonl, HumanEval.jsonl, mbpp.jsonl
│   └── results/                saved experiment runs
├── app/
│   ├── backend/                FastAPI (main.py) + SQLite (db.py)
│   └── frontend/               React + Vite
├── tests/                      pytest suite
├── paper/                      figure + paper generators
└── docs/guide/                 this guide
```

## Conventions used throughout

- Commands run from the repo root and use the project's interpreter: `pilot/.venv/bin/python`.
- "Note" and "memory" mean the same thing: one stored record of an attempt.
- "Poison" always means a **deliberately planted wrong note**, used to test defenses.
- Numbers quoted from experiments come from `pilot/results/eval/*.json` and can be regenerated with `pilot/report.py`.
