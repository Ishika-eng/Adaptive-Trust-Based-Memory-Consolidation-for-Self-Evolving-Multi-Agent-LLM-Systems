# Local-model pilot (Qwen2.5-Coder-7B-Instruct, Ollama) — conventions benchmark

Status: **pilot, single seed, 24 rule tasks. Not evidence yet; a diagnosis.** Written 2026-10-13.

## Setup
- Model: `qwen2.5-coder:7b-instruct` (4-bit, Ollama, M3 Pro 18 GB, ~24 tok/s), temperature 0, no API calls.
- Benchmark: `pilot/benchmarks/build_conventions.py` -> `pilot/data/conventions.jsonl`. 3 rule families x 8 tasks
  (`money_paise`, `none_on_invalid`, `ids_padded`) + 4 controls. Each family has one hidden rule the task text never
  states. Validated by running: every reference passes its tests; every "naive" (rule-ignoring) solution fails them.
- Results: `pilot/results/local/` (MBPP/HumanEval baseline), `pilot/results/local_conv/` (conventions).

## Results
| Setting | Pass rate |
|---|---|
| MBPP/HumanEval pool, no memory (40 tasks) | 31/40 = 77.5% (Gemini earlier: ~85-90%) |
| Conventions, no memory (28 tasks) | 6/28 = 21%; controls 4/4; rule tasks 2/24 |
| Conventions, online stream: static / reflexion / ATMC / ATMC+reflection | 21% / 21% / 25% / 21% |
| **Oracle**: exact rule written in the prompt (rule tasks) | 13/24 = 54% |

Offline test (lessons made from the other tasks' failures in the same family, 3 per task; rule tasks, n=24):

| Notes given | Pass |
|---|---|
| none | 2/24 |
| Qwen-written lesson, original prompt | 8/24 |
| Qwen-written lesson, "state the convention" prompt | 7/24 |
| Evidence only (call, expected+type, got+type; no model-written lesson) | 7/24 |
| Gemini-written lesson (Qwen still solves) | 10/24 |

## What this does and does not show
- There is headroom (2/24 -> 13/24 when the rule is known) and retrieval finds same-family notes (68% of top-3
  neighbours vs ~30% by chance).
- Notes of every kind raised the offline score to 7-10/24, but this is one run on 24 tasks: the differences between
  note types (7 vs 8 vs 10) are NOT distinguishable; only "some notes vs none" is suggestive, and it is not tested.
- The ONLINE stream (what the product does) showed no real gain (ATMC 25% vs 21%). In the online stream a task only
  has notes from earlier tasks, so the first task of each family never benefits.
- A single failure is ambiguous evidence (expected 1799 vs got 1799.1 does not say "integer paise"); even Gemini's
  lesson for `money_paise` gave 0/8. The rule is better inferred by combining several failures.
- The 7B model's own reflections are often wrong (blamed floating point; invented "raise IndexError").

## Not yet done
- More tasks per family and more seeds (this pilot cannot support a significance test).
- Combining several failures into one rule note (the compression pillar is the natural place).
- Trust-protected vs unprotected learning when a wrong lesson is written (the actual ATMC claim) on this benchmark.
