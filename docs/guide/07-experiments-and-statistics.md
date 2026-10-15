# 07 · Experiments and statistics

Files: `pilot/experiment.py` (241), `pilot/report.py` (147). Tests: `tests/test_experiment.py` (8 functions), `tests/test_report.py` (6).

## What you are building and why

A way to answer, with numbers a skeptic will accept: **does ATMC do what we say?** Concretely:

- run several *conditions* (memory systems / settings) on the **same tasks** with the same seeds;
- save every run to its own file so a quota stop or a crash loses nothing;
- turn the files into tables with **paired statistics**.

Rules this harness enforces (each one prevents a classic mistake):

1. **Same tasks for every condition.** A seed fixes the task sample and order; fixed-order conditions see an identical sequence.
2. **One file per (experiment, domain, condition, seed)**, skipped if it exists → resumable.
3. **Config snapshot inside every result** → reproducible from the file alone.
4. **Pairs, not averages alone.** Conditions are compared task-by-task on the same inputs.

## 7.1 Presets and conditions (`experiment.py`)

A `Condition` = a memory kind plus overrides:

```python
@dataclass
class Condition:
    kind: str              # none | static | reflexion | atmc
    pillars: dict          # enable_trust / enable_forgetting / enable_compression
    agent: dict            # overrides for cfg.agent (use_planner, use_reflection, curriculum)
    retrieval: dict        # overrides for cfg.retrieval (e.g. min_similarity)
```

A `Preset` = named conditions + `poison` (plant wrong notes?) + `heldout` (unseen tasks to score) + `pool_factor` (train-pool multiple when the agent chooses its own tasks).

| Preset | Conditions | Poison | Held-out |
|---|---|---|---|
| `poison` | static, atmc | yes | – |
| `ablation` | full_system, wo_trust, wo_forgetting, wo_compression | yes | – |
| `baselines` | no_memory, static, reflexion, atmc | no | – |
| `baselines_poison` | no_memory, static, reflexion, atmc, static_minsim, atmc_minsim | yes | – |
| `agent` | atmc, +planner, +reflection, +planner+reflection, random choice, weakness choice | no | 20 |

`condition_cfg` sets `curriculum: fixed` unless the condition overrides it, so controlled comparisons are exact.

## 7.2 What one run does (`run_one`)

1. Path: `pilot/results/eval/<exp>__<domain>__<condition>__seed<seed>.json`; if it exists and `--force` is not set → "skipped (already done)".
2. Load `pool_n + heldout` tasks for the seed; `train_pool` is the first `pool_n = n × pool_factor`; the **training sequence** is its first `n`; the rest are **held-out**.
3. Build the store with `build_store(kind, cfg, embedder, llm.summarize, **pillars)`.
4. **Plant poison** (if the preset says so): for every `evaluation.poison_every`-th (3rd) task of the sequence, `domain.poison_specs(...)` → `store.add(text, embedding, correct=False, is_poison=True)`. This is 13–14 planted notes per 40-task run.
5. Build the agent; for fixed curriculum set `agent.pool = sequence`.
6. Snapshot LLM usage; run `n` steps, logging each `StepResult.to_log()` and printing a progress line.
7. **Held-out phase** (if any): `evaluate_frozen(agent, task)` plans (if a planner), retrieves, answers and checks **without learning**, then **restores each note's `hits`** (retrieval would otherwise change importance and leak learning from the test set).
8. Summarise and write the JSON.

### The summary block (every metric the report uses)

`pass_rate` (+ first/second half), `heldout_pass_rate`, `poison_exposure` (planted notes in context, total), **`poison_exposure_after_warmup`** (excluding the first `top_k=3` steps, before any genuine note can exist), `poison_used` (cited), `n_poison_seeded`, `final_memory_size`, `n_compressions`, `n_poison_laundered`, `llm_calls` (logical), `api_calls` (real), `cache_hits`, `tokens_in/out`, `mean_retrieval_ms`, `wall_seconds`.

The file also stores `config`, `condition_spec`, `embedder`, the per-step `steps`, and `heldout_steps`.

### Quota handling

`llm.QuotaExhausted` propagates out; `main()` prints a message and `sys.exit(3)`. **No partial file is written** (a test enforces this). On the next run, completed conditions are skipped and the interrupted run replays its already-answered calls from the response cache, so resuming is cheap. Runs go **seed-major** (all conditions of seed 42, then seed 123 …), so a stop leaves complete seeds.

### Running it

```bash
pilot/.venv/bin/python pilot/experiment.py --list
pilot/.venv/bin/python pilot/experiment.py poison           --domain math --seeds 42,123,456
pilot/.venv/bin/python pilot/experiment.py ablation         --domain code
pilot/.venv/bin/python pilot/experiment.py baselines_poison --domain code --only atmc,static
pilot/.venv/bin/python pilot/experiment.py agent            --domain code --heldout 20
```

Flags: `--seeds`, `--n` (training tasks per run), `--heldout`, `--only` (comma list of conditions), `--embedding fastembed|gemini|tfidf`, `--force`. It prints an **estimate of LLM calls** before starting; compare it to your daily quota.

**Run only one evaluation process at a time** (rate limits are per project).

## 7.3 The report (`report.py`)

```bash
pilot/.venv/bin/python pilot/report.py baselines_poison --domain code --ref no_memory
```

`build_report(exp, domain, ref)` → loads every run for the experiment, and per condition computes mean ± sd **across seeds** of: pass rate, held-out, poison in context (after warm-up), poison used, memory size, tokens per task, retrieval ms. Then, for every other condition, a **paired comparison against the reference condition**, both on training steps (`in_run`) and held-out tasks. `mean_sd` returns `None` (JSON `null`) instead of NaN when there is no data — NaN is not valid JSON and broke the API once.

### The paired comparison, explained from first principles

For each (seed, task) that both conditions attempted, we have two outcomes (pass/fail). Let `diff = (B passes) − (A passes)` ∈ {+1, 0, −1}.

- **Difference**: mean of diffs, in points.
- **Bootstrap 95% CI**: resample the diffs with replacement 10,000 times (fixed RNG seed 0), take the mean each time, report the 2.5th and 97.5th percentiles. It answers "how much could this difference move if we repeated the experiment?"
- **Exact McNemar test**: ignore tasks where both agree; among the **discordant** pairs let `b` = B right/A wrong and `c` = A right/B wrong. Under "no real difference" `b` and `c` split 50/50, so the two-sided exact p is `2 × Σ_{i≤min(b,c)} C(n,i)/2ⁿ` with `n=b+c` (capped at 1). **Worked example from our data**: planner vs plain ATMC on held-out tasks had `b=4, c=0`: `p = 2 × (1/16) = 0.125`. Even with 4–0, only 4 tasks changed, so we cannot call it significant.

Why pairs? Task difficulty varies enormously; comparing two overall percentages ignores that both conditions saw the same hard and easy tasks. Pairing removes that noise.

### Reading a table honestly

- Differences of a few points on 40 tasks are within run-to-run noise (temperature 0 is still not perfectly repeatable).
- A bootstrap interval excluding 0 *and* a McNemar p of 0.125 can coexist; report both.
- "Not significant" is not "no effect" — say what sample size could detect.

## Pitfalls

- **Comparing across eras.** Results from before real embeddings, before the credit-assignment fix, or from a different temperature are not comparable; they are archived in separate folders.
- **Counting exposure before warm-up** inflates ATMC's number with a structural floor; report both totals and after-warm-up.
- **Default arguments bound at import** (the results directory in `build_report`) made tests read the wrong folder; resolve paths at call time.

## Checkpoint

```bash
pilot/.venv/bin/python -m pytest tests/test_experiment.py tests/test_report.py -q
pilot/.venv/bin/python pilot/experiment.py poison --domain math --seeds 42 --n 6   # a tiny real run (about 12 model calls)
pilot/.venv/bin/python pilot/report.py poison --domain math --ref static
```

On a **fresh checkout** the results folder is empty, so the tiny run executes. In *this* repository the seed-42 files already exist, so it prints "skipped (already done)" — that is the resume feature working. **Do not add `--force` here**: it would overwrite the committed evidence. To experiment safely, copy the repo or write to a scratch results folder.
