# Experiment 1 — Does crediting only the notes the model cited separate good notes from bad ones?

**Status:** protocol written 2026-10-13 **before** any real-model run. **Amendment 1 (same day) — see §9:** version 1 of the run (6 of 9 runs) turned out to have an ill-posed trust metric, so the primary experiment is now the *isolated* version (`credit_isolated`). Final results (all 15 runs, five seeds) are in §11.
**Why this experiment:** the novelty analysis (`docs/NOVELTY_ANALYSIS.md`) found that outcome-driven per-memory scores already exist (MemRL, CoEvo-Mem) and credit **every retrieved/injected memory**; MemRL lists the resulting "credit-assignment ambiguity" as a limitation. Our rule credits **only notes the model reported using** (`USED:` line). That choice is our most specific claim, and it had never been tested against the alternative.

## 1. Question and hypotheses

Under identical conditions, which credit rule makes a note's trust a better signal of whether the note is wrong?

| Rule | What gets updated after an outcome |
|---|---|
| `cited` (the project's rule) | only notes the model reported using |
| `all` (the MemRL-style rule) | every retrieved note |
| `asymmetric` (**exploratory**, not pre-registered) | on success only cited notes are rewarded; on failure every retrieved note is blamed |

- **H1 (separation).** `cited` ends with higher **trust AUC** than `all` — a genuine note is more often trusted more than a planted wrong note.
- **H2 (no harm).** `cited` is not worse than `all` on downstream outcomes: pass rate, and planted-note exposure after the first pass.
- **H3 (limits).** `cited`'s advantage shrinks when the model's citations are unreliable.

**What would count against our claim.** If `cited` does not beat `all` on trust AUC (paired difference < 0.05, or not in the same direction in at least 4 of 5 seeds), **or** `cited` is worse on pass rate by more than 2 points or on exposure by more than 10%, then the claim that citation-conditioned credit is better is **not supported**, and we report it that way.

## 2. Design

- **Domain / model:** code (HumanEval + MBPP), `gemini-3.5-flash-lite`, temperature 0, `BAAI/bge-small-en-v1.5` embeddings; ATMC with forgetting and merging **on** as in the main system (only the credit rule differs).
- **Tasks:** 40 per seed, **three passes over the same tasks** (120 steps per run), so planted notes are retrieved repeatedly and credit assignment has something to act on.
- **Poison:** a wrong solution (single-site mutation verified to fail its tests) planted for every 3rd task and **recorded as a success** (`Outcome: CORRECT`, trust starts at 0.55). This is the realistic false-positive case that the earlier experiments did not test.
- **Seeds:** 42, 123, 456, 789, 1011 (analysis unit = seed). First batch: 42, 123, 456.
- **Conditions:** `credit_cited`, `credit_all` (primary); `credit_asymmetric` (exploratory).

## 3. Metrics (fixed in advance)

**Primary:** `trust_auc` at the end of each run = P(genuine note trust > planted note trust), ties count ½. 0.5 means no separation.
**Secondary:** pass rate; planted notes retrieved after pass 1; planted notes cited; final mean trust of planted and of genuine notes; fraction of each blocked (< 0.30); per-step mean trust of planted notes (in the result file).

## 4. Analysis plan

Per-seed values are reported; the comparison is paired by seed (`cited − all`) with the mean difference and the number of seeds in which `cited` is higher. With five seeds no p-value is claimed; the pass-rate comparison also uses the paired per-task bootstrap and exact McNemar test from `report.py`. The exploratory arm is reported separately and labelled exploratory.

## 5. What I had seen before finalising this protocol (disclosure)

1. **A simulation** (`pilot/credit_sim.py`) using the *real* `AdaptiveTrustStore` with a simulated model. Default parameters (200–500 paired trials): trust AUC `cited` 0.90, `all` 0.77, `asymmetric` 0.975. `cited` was **better on separation but not on harm**: `all` had slightly fewer harmful steps and higher accuracy (20.1 vs 21.5 harmful steps) because blaming everything retrieved blocks culprits whose citations are missing. `asymmetric` had the highest trust AUC and the fewest harmful steps in all eight swept regimes (at the default settings: about 58 fewer later exposures and +6 points accuracy vs `all`), at the cost of slightly more genuine notes blocked (3.2% vs 1.4–2.0%). **These are simulation results driven by assumed model behaviour, not evidence about a real model.** The asymmetric rule was designed *after* seeing them, which is why it is exploratory.
2. **A tiny real-model smoke run** (4 tasks × 3 passes, one seed, code domain, outside the results folder): all 12 steps passed in both arms. Planted-note trust ended at **1.00** under `all` and **0.72** under `cited`, and both ended above genuine-note trust (0.60 / 0.56), i.e. trust AUC 0 for both. The model **cited the planted notes 3 times while still answering correctly**.

**Consequence for expectations.** The smoke run suggests that with a strong model that solves tasks despite a wrong note, planted notes can gain trust under *either* rule (a cited-and-correct wrong note is rewarded), so trust AUC could be low for both and the difference between rules may be small. That is a real possibility and would be reported as such: outcome-driven trust measures association with success, not truth.

## 6. How to run

```bash
# one process at a time; resumable; prints an estimate of model calls first
pilot/.venv/bin/python pilot/experiment.py credit --domain code --seeds 42,123,456
pilot/.venv/bin/python pilot/experiment.py credit --domain code --seeds 789,1011        # second batch
pilot/.venv/bin/python pilot/report.py credit --domain code --ref credit_all
pilot/.venv/bin/python pilot/credit_sim.py --trials 500          # mechanism test (no API)
pilot/.venv/bin/python pilot/credit_sim.py --sweep --trials 100  # robustness grid (no API)
```

Estimated cost: about 1,200 model calls for three seeds and three conditions (about 800 without the exploratory arm: `--only credit_cited,credit_all`). Scale to your daily quota; finished runs are skipped on resume.

## 7. Interpretation guide (decided now)

| Outcome | What we may say |
|---|---|
| `cited` > `all` on AUC and not worse on harm | Citation-conditioned credit is more precise attribution; supports the novelty claim |
| `cited` > `all` on AUC but worse on harm/exposure | More precise but slower to block bad notes; report the trade-off; consider `asymmetric` |
| No AUC difference, or `all` better | The citation-credit claim is **not supported**; drop it from the novelty statement |
| Both rules give AUC ≈ 0 | Outcome-driven trust cannot separate wrong notes the model silently corrects; report as a limitation, and consider stronger evidence (tests that exercise the note) |
| `asymmetric` best | A candidate design improvement, **to be confirmed on a fresh set of seeds** before it is adopted or claimed |

## 8. Implementation (files)

`pilot/agent.py` (credit rules), `pilot/config.py` and `configs/default.yaml` (`agent.credit`), `pilot/experiment.py` (`credit` preset, `poison_label`, `repeats`, trust metrics, `--out`), `pilot/report.py` (trust columns, `--dir`), `pilot/credit_sim.py` (simulator), `tests/test_credit_experiment.py` (14 tests). Results go to `pilot/results/eval/` (real) and `pilot/results/sim/` (simulation).

## 9. Amendment 1 (2026-10-13): version 1 was confounded; the primary experiment is now the isolated version

**What happened.** The first real batch (`experiment.py credit`, forgetting and merging **on**, as §2 specified) was stopped by the daily quota after 6 of 9 runs: seeds 42 and 123 finished for all three conditions, seed 456 not started. Before interpreting anything I traced the per-step logs of one run and found a flaw in **my design**, not a result:

- Merging replaced many notes with summaries. At the end only **3 of the 13 planted notes still existed as individual notes**; the others had been merged. "Planted trust" and **trust AUC were computed on those 3 survivors**, and a merged summary inherits the planted flag if any parent was planted and a trust equal to the mean of its parents.
- The "planted notes shown" count (about 360 = 3 per step for 120 steps in two arms) therefore mostly counts **merged summaries that contain some planted content**, not individual wrong notes.
- The simulation used for the mechanism test had forgetting and merging **off**, so the real run and the simulation were not testing the same thing.

So the version 1 trust metrics cannot be used to compare the credit rules. This was caught from the data before any claim was made.

**What changes.** The primary experiment is the new preset **`credit_isolated`**: identical to the original design (code domain, 40 tasks × 3 passes, wrong notes recorded as successes, seeds 42/123/456/789/1011) **except forgetting and merging are off** in all arms, so every planted note keeps its identity and trust to the end and only the credit rule differs. Conditions: `iso_cited`, `iso_all` (primary), `iso_asymmetric` (exploratory). Version 1 stays in the repository as a secondary "full system" run, reported as such, with only pass rate and the exposure counts interpreted (and the exposure count flagged as including merged summaries).

**What does not change.** The hypotheses H1–H3, the primary metric (trust AUC), the decision rules in §1 and §7, the analysis plan, and the rule that the asymmetric arm is exploratory.

**New measurement.** Each step now records how the model's `USED:` line was read (`listed`, `none`, `missing`, `unparseable`; `cite_status`), and each run summarises it (`cite_listed_frac`, `cite_missing_frac`). A missing or unparseable line makes the parser credit **every** retrieved note, which would silently turn the `cited` rule into the `all` rule. This directly measures how reliable self-citation is (H3) and is the first thing to check if `cited` and `all` behave alike.

**Version 1, what the six finished runs showed (disclosed, confounded, two seeds, not used for any decision):**

| Arm | Trust AUC (seeds 42, 123) | Planted notes remaining | Pass rate | Notes |
|---|---|---|---|---|
| `credit_all` | 0.00, 0.00 | 3 of 13, 3 of 14 | 0.95, 0.93 | planted notes' mean trust 1.00 |
| `credit_cited` | 0.17, 0.00 | 3 of 13, 3 of 14 | 0.94, 0.94 | planted notes' mean trust 0.70, 0.75 |
| `credit_asymmetric` | 1.00, undefined (no planted notes left) | 1 of 13, 0 of 14 | 0.92, 0.93 | planted notes shown 81 and 21, versus 360 in the other arms |

Pass rates were similar in all arms. These numbers are shown only for transparency; they are **not** evidence for or against H1–H3.

**How to run the primary experiment.**

```bash
pilot/.venv/bin/python pilot/experiment.py credit_isolated --domain code --seeds 42,123,456
pilot/.venv/bin/python pilot/report.py credit_isolated --domain code --ref iso_all
```

About 1,080 model calls for three seeds and three conditions (720 without the exploratory arm: `--only iso_cited,iso_all`); there are no merging calls, and the cache cannot help because memory contents differ.

## 10. Results — batch 1 of `credit_isolated` (seeds 42, 123, 456; 2026-10-13)

Both primary arms are complete for all three seeds; the exploratory arm has seeds 42 and 123 only (seed 456 hit the daily quota). The formal decision rule in §1 needs five seeds, so this is the first batch, read against the pre-registered rule.

| Arm | Trust AUC per seed (42 / 123 / 456) | Mean | Planted trust | Genuine trust | Planted shown after pass 1 | Planted blocked | Pass rate |
|---|---|---|---|---|---|---|---|
| `iso_cited` | 0.42 / 0.39 / 0.47 | **0.429** | 0.57 | 0.55 | 240 / 240 / 200 | 0–7% | 0.93 / 0.93 / 0.80 |
| `iso_all` | 0.38 / 0.39 / 0.39 | **0.390** | 0.63 | 0.55 | 240 / 240 / 240 | 0% | 0.95 / 0.93 / 0.80 |
| `iso_asymmetric` (exploratory) | 0.74 / 0.65 / – | 0.694 | 0.41 | 0.53 | 30 / 14 / – | 46% / 29% | 0.90 / 0.93 / – |

Paired `cited − all`: AUC **+0.04, +0.00, +0.08** (mean +0.039). Pass rate −0.8 points (95% CI [−3.3, +1.7], McNemar p = 1.0). How the model's `USED:` line was read: it **named notes on 7% of steps and wrote "USED: none" on 93%**; it was never missing or unparseable.

### What this says, against the pre-registered rules

1. **H1 (cited separates better than all): not supported.** The mean AUC difference (+0.039) is below the 0.05 threshold set in advance, and it was zero in one of three seeds. Per §7, this means the claim that citation-conditioned credit is better is **not supported by this experiment** and should not be presented as a result.
2. **H2 (no harm): holds, trivially.** Pass rates did not differ between the two primary arms.
3. **§7 "both rules near chance": applies, and it is worse than chance.** Both AUCs are **below 0.5** (0.43 and 0.39): planted wrong notes end *more* trusted than genuine notes. Outcome-driven trust cannot separate wrong notes that the model silently overrides.
4. **H3 (citations unreliable): confirmed in a different form.** The model almost never lists notes, so the `cited` rule almost never updates anything; it behaves like "no learning" (planted trust 0.57 vs the 0.55 starting value).
5. **Exploratory arm:** asymmetric credit is the only rule that moved planted notes: AUC about 0.69, planted notes shown after pass 1 reduced by roughly 90% (30 and 14 vs 240), 29–46% of planted notes blocked, at a cost of about 4% of genuine notes blocked and a pass rate that was equal in one seed and, in the other (seed 42), 5 points below credit-all (3 below cited-only). **Two seeds, exploratory, designed after seeing simulation results: a hypothesis to confirm, not a finding.**

### Likely mechanism (a hypothesis, supported by simulation but not directly measured in the real runs)

A planted wrong note recorded as a success starts at trust 0.55, the same as a genuine note. Planted notes are near-duplicates of tasks, so they are retrieved on almost every step (3 per step in the primary arms). Each retrieval raises `hits`, which raises *importance* and therefore their rank, so they keep being retrieved: a **retrieval-frequency lock-in**. Under `all`, being retrieved during a successful answer also rewards them (planted trust 0.63). Only a failure that is blamed on them can lower their trust, and a model that overrides them never provides one. The simulator, with the real store and a model that cites rarely, reproduces the pattern (AUC 0.37 cited, 0.23 all, 0.89 asymmetric). The per-note hit counts were not stored in the real result files, so this part is not verified directly.

### Consequences

- **Claim C ("credit only cited notes") cannot be listed as a contribution** unless a later batch changes this result. `docs/NOVELTY_ANALYSIS.md` has been updated.
- **A weakness is now documented:** false-success notes lock in under every rule except asymmetric credit, and even then only partly.
- **Open follow-ups:** (a) confirm asymmetric credit on fresh seeds before adopting it; (b) test replacing retrieval-count importance with verified-success importance; (c) finish batch 2 (seeds 789 and 1011) for the formal five-seed rule.

## 11. Final results — `credit_isolated`, all five seeds (42, 123, 456, 789, 1011; 2026-10-13)

All 15 runs finished. Seeds 789 and 1011 were not part of batch 1, so they are the fresh seeds §7 required for confirming the exploratory arm.

| Arm (mean over 5 seeds) | Trust AUC | Planted notes' trust | Genuine notes' trust | Planted notes shown after pass 1 | Planted blocked | Genuine blocked | Pass rate |
|---|---|---|---|---|---|---|---|
| `iso_cited` | **0.415** | 0.57 | 0.55 | 232 | 1% | 0% | 0.885 |
| `iso_all` | **0.391** | 0.64 | 0.55 | 240 | 0% | 0% | 0.885 |
| `iso_asymmetric` (exploratory) | **0.709** | 0.39 | 0.52 | **18** | 44% | 7% | 0.873 |

Per-seed trust AUC (42 / 123 / 456 / 789 / 1011): cited 0.42 / 0.39 / 0.47 / 0.39 / 0.39; all 0.38 / 0.39 / 0.39 / 0.39 / 0.39; asymmetric 0.74 / 0.65 / 0.69 / 0.67 / 0.81.

### Pre-registered tests (§1, §7)

| Test | Result | Verdict |
|---|---|---|
| **H1** `cited` beats `all` on trust AUC by at least 0.05, higher in at least 4 of 5 seeds | difference **+0.024**; `cited` higher in **2 of 5** seeds, equal in 3 | **Not supported** |
| **H2** `cited` is not worse than `all` on harm | pass rate 0.885 vs 0.885; planted exposure 232 vs 240 | Holds (no difference) |
| **H3** citations unreliable | the model named notes on only **9–10%** of steps and wrote "USED: none" on the rest; the line was never missing | Confirmed: the `cited` rule almost never updates |
| Both rules near chance | AUC 0.415 and 0.391, both **below 0.5** | Applies: reported as a limitation |

**Conclusion on the original claim:** crediting only the cited notes did **not** separate wrong notes from good ones better than crediting every retrieved note. The claim is withdrawn.

### The exploratory arm, confirmed on fresh seeds

On the five seeds, `asymmetric` minus `all`: trust AUC **+0.354, +0.257, +0.292, +0.273, +0.413** (higher in 5 of 5; fresh seeds +0.273 and +0.413); planted notes shown after pass 1 **down by 210–232 in every seed** (about 92%). Costs: pass rate **−1.2 points on average** (−5.0, 0, +0.8, 0, −1.7; not significant in the paired per-task test, McNemar p = 1.0), and **7% of genuine notes blocked on average** (4% to 11% per seed; 0% in the other arms).

**Status:** the direction and size replicated on fresh seeds, so asymmetric credit is a **confirmed candidate improvement in this setting**. It is not yet a general result:

- It was found after seeing simulation output, tested on one model, one domain, and a poison we constructed.
- It probably works partly because the planted notes are retrieved on almost every step (the lock-in), so they are present at nearly every failure and get blamed far more often than genuine notes. A wrong note retrieved only rarely would not be exposed this way, and any frequently useful genuine note risks the same blame (7% of genuine notes were blocked). This is a mechanism hypothesis and was not isolated.
- Planted-note trust separates only partially (AUC 0.71; 44% of planted notes blocked), and the pass rate dipped slightly.

### What should and should not be claimed

- **Do not claim** that citation-conditioned credit improves trust assignment.
- **May claim**, with the caveats above, that in a setting where wrong notes are recorded as successes and the model rarely cites notes, both "credit cited" and "credit all" rank wrong notes above genuine ones, while an asymmetric rule (reward cited notes, blame every retrieved note on failure) reduced later exposure by about 92% and raised trust AUC from 0.39 to 0.71 at a small cost in genuine notes blocked.
- **Next tests:** retrieve-rarely wrong notes (is the effect just the lock-in?); a second model; replacing retrieval-count importance with verified-success importance; whether to make `asymmetric` the default.
