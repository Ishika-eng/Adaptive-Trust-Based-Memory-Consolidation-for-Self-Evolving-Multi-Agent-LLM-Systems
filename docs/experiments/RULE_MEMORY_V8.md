# Rule memory v8: trying to close the stable-data gap, with a fair test

Written 2026-10-15, BEFORE any candidate was built or run. Follows RULE_MEMORY.md, where the v7 rule memory failed the clean-stream rule
(42.1% vs plain memory 46.4%, -4.2 points, CI -8.5 to 0.0) while winning the rule-change rule (+27.6 points).

## What the log analysis showed (no new runs)
Reconstructing from the saved v7 logs what each memory would have shown the model for the 268 rule questions: plain memory showed 0.80 same-family worked solutions
per question (56% of questions had at least one); v7 slots showed 1.17 (59%) with the latest two solutions, and 1.28 (63%) if the two solutions nearest to the question
were chosen. So **slot retrieval is not under-supplying relevant solutions**; the 4-point loss comes from something else, most plausibly how the slot note reads
(about three times longer than a plain note, with evidence and solutions from several functions mixed together). This is an approximation: failure types were ignored.

## The three candidate fixes (each follows from that diagnosis; all constants fixed here)
| Candidate | Change | Why |
|---|---|---|
| `rm_short` | Each slot note shows only the 1 evidence line and the 1 solution nearest to the question (storage and bounds unchanged) | A shorter, less mixed note |
| `rm_evidence_slots` | Evidence is kept in slots (bounded); worked examples and solved tasks stay as separate per-question notes | Keeps plain memory's nearest-example retrieval and bounds only the evidence |
| `rm_k2` | v7 slots, but only the top 2 slots are retrieved | Same amount of text in the prompt as plain memory |

## Phase 1: choose ONE, on data separate from the final test
Design seeds **7, 11, 19**, clean 110-question stream, conditions: `static_rm` (control), `rule_mem` (v7) and the three candidates.
The chosen candidate is the one with the highest mean first-attempt pass rate, provided it beats v7 by at least 1 point on these seeds (ties: the smaller memory).
If none does, **stop and report failure; no confirmation run is made**. The choice is then frozen: no constants change after this point.

## Phase 2: confirmation on data used for no decision
(i) Clean stream, new seeds **101, 202, 303**. (ii) The **60 held-out questions** (6 families, written after the experiments; three of them written after v7 ran), seeds 1 to 5.
Conditions: `no_memory`, `static_rm`, v7 and the frozen candidate. A held-out family where `no_memory` already passes at least 60% is reported separately (known practice) and excluded from the pooled test.
Statistics as before: paired bootstrap 95% CI on the pooled paired questions, with each set also reported alone.

## The two bars (both fixed now)
- **Strict bar:** the frozen candidate beats `static_rm` by **>= +3 points** of first-attempt pass rate, **95% CI above 0**, on the pooled confirmation sets.
- **Realistic bar (primary claim if the strict bar fails):** (a) not worse than `static_rm` by more than **2 points** (point estimate) on the pooled confirmation sets;
  (b) memory at least **50% smaller** than `static_rm` (notes or slots at the end of the 110-question stream); (c) on the dev rule-change stream
  (seeds 42/123/456) phase B beats `static_rm` by >= 10 points (CI above 0) and is not more than 3 points below `atmc_v6`.
If (a) to (c) hold the claim is "as accurate as plain memory, much smaller, and far better after a rule change". If the strict bar holds the claim is stronger and is stated as such.
If neither holds, v6 stays the headline and the rule memory is reported as tested.

## Honest expectations
Strict bar: about one in four. Realistic bar: about even odds (v7 is 4.2 points short on accuracy and already clears the size and drift parts).

## Amendment 1 (added 2026-10-15, before any v8 run)
New task orders of the same 110 questions are not new data: every design choice from v1 to v7 was made while looking at how those questions behaved. The 60 held-out
questions are the only set no design decision has seen. Therefore **on the 60 held-out questions alone, the frozen candidate's point estimate against `static_rm` must be
non-negative (>= 0 points) for EITHER bar to count as passed.** If the held-out set disagrees with the 110-question runs, the outcome is reported as "not confirmed", together
with the size of the gap between the two sets (a gap is itself evidence of overfitting to the 110 questions).

## RESULT, Phase 1 (2026-10-15; design seeds 7/11/19; 15 runs; 330 paired questions)
| | first-attempt | second half | after retry | memory at end | per seed |
|---|---|---|---|---|---|
| plain memory (static_rm) | 44.5% | 52.1% | 70.3% | 138 | 42.7 / 45.5 / 45.5 |
| v7 rule memory | 42.7% | 48.5% | 66.1% | 53 | 39.1 / 41.8 / 47.3 |
| rm_short | 39.1% | 43.6% | 69.1% | 51 | 35.5 / 40.9 / 40.9 |
| rm_evidence_slots | 43.0% | 52.1% | 68.5% | 119 | 44.5 / 42.7 / 41.8 |
| rm_k2 | 40.3% | 47.9% | 67.0% | 51 | 40.0 / 38.2 / 42.7 |
Selection rule (fixed above): the best candidate is `rm_evidence_slots` at 43.0%, which beats v7 (42.7%) by **+0.3 points, below the required 1 point -> FAILS -> STOP.**
**No candidate is frozen and Phase 2 (confirmation on new seeds and the 60 held-out questions) is NOT run.** Neither bar was therefore tested on confirmation data.
Observations (not decisions): v7 against plain memory on these fresh seeds is -1.8 points (CI -6.1 to +2.4, p = 0.49) with 62% less memory, a smaller gap than the -4.2 on the original seeds
(so part of that gap was seed noise). Every candidate is within the noise of v7 or worse; rm_short is significantly worse than plain memory (-5.5, p = 0.015).
Any further use of v7 on the confirmation sets would be a post-hoc deviation from this pre-registration and would have to be declared as such (Amendment 2, written before running).
