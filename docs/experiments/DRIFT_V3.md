# Stronger rule-change evidence: recurring stream, fair rivals, memory-hygiene audit

Written 2026-10-16, BEFORE the streams are built and before any run of them. Nothing below may be changed after the first run
except by a dated amendment at the end of this file.

## Why
The rule-memory result so far (docs/experiments/RULE_MEMORY.md) rests on two streams (29 and 20 questions, 3 seeds), one rule change per
stream, and one comparison that matters (plain memory, which is a weak rival after a change). A reviewer will ask for the fair rivals:
- a memory that simply deletes any note the checker proves wrong ("keyed revocation", the idea in TEPA-style work);
- a memory where the newest verified solution for a function replaces the older ones ("last write wins");
- what happens when the rule changes BACK (does demoting notes retire still-good knowledge?);
- how many notes the memory hides that were still correct, and how many it keeps showing that are wrong.

## Streams (all built by `pilot/benchmarks/build_drift.py`, every task validated by running a correct and a wrong solution, as before)
**Recurring stream (`conventions_drift3`)**: the 49 validated questions of the two earlier drift streams (5 families), asked in three phases:
A (old rule) -> B (new rule) -> A2 (the OLD rule again, same tests as A). 147 tasks per seed. Questions inside a phase are shuffled per seed.
**Transfer stream (`conventions_drift4`)**: the same families, but the questions are split in two halves by a fixed rule (even / odd index):
phase A asks half 1 under the old rule, phase B asks half 2 (functions never seen before) under the NEW rule. No stored solution can be
re-run against a phase-B question, so the detector cannot fire on its own evidence. About 49 tasks per seed. This is where a keyed method
has nothing to key on, and it is reported as an honest test of where rule-level memory is needed, with no pass rule.

## Conditions
| name | what it is |
|---|---|
| `no_memory` | no memory |
| `static_rm` | plain memory with worked examples and memory-aware retry (the control used in every earlier test) |
| `static_rv` | plain memory + the detector: drops the one note a re-run proves stale (keyed revocation) |
| `static_lww` | plain memory, and when a new verified solution for function F is stored, every older note that mentions F is deleted (last write wins) |
| `atmc_v6` | ATMC v4 + detector + neighbourhood demotion (best non-slot version) |
| `rule_mem` | rule memory v7, unchanged (no tuning for this stream) |
Seeds 101, 202, 303 (fresh: never used for any decision). Model: Qwen2.5-Coder-7B-Instruct via Ollama, temperature 0.

## Memory-hygiene audit (new measurement)
At the end of each phase, for every stored verified solution whose function is asked in the stream, re-run it against the CURRENT checker:
- **wrongly shown** = fails now, but the memory would still show it (trust >= 0.30, not removed);
- **wrongly hidden** = passes now, but the memory would not show it (removed, stale, or trust < 0.30).
Reported as fractions of audited solutions. This is the "premature retirement" number: the A2 phase tests it directly, because the notes
demoted in B are right again in A2.

## Predictions and pass rules (first-attempt pass, paired per question, 95% bootstrap CI, exact McNemar; recurring stream only)
- R1 (replication): in phase B, `rule_mem` beats `static_rm` by >= 10 points with the CI above 0. (Expected: likely.)
- R2 (fair rivals): in phases B and A2 pooled, `rule_mem` beats BOTH `static_rv` and `static_lww` by >= 3 points with the CI above 0.
  Honest expectation: unlikely against `static_lww`, because every question recurs and a function-keyed rule is a strong rival for recurring
  questions. Readings, fixed now: beats both = "rule memory is better than the fair rivals on rule changes"; CI includes 0 for either =
  "ties that rival, and the benefit over plain memory is mostly from revocation, which the rival also has"; below by >= 3 = "loses to that rival".
- R3 (no harm before the change): phase A pass of `rule_mem` is within 5 points of `static_rm`.
- R4 (hygiene): at the end of phase B, `rule_mem` wrongly hides <= 15% of audited solutions; and in A2 it recovers (its A2 pass rate is
  not below its A-phase pass rate by more than 10 points).
- Cost table (not a pass rule): notes stored, characters shown per question, model calls and wall-clock per question, per condition.
- Transfer stream: report phase-B pass for every condition against `no_memory`. No pass rule; expectation: every memory is at or below
  `no_memory` there, because nothing can prove the older notes stale.

## What would count as a conclusion
- If R1, R3 and R4 hold and R2 holds against both rivals: the rule-change claim is "better than plain memory and than function-keyed
  revocation and last-write-wins".
- If R1, R3, R4 hold and R2 fails: the claim is "rule memory recovers from rule changes as well as simple keyed revocation, with much less
  memory"; the paper must say that the revocation, not the slots, is what protects the memory.
- If R1 or R4 fails: we withdraw the drift claim for rule memory and keep only "plain memory collapses after a rule change" and the detector result.
No tuning of any condition after seeing these results. Results go in this file under a dated RESULT heading, whichever way they go.

## RESULT (2026-10-16; seeds 101/202/303; recurring stream 147 tasks per run; analysis `pilot/report_drift3.py`, output `pilot/results/drift_v3_report.md`)
First-attempt pass (%), A (old rule) / B (new rule) / A2 (old rule again):
no memory 10.2 / 44.9 / 10.9 | plain memory 39.5 / 10.9 / 34.0 | delete-contradicted 38.1 / 10.9 / 20.4 | last-write-wins 34.0 / 10.9 / 28.6 |
ATMC v6 33.3 / 26.5 / 23.1 | rule memory 33.3 / 34.0 / 31.3. Memory at the end (notes): 0 / 186 / 142 / 76 / 100 / 41.
- **R1 MET.** Phase B, rule memory vs plain memory +23.1 points (CI +15.0 to +31.3, p < 0.001; needed >= +10).
- **R2 MET (pooled B+A2).** vs delete-contradicted +17.0 (CI +11.6 to +22.4); vs last-write-wins +12.9 (CI +6.8 to +18.7). The win is carried by phase B
  (+23.1 against both). In A2 alone rule memory beats delete-contradicted (+10.9, CI +4.1 to +17.7) but only ties last-write-wins (+2.7, CI -6.1 to +11.6)
  and plain memory (-2.7, CI -11.6 to +6.1). My written expectation (a tie with last-write-wins) was wrong for phase B: each question appears once in B, so
  function-keyed revocation has nothing to act on there, while the rule slot generalises from the first failures to later questions of the same family.
- **R3 NOT MET.** Phase A, rule memory vs plain memory -6.1 points (CI -12.2 to 0.0, p = 0.09); the limit was -5. Missed by 1.1 points; reported as failed.
- **R4 MET.** At the end of B rule memory wrongly hid 2 of 67 audited solutions (3.0%; limit 15%), and A2 (31.3) is within 2 points of A (33.3).
- Hygiene (audited / wrongly shown / wrongly hidden, summed over seeds, end of A2): plain 242/75/0, delete-contradicted 88/7/0, last-write-wins 124/38/0,
  **ATMC v6 241/9/78**, rule memory 61/0/1. ATMC v6's neighbourhood demotion retires 78 solutions that are valid again when the rule reverts (premature retirement, measured).
- Phase B is still below no memory for every method (rule memory -10.9 points, CI -19.0 to -2.0, p = 0.02): recovery is partial.
- The three identical phase-B rates (10.9) of plain / delete-contradicted / last-write-wins were a coincidence of totals; per-question agreement is 139/147 and 137/147.
- Cost per step: rule memory sends about 1250 prompt tokens against about 730 for plain memory (+72%) because it shows evidence and solutions together; its STORED memory is
  41 notes against 186, so the saving is storage, not prompt size. Model calls per step are about the same (1.67 against 1.71).
- Transfer stream (49 tasks, 72 paired questions in B; unseen functions under the new rule): no memory 41.7; plain 26.4; delete-contradicted 29.2; last-write-wins 20.8;
  ATMC v6 30.6; rule memory 34.7. Every memory is at or below no memory (as predicted); rule memory is the least harmful (-6.9, CI -19.4 to +4.2) and is significantly
  better only than last-write-wins (+13.9, CI +4.2 to +23.6); vs plain memory +8.3 (CI -2.8 to +19.4).
**Reading (the pre-registered branches):** R1, R2, R4 hold and R3 fails by 1.1 points. The claim we can defend: *after a rule change, rule memory recovers much better than
plain memory, than delete-contradicted and than last-write-wins, and it keeps its memory clean; it costs about 6 points of accuracy on the old rule before the change
(CI up to 0), stores 78% fewer notes, sends a longer prompt, and does not return to the no-memory level on unseen functions.* The R3 failure stays reported as a failure.

## Amendments
(none yet)
