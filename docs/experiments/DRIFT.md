# Rule change (drift) and abstention: predictions fixed before any run

Written 2026-10-14. Motivation: the closest published work says (a) outcome-based memory scores ("Memory Worth", arXiv 2604.12007)
show their clearest benefit on STALE memories, and (b) a coding-agent memory paper (NeurIPS 2026 workshop, "What Should Coding Agents
Remember?") improves by retrieving only validated repairs and abstaining otherwise. Our clean benchmark tests neither.

## Part 1: the rule-change stream (`conventions_drift`, built by `pilot/benchmarks/build_drift.py`)
29 questions in three families (dates, sorted output, error handling). The stream asks each question first under the OLD rule (phase A),
then again under the OPPOSITE rule (phase B). Every note saved in phase A about a question is therefore wrong for it in phase B.
Validated by running: the phase-B reference passes the phase-B tests and the old phase-A solution fails them. 58 tasks per seed.

Conditions: `no_memory`, `static_rm` (plain memory + worked examples + memory-aware retry), `atmc_v4` (learned utility, signature retry, rule cards).

Predictions:
- H1: in phase B, `static_rm` does WORSE than `no_memory` at the start (stale notes mislead). If false, drift is not hurting and the test is uninformative.
- H2 (the ATMC claim): over phase B, `atmc_v4` beats `static_rm` by >= 3 points of true pass rate with the 95% bootstrap CI above 0.
- If H2 fails, the conclusion is that our trust/utility mechanism does not protect against rule changes better than plain memory in streams of this length.
  (Honest expectation: utility is learned per note from repeated exposures; each note is shown only a few times here, so the effect may be small.)

## Part 2: abstention (clean 110-question stream, `learning_v5`)
`static_val`: plain memory that shows only verified solutions (worked examples and solved notes). `atmc_v5`: v4 + the same abstention (plus rule cards).
Predictions: abstention helps; `atmc_v5` beats `static_val` by >= 3 points with the CI above 0. Reported against the v4 runs of the same seeds as well.
In our v3 logs, questions that saw only bare evidence passed 8% (against 15% when nothing relevant was shown, p = 0.27), which motivates this; it is a
hypothesis from development data, so any win must still be confirmed once on the held-out families.

## RESULT, part 1: rule-change stream (2026-10-14, 9 runs, seeds 42/123/456, 29 questions x 2 phases)
| | phase A (old rule) | phase B (new rule) | B: first / middle / last third |
|---|---|---|---|
| no memory | 5.7% | 44.8% | 37.0 / 46.7 / 50.0 |
| plain memory (static_rm) | 52.9% | 4.6% | 3.7 / 3.3 / 6.7 |
| ATMC v4 | 46.0% | 10.3% | 7.4 / 6.7 / 16.7 |
- H1 CONFIRMED: after the rule flips, plain memory is far WORSE than no memory (-40.2 points, CI -50.6 to -29.9, p < 0.001): stale notes mislead.
- H2 NOT MET: ATMC v4 vs plain memory +5.7 points over phase B, 95% CI 0.0 to +11.5, McNemar p = 0.125. The point estimate exceeds +3 but the interval touches 0.
  ATMC was ahead or equal on all three seeds (10.3/3.4/17.2 vs 3.4/3.4/6.9), but it also collapses: 34.5 points BELOW no memory (CI -44.8 to -24.1).
- Reading: our trust/utility updates are far too slow for a rule change (each note is shown only a few times). Neither memory recovers within 29 questions.

## Part 3: re-verification detector (rules fixed 2026-10-14, before any run of it)
Why: part 1 showed both memories collapse after a rule change because trust/utility updates are too slow (each note is shown a few times).
Mechanism: after a verified FAILURE, every stored solution for the same function is re-run against the CURRENT checker. One that now fails was right
once and is wrong now -> proven stale. ATMC sets its trust to 0 and demotes every OLDER note within cosine 0.75 of it by 0.35 at once, so the
obsolete neighbourhood drops below the trust gate together. Plain memory with the same detector (`static_rv`) only drops the one proven-stale note.
Ideas: re-verification before reuse (dual-buffer/probation consolidation, arXiv 2603.07670), verification gating (Atlas, arXiv 2603.15666),
outcome-based deprecation (Memory Worth, arXiv 2604.12007). Inert on a clean stream by construction: it needs a stored solution for the same
function, and every question appears once there.

Conditions on the dev stream (`conventions_drift`, seeds 42/123/456, n = 58): `static_rv`, `atmc_v6`; compared with the existing part-1 runs of the same seeds.
Success rule (phase B, true outcomes, paired, 95% bootstrap CI):
 (a) atmc_v6 beats static_rm by >= 10 points, CI above 0;
 (b) atmc_v6 beats static_rv by >= 3 points, CI above 0 (ATMC's neighbourhood demotion adds something beyond simply dropping the stale note);
 (c) atmc_v6's phase-A pass rate is within 3 points of atmc_v4's (no harm before the change).
If (a)-(c) hold, run `drift2_final` ONCE on the second stream (`conventions_drift2`: usernames, error handling; written after the dev stream and never used
for any decision) with no changes, and the same (a)-(c) must hold. If (b) fails, the honest conclusion is that re-verification helps, but ATMC's
propagation adds nothing over a plain memory that uses the same detector.
Expectation: (a) likely (a stale twin is shown at similarity 1.0, so removing it matters), (b) uncertain.

## RESULT, part 2: abstention on the clean 110-question stream (2026-10-15, 6 runs, seeds 42/123/456, 330 paired questions)
First-attempt pass rate: plain memory shows every note 46.4% | plain memory + abstention 46.1% | ATMC v4 45.5% | ATMC v5 + abstention 40.0% (no memory 24.2%).
- Registered rule (atmc_v5 beats static_val by >= 3 points, CI above 0): **NOT met, and reversed**: -6.1 points, 95% CI -10.0 to -2.1, p = 0.0055.
- Abstention alone did nothing for plain memory (-0.3, CI -4.8 to +3.9) and hurt ATMC (atmc_v5 vs atmc_v4 -5.5, CI -10.0 to -1.2, p = 0.018).
- The motivating observation from the v3 logs (bare evidence passed 8% vs 15% for nothing) was noise (p = 0.27); this run does not support abstaining from bare evidence.
- Retry yield fell with abstention (plain memory fixed 67 of 175 failures vs 74 of 175; after retry 66.4% vs 68.8%): the retry also loses the evidence notes.

## RESULT, part 3: re-verification detector on the dev rule-change stream (2026-10-15, 6 runs, seeds 42/123/456, 87 paired phase-B questions)
| | phase A (old rule) | phase B (new rule) | B: first / middle / last third |
|---|---|---|---|
| no memory | 5.7% | 44.8% | 37.0 / 46.7 / 50.0 |
| plain memory (static_rm) | 52.9% | 4.6% | 3.7 / 3.3 / 6.7 |
| plain memory + detector (static_rv) | 51.7% | 4.6% | 0.0 / 3.3 / 10.0 |
| ATMC v4 | 46.0% | 10.3% | 7.4 / 6.7 / 16.7 |
| **ATMC v6 (v4 + detector with neighbourhood demotion)** | 43.7% | **26.4%** | 14.8 / 16.7 / 46.7 |
Pre-registered rules, all met on the dev stream:
 (a) atmc_v6 vs static_rm: +21.8 points, 95% CI +13.8 to +31.0, p < 0.0001 (needed >= +10, CI above 0) -> MET
 (b) atmc_v6 vs static_rv: +21.8 points, CI +13.8 to +31.0, p < 0.0001 (needed >= +3, CI above 0) -> MET. The plain-memory detector alone changed nothing
     (static_rv vs static_rm 0.0): finding the stale note and dropping only it is not enough; demoting its whole older neighbourhood is what helps.
 (c) phase A, atmc_v6 vs atmc_v4: -2.3 points, CI -10.3 to +5.7 (needed within 3) -> MET
Still 18.4 points below no memory over all of phase B (CI -28.7 to -8.0), although the last third reaches 46.7%, about the same as no memory (50.0%).
Caveat: the detector was designed after seeing the part-1 drift results on this same stream; its constants (radius 0.75, penalty 0.35) were fixed beforehand and not tuned.
Next, per the rule above: ONE run of `drift2_final` on the unseen second stream (`conventions_drift2`), no changes.

## RESULT, part 4: FINAL TEST on the unseen rule-change stream (2026-10-15, run once, nothing changed; 15 runs, seeds 42/123/456, 60 paired phase-B questions)
Stream `conventions_drift2` (usernames: lower-cased -> case-sensitive; error handling: return None -> raise), written after the dev stream and never used for any decision.
| | phase A (old rule) | phase B (new rule) | B: first / middle / last third |
|---|---|---|---|
| no memory | 15.0% | 45.0% | 38.9 / 38.1 / 57.1 |
| plain memory (static_rm) | 28.3% | 8.3% | 0.0 / 14.3 / 9.5 |
| plain memory + detector (static_rv) | 28.3% | 8.3% | 0.0 / 14.3 / 9.5 |
| ATMC v4 | 28.3% | 15.0% | 11.1 / 19.0 / 14.3 |
| **ATMC v6 (detector + neighbourhood demotion)** | 28.3% | **21.7%** | 11.1 / 28.6 / 23.8 |
Pre-registered rules, all met on the unseen stream, narrowly:
 (a) atmc_v6 vs static_rm: +13.3 points, bootstrap 95% CI +1.7 to +25.0 (needed >= +10, CI above 0) -> MET; exact McNemar p = 0.057 (not below 0.05)
 (b) atmc_v6 vs static_rv: +13.3, same CI (needed >= +3, CI above 0) -> MET (the plain-memory detector again changed nothing: static_rv vs static_rm 0.0)
 (c) phase A, atmc_v6 vs atmc_v4: 0.0 (identical before the change) -> MET
atmc_v6 was ahead of or equal to plain memory on all three seeds (phase B 0.30/0.15/0.20 vs 0.05/0.15/0.05). It remains 23.3 points BELOW no memory over phase B (CI -36.7 to -10.0).
The effect is smaller than on the dev stream (+13.3 vs +21.8), as expected when moving to unseen data. Sample is small (60 paired questions).
