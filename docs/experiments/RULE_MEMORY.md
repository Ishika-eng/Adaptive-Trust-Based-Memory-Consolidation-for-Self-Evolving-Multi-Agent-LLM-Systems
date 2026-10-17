# Rule memory (v7): one slot per rule, overwritten when the rule changes

Written 2026-10-15, BEFORE any v7 code ran. Motivation (docs/LITERATURE_GAP_ANALYSIS.md): the papers that show a clear gain organise what they learn
(MEMCoder keeps one guideline and the latest working code per API, bounded; ReMe prefers key-point experiences and an abstracted index), while our
memory stores raw per-question evidence retrieved by question wording, which cannot see that different functions share a rule; and TEPA-style
overwriting is a form of revocation.

## Design (all deterministic, no LLM in the memory)
- A **slot** = one rule. It holds: up to 6 checker-evidence lines (task + the checker's expected-vs-got, with types) and the **latest 2 verified solutions**
  (older ones are overwritten), rendered as ONE memory note, so the prompt budget is the same 3 notes as every other condition.
- A new evidence note joins the most similar slot (task-embedding cosine >= 0.75 AND a shared failure signature, or the slot has none yet); otherwise a new slot.
  A verified solution (a passed attempt or a verified retry) joins the most similar slot (cosine >= 0.75) or starts a slot.
- **Overwrite on change:** when a stored solution is proven stale (re-run against the current tests after a failure, as in v6) its whole slot is reset (evidence and
  solutions dropped) and the older slots within cosine 0.75 lose trust, as in v6. New evidence that lands in a demoted slot refounds it fresh instead of hiding.
- Memory is bounded per rule (6 evidence lines, 2 solutions); no eviction, so the slot count is reported.
Constants (0.75, 6, 2, penalty 0.35) are fixed here and not tuned.

## Conditions
`rule_mem` = trust-only ATMC + evidence-note trust + retry with memory + stale re-verification + rule memory (no utility ranking, no abstention).
Controls: `static_rm` (plain memory + examples + memory retry) and `atmc_v6` (v6), same seeds, already run.

## Pre-registered success rules
1. **Clean stream** (110 questions, seeds 42/123/456): `rule_mem` beats `static_rm` by >= 3 points of first-attempt pass rate, 95% bootstrap CI above 0.
2. **Dev drift stream** (`conventions_drift`, 58 tasks per seed), phase B: `rule_mem` beats `static_rm` by >= 10 points (CI above 0) AND is not more than 3 points
   below `atmc_v6` (non-inferior to the detector design).
3. Only if 1 and 2 hold: ONE run of `rule_mem` on the unseen stream (`conventions_drift2`), compared against the existing `static_rm` and `atmc_v6` runs there; the same rule 2 must hold. No changes.
If 1 fails: the headline stays the drift result (v6) and the rule memory is reported as tested, with its numbers. If 2 fails but 1 holds: report both.
Honest expectation: about a one-in-three chance for rule 1 (plain memory is hard to beat on stable data); rule 2 is more likely, since overwriting is itself a revocation.

## RESULT (2026-10-15; 3 seeds; clean 330 paired questions, drift 87 paired phase-B questions)
**Rule 1, clean stream: NOT MET.** First-attempt pass: rule memory 42.1% vs plain memory (static_rm) 46.4%: -4.2 points, 95% CI -8.5 to 0.0, p = 0.065; behind on all three seeds
(38.2/46.4/41.8 vs 41.8/48.2/49.1). After one retry 64.2% vs 68.8%. Memory size 49 slots vs 135 notes (-64%). Still +17.9 points above no memory (p < 0.0001).
**Rule 2, dev drift stream: MET.** Phase B (new rule): rule memory 32.2% vs plain memory 4.6%: +27.6 points (CI +18.4 to +36.8, p < 0.0001; needed >= +10).
vs atmc_v6 (26.4%): +5.7 points (CI -2.3 to +14.9, p = 0.30), so not more than 3 points below it (non-inferior; not shown to be better).
Per phase B thirds: 11.1 / 36.7 / 46.7% (v6: 14.8 / 16.7 / 46.7). Phase A: 51.7% (v6 43.7%). Still 12.6 points below no memory over phase B (p = 0.027).
Slots at the end: 20 / 21 / 26; stale solutions proven per run: 8 / 11 / 9.
**Per the rule fixed above, because rule 1 failed the unseen-stream run was NOT done and the headline stays the v6 drift result.**
Reading: bounding memory to one slot per rule costs about 4 points on stable data (the slots keep only 6 evidence lines and 2 solutions, so some per-question detail is lost),
but the overwrite makes recovery after a rule change faster in the middle of the stream. v6 gets most of the drift benefit with no stable-data cost (it ties plain memory there).
