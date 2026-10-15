# ATMC v4: learn which memory helps

Written 2026-10-14, **before any v4 code ran**. Follows ATMC_V3.md, where ATMC v3 tied plain memory + the same worked examples
(+0.9 points, 95% CI -3.6 to +5.2, p = 0.79; the pre-registered +3 rule was not met).

## What the saved v3 logs showed (analysis of `learning_v3__code__static_repair__seed*`, 3 seeds, rule questions)
Memory contents were reconstructed step by step from the logs (which notes plain similarity retrieval would have shown).

| Shown to the model | Share of questions | Pass rate |
|---|---|---|
| a worked example from the same rule family | 49% | 56% |
| only failure evidence from that family | 28% | 8% |
| nothing from that family | 22% | 15% |

So (1) worked examples carry nearly all of the benefit, (2) evidence alone barely helps, (3) ranking notes by question
similarity cannot reach across functions that share a rule but not wording (`none_on_invalid`: 23 of 30 questions saw nothing from
their family). Plain memory and ATMC both ranked by similarity and ATMC's trust had no usable signal (the model cites notes in
~12% of steps), so ATMC had nothing to add.

## Changes (each answers one finding; see also MemRL, ACE, ExpeL in ATMC_V3.md)
1. **Learned utility ranking.** Every retrieved note is credited with the outcome of the answer it was shown for (not only
   notes the model cites). Utility = (wins + 2*kind_rate) / (shown + 2), where kind_rate is the running pass rate of that *kind*
   of note (worked example / rule card / evidence / solved), so the system learns that examples help more than bare evidence.
   Ranking score = similarity * (0.5 + utility). Constants fixed here, not tuned.
2. **Failure-signature retrieval for the retry.** After a failed attempt, the retry is shown notes chosen by the checker's
   failure signature (expected type, got type) plus utility, so a rule seen in one function reaches a differently-worded one.
3. **Retry uses memory.** The retry prompt now includes retrieved notes. Plain memory gets the same: its retry shows its top-3
   similarity notes (control `static_rm`).

## Conditions (dev: the 110-question benchmark, seeds 42/123/456, first-attempt pass rate)
- `static_rm`: plain memory + examples + retry with similarity memory (the control)
- `atmc_v4_simonly`: ATMC cards + examples + retry with similarity memory, no learned utility (isolates change 1+2)
- `atmc_v4`: all three changes

## Success rule (fixed now)
1. Dev: `atmc_v4` beats `static_rm` by >= 3 points on first-attempt pass rate, 95% bootstrap CI above 0.
2. Only if (1) holds: ONE run on the held-out families (`conventions_heldout`, written after the v1-v3 tuning, never used for a
   decision) with the frozen v4 configuration against `static_rm`, 5 seeds; the same +3 rule must hold. No changes between.
3. If (1) fails, the conclusion stands: ATMC does not beat well-built plain memory on this model. No further variants on this benchmark.

## RESULT (2026-10-14, 9 runs: 3 conditions x seeds 42/123/456, 330 paired questions)
First-attempt pass rate: static_rm 46.4%, atmc_v4_simonly 47.9%, atmc_v4 45.5% (no memory 24.2%).
Pre-registered comparison, atmc_v4 vs static_rm: -0.9 points, 95% CI -5.2 to +3.3, p = 0.78. **The +3 rule was NOT met.**
The learned-utility ranking and signature retrieval did not help (atmc_v4 vs atmc_v4_simonly: -2.4, CI -6.7 to +1.8).
Memory-aware retry helped the plain-memory control (+3.3 vs the v3 plain+examples run, CI 0.0 to +7.0, p = 0.09, borderline).
Conclusion per the rule fixed above: ATMC does not beat well-built plain memory on this model on the clean benchmark; no further ATMC variants
are to be tried on this benchmark. The rule-change stream and abstention (DRIFT.md) are separately registered experiments.
