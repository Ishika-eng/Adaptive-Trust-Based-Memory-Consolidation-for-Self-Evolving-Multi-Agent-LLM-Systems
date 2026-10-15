# 13 · Lessons and mistakes

A build log is only useful if it is honest. This is the chronological list of what went wrong, how it was found, how it was fixed, and the rule it left behind. Many of these would otherwise be repeated by whoever rebuilds the project.

## Timeline of the build (from `git log`)

| Date | Commit (summary) | What it added |
|---|---|---|
| Aug 9 | Create README | The design document: dual loop, formulas, ablation plan (Sections 1–15 of `README.md`) |
| Sep 22 | ATMC engine, pilot experiments, demo app | `memory.py`, `stores.py`, `llm.py`, GSM8K pilot, poison pilot, first FastAPI + React demo; renamed from "TrustMem-Agent" to ATMC (a different paper already used that name) |
| Sep 23–24 | README status; SQLite persistence; compression pillar + 4-way ablation; 4-page frontend; mid-semester review assets | Persistence, the fourth pillar, the ablation runner |
| Oct 9 | Central config + real embeddings; results re-run and TF-IDF-era runs archived | `config.py`, `embeddings.py` |
| Oct 10 | Self-evolving coding assistant, evaluation harness, Assistant UI; results and response cache; relevance floor; chat assistant | Everything from chapters 05–08 |
| Oct 11 | Agent-component evaluation; follow-ups and suggestions; frontend redesign; auto-verification, strategy evolution, memory management; paper | Chapters 09–10, 12 |

The project began as a small **pilot** to answer one question quickly — "does trust-aware memory beat static memory on GSM8K under planted wrong notes?" — with deliberately minimal machinery (one seed, TF-IDF, a numeric checker). Only after that gave a signal was the full system built. **Pilot first, then build** is itself a lesson.

---

## The mistake log

### 1. Crediting every retrieved note (credit-assignment bug)
- **What happened:** the first trust update rewarded or punished *every retrieved note* with the task outcome.
- **Why wrong:** a wrong note that was retrieved but ignored got rewarded whenever the model succeeded anyway — exactly backwards.
- **Fix:** the model must end its answer with `USED: …`; only cited notes are updated (`extract_used_mask`, chapter 04). Results from before the fix were archived (`pilot/results/archive_pre_credit_fix/`).
- **Rule:** *ask who actually contributed to an outcome before assigning credit.*

### 2. The organic-mistake pilot found nothing
- The model rarely made mistakes on its own, so there was little bad memory to defend against (the **ceiling effect**). That is why experiments **plant** wrong notes.
- **Rule:** if your system fixes a rare event, create the event in a controlled way and say that you did.

### 3. Saying forgetting "purged" bad notes
- **Claimed:** ATMC deleted 8 of 14 poison notes through forgetting.
- **Found by reading the logs:** poison count dropped by 2 while memory dropped by 1 — the notes had been *absorbed by compression merges*. Pruning needs ~76 steps to fire (chapter 03) and our runs are 40.
- **Fix:** corrected the claim, added `test_prune_does_not_fire_inside_a_short_run`.
- **Rule:** verify a mechanism from logs before naming it as the cause.

### 4. A conclusion that reversed with better evidence
- Under TF-IDF and one seed, "forgetting matters more than trust" looked true. With real embeddings and 3–5 seeds, **trust** is decisive (111 vs 0 exposures) and forgetting is small and inconsistent.
- The earlier claim still sits in the mid-semester review slides.
- **Rule:** single-seed results are hypotheses. Archive old-regime results separately and never mix them in one table.

### 5. Poison that could not be built
- 27 of 60 coding tasks had no operator to flip, so no wrong version existed. Added a fallback (`return None`) and restricted edits to the solution region (editing a docstring changes nothing).
- **Rule:** verify every poison actually fails its tests.

### 6. Small plumbing slips
- GSM8K tasks had no `id` → `KeyError` in the curriculum.
- A vacuous planner test (passed even if the planner never ran) was replaced.
- The API test's fake model cited `USED: 1`, so the poison note's trust became 0.40 not 0.35 — which revealed a *real* limitation: **a wrong note cited in a correct answer still gains trust.**
- `report.py` produced `NaN` (invalid JSON) and read a default directory bound at import time.
- Backend: "No module named db" (path setup); a leftover server held port 8000.

### 7. The API quota
- Extra seeds died with `429 … PerDay`. Fix: `QuotaExhausted`, one file per run, resume, seed-major order.
- Two evaluation processes at once exceeded 15 requests/minute; one died. Run **one**.
- A key was invalidated mid-run; the run crashed, but the response cache replayed the finished calls on resume.
- A new key from the *same* Cloud project does not add quota; one from a *different* project does.

### 8. The response cache saved less than hoped
- We expected large savings. Measured: 56% in the agent experiment (conditions with identical prompts, resumed runs) but about one call per run in the ablation (prompts include retrieved notes and diverge).
- **Rule:** measure an optimisation before describing it.

### 9. A thread-safety bug that hid behind tests
- The cache shared one SQLite connection across the server's worker threads → `ProgrammingError`. Tests passed (single thread). The browser showed "Failed to fetch".
- **Fix:** `check_same_thread=False` + a lock; a test that calls from several threads; and `_quota_guard` so unexpected errors become readable JSON.
- **Rule:** an uncaught server error is invisible to the browser; always return structured errors. Test the concurrency model you deploy.

### 10. Auto-verification was fooled three ways
1. A vague request ("clean up a string") was "verified" because all solutions guessed the same meaning → added `UNDERSPECIFIED`.
2. Two wrong solutions agreed on one trivial test and caused a *wrong switch* → require a body of reliable tests.
3. A clear follow-up flagged vague because its conversation was missing from the test prompt → pass the context.
- **Rule:** consensus measures self-consistency, not truth. Test your verifier with cases designed to fool it.

### 11. A weak threat model, discovered late
- Mining the logs: in the static code runs the model cited a planted note on **28 steps and failed on one**. Our planted notes say `Outcome: INCORRECT` (and ATMC stores them at trust 0.35), so they are easy to ignore. The honest result is *exposure* reduction, not accuracy gain; the stronger attack (wrong notes recorded as **correct**) is untested.
- **Rule:** ask "would a sceptic say my attack is too easy?" before the results, not after.

### 12. A fix that did not pay off
- Memory sometimes cost a task or two (MBPP/444 and /501): three **irrelevant** notes in the prompt changed the model's answer though it said `USED: none`. An opt-in relevance floor halved tokens but let planted notes back in and did not recover accuracy; it is off by default.
- **Rule:** report trade-offs. Not every diagnosed problem has a free fix.

### 13. Over-claiming in the title and abstract
- "Multi-agent", "novel", "reduces token overhead", "optimises long-term reasoning" were each checked against the evidence and **not supported**. The final paper claims only: outcome-verified memory, ~10× fewer planted notes shown, trust is the decisive pillar, no significant accuracy gain from the extras.
- **Rule:** a claim is as strong as the experiment that could have shown it false.

## Habits worth copying

- **Commit one file per commit** (the team preference), with the evidence (result JSON) committed too.
- **Save config in every result.**
- **Read the logs** (per-step JSON) when a number surprises you.
- **Report negative results** in the same table as positive ones.
- **Check references** against sources; do not write citations from memory.
- **Screenshot the UI** in light, dark and phone widths before calling it done.
