# ATMC v3: the verified playbook

Written 2026-10-14, **before** the v3 run. Purpose: fix the finding that ATMC (trust + forgetting + merging) did not beat plain
memory on the local model (see LOCAL_MODEL_PILOT.md and docs/results/local_learning_with_diag/), using designs that other
agent-memory systems report as working. Nothing below has been run yet when this was written.

## What the research says (and what we took from it)
| Source | Idea | Used here as |
|---|---|---|
| ACE, *Agentic Context Engineering* ([arXiv 2510.04618](https://huggingface.co/papers/2510.04618)) | An evolving playbook updated by small itemised deltas (not rewrites); duplicates merged by embedding; usefulness tracked per entry; warns about brevity bias and context collapse | **Rule cards**: similar failure evidence is folded into one card by cosine similarity, each result kept verbatim (no summarising away detail) |
| ExpeL ([arXiv 2308.10144](https://arxiv.org/abs/2308.10144)) | Insights with ADD / UPVOTE / DOWNVOTE / EDIT; a later paper criticises putting every insight in every prompt | **Trust** is the vote count; only the top-k relevant cards are retrieved |
| Voyager ([arXiv 2305.16291](https://arxiv.org/html/2305.16291v2)), Dynamic Cheatsheet ([arXiv 2504.07952](https://arxiv.org/abs/2504.07952)) | A library of code that was *verified* before being saved, reused on later tasks | **Worked examples**: a solution is stored only if it passes the project's tests |
| Self-repair studies on small code models (e.g. [arXiv 2607.26117](https://arxiv.org/pdf/2607.26117), preprint) | Showing a small model its own failed code can anchor it on the same mistake; the *content* of the failure evidence is what helps | The retry is a **fresh** attempt given only the checker's evidence, never the failed code |

Deliberate difference from ACE/ExpeL: their reflector is an LLM. Our 7B reflector invented rules (LOCAL_MODEL_PILOT.md), so the
reflector here is plain code that copies the checker's output and counts type patterns. Nothing can be hallucinated.

## The three conditions (first-attempt pass rate on the 110-question benchmark, seeds 42/123/456)
- `static_repair`: plain memory + the worked examples (control for "do the examples alone explain a gain?")
- `atmc_v3_nocards`: trust-only ATMC + worked examples
- `atmc_v3`: trust-only ATMC + worked examples + rule cards (similarity threshold 0.75, chosen from cluster purity alone)

The reported pass rate is always the FIRST attempt; the retry only feeds memory. The retry also reports its own pass rate.

## Prediction and the rule for success (fixed before running)
- `atmc_v3` beats `static_repair` by **at least 3 points** on first-attempt pass rate with the 95% interval above zero.
- If it does, it must be confirmed once on new families written after this note (`conventions_heldout`) with no changes.
- If it does not, the honest conclusion stands: ATMC does not beat well-built plain memory on this model.
