# Related work and novelty analysis for ATMC

> **Status banner (2026-10-15).** Sections 1 to 8 record the analysis as it stood on 2026-10-13, before the local-model experiments. **Sections 9 to 13 are current** and supersede them wherever they disagree (the claims in §3 and §4 are narrowed; the experiments proposed in §6 are mostly done). Read §11 and §12 first.

**Method.** Web search (standard and extended) plus reading of abstracts and, where the text was readable, full papers, run in October 2026. About 30 papers were examined; for each close one I extracted what it does and does **not** do. **Limits (please read):** this is not an exhaustive systematic review; for several papers (CoEvo-Mem, PoEM, Self-Cleaning, Σ-Mem, RMM, belief-based memory) I could read only the abstract; several of the closest works are 2026 preprints that are not peer-reviewed; the field is moving monthly. Re-run the searches before submission, and read each cited paper in full before relying on a statement about it.

---

## 1. What ATMC claims (decomposed into separable ideas)

| # | Idea |
|---|---|
| A | A per-note **trust score** that gates retrieval (and writes) |
| B | Trust **initialised and updated from deterministic, verified outcomes** (tests in a sandbox), not from an LLM judge |
| C | **Citation-conditioned credit**: only notes the model reports having used are updated |
| D | **Trust-dependent forgetting** and trust-gated importance in the ranking |
| E | **Merging** of old notes into summaries |
| F | **Evidence-weighted** learning: real tests = 1.0, user = 1.0, generated tests (with cross-solution consensus) = 0.5, plus a **dispute** path |
| G | **Strategy selection** per problem type (champion/challenger, Wilson bound, rollback) |
| H | **Weakness-targeted practice** curriculum |
| I | An **evaluation** of poison exposure vs accuracy with paired statistics and negative results |

## 2. Closest prior work, idea by idea

| Idea | Closest prior work | What it does | What it does **not** do (per what I could read) |
|---|---|---|---|
| **A trust-scored memory gating** | [Memory Poisoning Attack and Defense on Memory Based LLM-Agents](https://arxiv.org/abs/2601.05504) (2026) | Continuous trust 0–1 per memory; initial score from an LLM "guard agent" plus label checks and optional sandbox re-execution; **write-time and retrieval-time gating**; age-based trust decay | No outcome-driven update of a stored score over time; no retrieved-vs-used credit; reports trust is "highly susceptible to model overconfidence" and can be manipulated by "justification" clauses; small evaluation (one run per configuration) |
| | [A-MemGuard](https://arxiv.org/abs/2510.02373) (2025) | LLM-judge consensus over reasoning paths + separate "lessons" memory | No per-memory trust; failure detection is LLM-based, not ground truth; no forgetting |
| | [MemAudit](https://arxiv.org/abs/2605.23723), [Proof-of-Execution Memory](https://arxiv.org/abs/2608.16032) | Post-hoc causal attribution of harmful outputs to memories; a tamper-evident ledger of executed safety steps | MemAudit: post-hoc detection only (per abstract); PoEM: "does not inspect memory at all", no learned trust |
| **B outcome-driven per-memory scores** | [MemRL](https://arxiv.org/abs/2601.03192) (2026) | Per-memory Q-values updated from execution success/score; two-phase retrieval | Credits **all injected memories**; lists "credit-assignment ambiguity … raises the need for more precise attribution" as a limitation; no forgetting/pruning/merging; no trust gating or poisoning defense; evaluator type unspecified |
| | [CoEvo-Mem](https://arxiv.org/abs/2608.01739), [MemQ](https://www.alphaxiv.org/abs/2605.08374) | Task outcomes and trajectory feedback update memory values; credit over provenance graphs | No trust gate, forgetting or poisoning defense in the abstracts |
| | [ReasoningBank](https://arxiv.org/abs/2509.25140) (ICLR 2026) | Distils strategies from the agent's own successes **and** failures | Success/failure is self-judged; no per-note trust |
| **C citation-conditioned credit** | [RMM, "In Prospect and Retrospect"](https://arxiv.org/abs/2503.08026) (ACL 2025) | **The LLM's cited evidence is used as online RL feedback** to refine retrieval | Dialogue setting; signal is not tied to verified correctness; abstract does not say it updates per-note scores |
| | STAMP (July 2026, provenance-guided credit assignment for search agents; I did not confirm a stable link) | Credit to cited documents, checked by a reference verifier | RL for search agents, not memory |
| **D forgetting** | [MemoryBank](https://arxiv.org/abs/2305.10250) (AAAI 2024) | Ebbinghaus forgetting curve, strength reinforced on recall | No trust |
| | [FadeMem](https://arxiv.org/abs/2601.18642) (2026) | Adaptive exponential decay modulated by relevance, frequency and recency; fusion of memories | "Does not weight memories by how reliable their source is" (per search summary) |
| **E merging** | FadeMem (fusion) | — | — |
| | [Useful Memories Become Faulty When Continuously Updated by LLMs](https://arxiv.org/abs/2605.12978) | Shows LLM consolidation often produces **faulty** memories; recommends keeping raw episodes and gating consolidation | **A warning against our merging pillar** |
| **F generated-test evidence** | [CodeT](https://arxiv.org/abs/2207.10397), [CURE](https://arxiv.org/abs/2506.03136) | Generated tests + agreement to rank/reward solutions | Used for ranking/RL reward, not as down-weighted evidence gating a memory write; test reliability is an acknowledged open problem |
| | [Voyager](https://arxiv.org/abs/2305.16291), [Reflexion](https://arxiv.org/abs/2303.11366) | Verification/test feedback before storing skills or reflections | No continuing trust per entry |
| **G strategy selection** | [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) | Self-modifications validated empirically on benchmarks, archive of agents | Rewrites scaffolding; heavy; not per-task-type selection |
| | bandit LLM routers (PILOT, OrcaRouter) | Route queries across *models* | I found no paper doing bandit choice among prompting/solving *strategies* per task type (search not exhaustive) |
| **Multi-agent reliability** | [Σ-Mem](https://huggingface.co/papers/2607.27958) (2026) | Online **per-peer reliability** memory for multi-agent systems, updated from correctness feedback | — (this overlaps the "agent reputation" idea; see §5) |
| **Motivation** | [Self-Cleaning and Captured Anyway](https://arxiv.org/abs/2609.25052), [Manufactured Confidence](https://arxiv.org/abs/2606.29279), [AgentPoison](https://arxiv.org/abs/2407.12784), [MINJA](https://arxiv.org/abs/2503.03704), [PoisonedRAG](https://arxiv.org/abs/2402.07867) | Show that an agent's self-written or injected memory is a real error/attack channel | — |
| **Weights from verified experience** | [ReST-EM](https://arxiv.org/abs/2312.06585), STaR/RFT | Fine-tune on own samples filtered by a verifier (unit tests for code) | Established; our "trust-filtered self-training" would be an application |

Name collision: [TrustMem (arXiv 2606.25161)](https://arxiv.org/abs/2606.25161) is a different mechanism (RL-trained verification of memory *edits*), which is why the project is called ATMC.

## 3. Verdict, claim by claim

| Claim | Verdict |
|---|---|
| Trust-scored memory, gating retrieval/writes (A) | **Not novel.** Precedent exists and is recent |
| Outcome-driven per-memory scores (B) | **Not novel** as a mechanism (MemRL, CoEvo-Mem, MemQ) |
| Self-reported citations as a feedback signal (C) | **Not novel** as an idea (RMM). *Combined with a verified outcome to update a per-note trust score for defense:* **I found no paper that does this** (scope: the papers listed here) |
| Trust-dependent decay (D) | **Plausibly new but small**: I found none tying decay to trust, yet our own ablation shows it has a small, inconsistent effect |
| Merging (E) | **Not novel and risky** (see 2605.12978) |
| Evidence-weighted learning with consensus-gated generated tests and dispute (F) | **Incremental**: pieces exist (CodeT, CURE); the combination as a memory-write gate with down-weighting and user override I did not find |
| Strategy champion/challenger with rollback (G) | **Standard practice applied to a new place**; no gain shown |
| Weakness-targeted practice (H) | Not novel (Voyager has automatic curricula); no gain shown |
| Honest evaluation with negatives (I) | A strength, but small scale |

## 4. The defensible novelty statement

No component of ATMC is new on its own. The defensible contribution is a **specific combination and design stance**:

> *In the literature we examined, existing memory-poisoning defenses decide trust with LLM judgments or content/provenance checks at write time, and a leading trust-scoring defense reports that LLM-assessed trust is vulnerable to model overconfidence. Outcome-driven memory-value methods update scores from rewards but credit every injected memory and add no defense. ATMC anchors each note's trust in **deterministic verified outcomes** (no LLM judge in the loop), updates it **only for notes the model reports having used**, and links the resulting score to **retrieval gating, trust-dependent forgetting and evidence weighting**, with an explicit user override. We found no prior system that combines these in a poisoning-defense setting.*

Every clause above can be checked against the table in §2.

The sharpest, most checkable differences:
1. **vs. [2601.05504](https://arxiv.org/abs/2601.05504):** their stored trust is not updated by later outcomes (the text proposes "verification-of-effect" only as future work), they rely on an LLM guard whose overconfidence they document; ATMC's trust is outcome-anchored.
2. **vs. [MemRL](https://arxiv.org/abs/2601.03192):** they credit all injected memories and name that as a limitation; ATMC credits only cited notes and adds gating, forgetting and a defense setting.
3. **vs. [RMM](https://arxiv.org/abs/2503.08026):** they use citations to refine a retriever in dialogue; ATMC ties them to verified correctness and per-note trust.

## 5. What is **not** supported, and what reviewers will ask

- **No demonstrated benefit in accuracy.** The base model already scores ~90%; our evidence is exposure reduction.
- **Weak attack.** Planted notes are recorded as *failed* attempts. The literature attacks (MINJA, AgentPoison) are not evaluated; we do not compare to A-MemGuard or to LLM-judge trust defenses.
- **Citation reliability.** Self-citation can be wrong; our parser falls back to "all used" when it cannot parse, and a cited-and-correct wrong note still gains trust.
- **Merging may corrupt memory** (2605.12978) and our counter shows poison can be hidden inside summaries.
- **Scale:** 40 tasks per run, 3–5 seeds, one model family, Python and arithmetic only.
- **Multi-agent reputation is not novel** (Σ-Mem tracks per-peer reliability from correctness feedback). Any multi-agent extension must be differentiated, for example by tying authorship to *memory poisoning containment*.

## 6. Experiments that would make the novelty claim stick

1. **Citation ablation (cheap, direct).** ATMC with credit to *cited* notes vs. credit to *all retrieved* notes (the MemRL-style rule). This tests claim C head-on.
2. **LLM-judge trust baseline.** Re-implement a 2601.05504-style guard (LLM assigns the trust score) and compare to outcome-anchored trust under the same attacks.
3. **Wrong notes recorded as correct**, and a **MINJA-style** injection on a verifiable task, so the attack is as hard as the literature's.
4. **Trust-dependent vs trust-independent decay** (claim D) and **merging on/off with a faithfulness check** (claim E).
5. A task or model with **headroom** so accuracy effects are measurable.

## 7. Recommended wording

- **Use:** "outcome-anchored trust for agent memory", "citation-conditioned credit assignment", "we found no prior system combining … (to the best of our search, October 2026)".
- **Avoid:** "novel", "first", "state of the art", "multi-agent" (unless you build and test it), "self-evolving" without "memory-level".

---

## 8. Update after Experiment 1, batch 1 (2026-10-13)

`docs/experiments/EXP1_credit_assignment.md` §10 tested claim **C** (credit only the notes the model cited) against crediting every retrieved note, with wrong notes recorded as successes. **Result: not supported.** The trust-AUC difference was +0.039 (below the pre-registered 0.05), and both rules ended below chance (0.43 and 0.39). The model named notes on only 7% of steps, so the cited-only rule barely updates anything.

Revisions to the claims in §3 and §4:

| Item | Revised status |
|---|---|
| C — citation-conditioned credit | **Withdrawn as a contribution** (not better than crediting all, on this model and prompt) |
| Defensible core (A, B) | Outcome-anchored trust with no LLM judge in the loop remains a difference from the cited prior work, **but** it is only shown to work when wrong notes are recorded as failures (the earlier experiments); when they are recorded as *successes* it did not separate them |
| New candidate | **Asymmetric credit** (reward cited notes only; blame every retrieved note on failure). **Confirmed on fresh seeds** (5 of 5 seeds: AUC 0.71 vs 0.39, later exposure down about 92%, 7% of genuine notes blocked, pass rate −1.2 points; see EXP1 §11). Still one model, one domain, and a plausible dependence on the retrieval lock-in. **I have not searched the literature for it yet**, so no novelty claim is made |

Do not use the sentence in §4 that lists citation-based updating as a distinguishing element until a later experiment supports it.

**Final (all five seeds):** the cited-vs-all difference stayed at +0.024 (higher in 2 of 5 seeds), so claim C remains withdrawn. The asymmetric-credit result replicated on the two fresh seeds. Before presenting asymmetric credit as new, search the literature for asymmetric reward/blame credit assignment in agent memory and for "blame all retrieved on failure" rules.

---

## 9. What happened to the proposed experiments, and the local-model results (2026-10-15)

All results below are from the local Qwen2.5-Coder-7B and a self-written benchmark of hidden project conventions (110 questions, plus two rule-change streams); three seeds, paired statistics, predictions written before each run (docs/experiments/). Each row replaces or tests a claim in §3.

| ATMC idea (§1) | Test | Result | Status |
|---|---|---|---|
| A trust gate | Trust-only ATMC vs plain memory, clean stream | 38.2% vs 40.0% (-1.8, p = 0.38) | **Ties; no accuracy benefit** |
| A trust as a defence | Checker wrongly passes 25% / 50% of wrong answers | trust-only 35.5% / 34.2% vs plain 34.8% / 31.8% (+0.6 / +2.4, n.s.) | **Not shown** on the local model. On Gemini, trust cut planted wrong notes shown from 52.7 to 0.0 per run (the earlier result stands) |
| B outcome-driven scores | Learned utility ranking + failure-signature retrieval (v4) | 45.5% vs 46.4% (-0.9, p = 0.78); pre-set +3 rule failed | **No gain** |
| C citation credit | Exp 1, five seeds | withdrawn (§8) | **Withdrawn** |
| Asymmetric credit | Same candidate on the local model | clean -10.6 points (29.4% vs 40.0%), 25% noise -3.9, 50% noise 0.0 | **Not supported** beyond its Gemini result |
| D forgetting | Added to trust-only ATMC | 38.2% -> 30.6% (-7.6, p = 0.002) | **Hurts** on this task stream |
| E merging | LLM merging; deterministic rule cards | merging 30.9% (no change); rule cards +3.3 over no cards (n.s.) | Neutral. Consistent with the warning that LLM-rewritten memory degrades (§2) |
| Abstention (show only verified notes) | v5 | plain memory 46.1% vs 46.4% (no change); ATMC 40.0% (-6.1 vs plain, p = 0.006) | **Does not help** |
| Plain memory vs no memory | core effect | 24.2% -> 40.0% (+15.8, p < 0.0001); 46.4% with worked examples and memory retry; about 69% after one evidence-guided retry | **Supported** |
| Stale-note handling (new, J below) | Rule-change stream, re-verification + neighbourhood demotion (v6) | dev stream: 26.4% vs 4.6% plain (+21.8, CI +13.8 to +31.0); **unseen stream: 21.7% vs 8.3% (+13.3, CI +1.7 to +25.0, McNemar p = 0.057)**; plain memory with the same detector: no change | **Supported, moderately** (small sample) |

## 10. Additional prior work found in the second search pass

| Paper | What it does | Relation to ATMC |
|---|---|---|
| **SAGE**, [arXiv 2409.00872](https://arxiv.org/html/2409.00872v2), journal version [Neurocomputing 647 (2025), 130470](https://sciencedirect.com/science/article/abs/pii/S0925231225011427) | User, Assistant and Checker agents; reflection; memory decay by the **Ebbinghaus forgetting curve**. Reported gains over the same model without SAGE (journal abstract: GPT-4 2.26x on database tasks; open-source models +5.0 to +48.0 absolute points; arXiv abstract: "57.7% to 100%") | The nearest published design to our forgetting. As far as I could read, **no plain-memory control and no ablation of the forgetting curve**; our measurement says forgetting cost 7.6 points here. (MARS, [arXiv 2503.19271](https://arxiv.org/pdf/2503.19271), carries the same method.) I could not read the journal full text |
| [Memory Worth, "When to Forget"](https://arxiv.org/pdf/2604.12007) | Two counters per memory (success or failure when retrieved); provably converges to the conditional success rate; deprecates stale memories; associational, not causal | **Prior art for outcome-based per-memory scores and staleness deprecation.** Shown in simulation over thousands of episodes; ours uses an executable check to react within tens of questions |
| [What Should Coding Agents Remember?](https://neurips.cc/virtual/2026/169590) (NeurIPS 2026 workshop) | Retrieve only **validated repair episodes**; a relevance model can abstain; 20.2% -> 30.9% on 94 held-out tasks; contrastive retrieval 28.1% -> 42.7% | The closest coding paper. Our abstention test did **not** reproduce a benefit; gains there are against no memory, not against a plain retrieval baseline |
| [Evo-Memory](https://arxiv.org/html/2511.20857v1) | Task-stream benchmark for test-time memory evolution; the simple ExpRAG baseline "outperforms several more complex designs"; gains grow with task similarity | Our protocol is the same shape; our tie with plain memory on stable data agrees with their single-turn result |
| [ReMe](https://arxiv.org/pdf/2512.10696) | Keypoint experiences, scenario-aware indexing, utility-based deletion; Qwen3-8B with memory beats Qwen3-14B without | The small-model precedent for our setting |
| [Useful Memories Become Faulty When Updated by LLMs](https://neurips.cc/virtual/2026/174843) | LLM consolidation degrades memory; keep raw episodes, gate consolidation | Our rule cards are deterministic and verbatim, which fits their recommendation |
| [ACE](https://arxiv.org/pdf/2510.04618), [ExpeL](https://arxiv.org/pdf/2308.10144), [Dynamic Cheatsheet](https://arxiv.org/abs/2504.07952), [Voyager](https://arxiv.org/abs/2305.16291), [ReasoningBank](https://arxiv.org/pdf/2509.25140) | Itemised playbook updates; insight votes; reuse of validated code; verified skill library; strategies from successes and failures | Prior art for worked examples and rule-style notes |
| [MemCoder](https://arxiv.org/abs/2603.13258), [SWE-Exp](https://www.microsoft.com/en-us/research/?p=1180545), [SICA](https://arxiv.org/pdf/2504.15228), [Live-SWE-agent](https://arxiv.org/html/2511.13646v3) | Coding agents that learn from validated experience, evolve tools, or edit themselves; all on SWE-bench-style tasks with large models | Different scale and level (they evolve the scaffold; we evolve memory) |
| [STALE](https://arxiv.org/pdf/2605.06527) and its CUPMem system | Benchmark for noticing that a memory is no longer valid; best model 55.2%; CUPMem adds structured state consolidation and **propagation-aware search** | **The closest published work to our stale-note idea**, in conversational memory. Their propagation is learned or LLM-based; ours is triggered by executing the stored solution against the current tests |
| MemStrata (June 2026, known only from a search summary; **I did not confirm its link**, check before citing); *bettermemory* (a vendor package, unverified) | Embedding similarity cannot separate contradictions from restatements, so use deterministic supersession; a coding memory tool scores memories by file and commit changes | Warns that our neighbourhood demotion, which uses embedding similarity, can be wrong. Ours demotes only after an executable contradiction is proven |
| [Learning to Commit](https://arxiv.org/html/2603.26664), [Naturalize](https://arxiv.org/pdf/1402.4182) | Repository memory for conventions; learning a codebase's naming conventions | Related to the conventions setting. I found **no benchmark that hides conventions and measures recovery from verifier feedback** (one search) |
| Self-repair on small code models ([preprint](https://arxiv.org/pdf/2607.26117) and others) | Small models can anchor on their own failed code; the specific failure evidence is what helps | Supports retrying from evidence only |

## 11. The current defensible novelty statement

No component is new on its own, and several of our original claims have been dropped (C, D, merging, abstention, trust as an accuracy or noise defence on the local model). What remains, in order of strength of evidence:

1. **Executable staleness detection for a coding assistant's memory.** After a failed answer, every stored solution for the same function is re-run against the current tests; one that now fails is **proven stale**, and the older notes about similar tasks are demoted together. On two rule-change streams this recovered faster than plain memory (+21.8 points on the stream it was designed on, **+13.3 on an unseen one**), while a plain-memory control with the same detector did not recover at all. The mechanism that matters is the **neighbourhood demotion**, not the detection. Prior art exists for outcome-based staleness (Memory Worth), propagation-aware revision (CUPMem) and deterministic supersession (MemStrata); I found none that triggers it by re-executing stored code against changed tests in a coding assistant, but my search for this was narrow (three queries).
2. **A fully local, private self-evolving coding assistant** on a 7B model that learns hidden project rules from its own verified failures (24.2% -> 46.4% first attempt, about 69% after one retry), with a retry that uses only the checker's evidence (never the failed code) and stores the verified fix as a worked example. The ingredients exist; the small-model, local, conventions-learning setting is the contribution.
3. **A controlled, pre-registered comparison against a plain-memory control**, including negative results (forgetting, abstention, citation credit and utility ranking did not help). Several published memory designs report gains only against no memory; this is the stricter test.
4. **A validated benchmark of hidden conventions with rule-change streams** (self-written; see limits).

## 12. What is not supported, and what reviewers will ask

- **ATMC does not beat a well-built plain memory on stable data** (ties at v3 and v4; loses at v1 and v5). Say so.
- **Stale-note result: moderate evidence.** 60 paired questions on the unseen stream, interval lower bound +1.7, exact p = 0.057. The detector was designed after seeing the first stream's results (its constants were fixed in advance, not tuned). After a rule change ATMC is **still below having no memory** (21.7% vs 45.0%).
- **Breadth.** One self-written benchmark, one 7B model, Python only, three seeds. Published systems use public benchmarks (AgentBench, AppWorld, SWE-bench). A public-benchmark run (for example BigCodeBench) has been scoped but not done.
- **Similarity-based demotion.** The neighbourhood radius uses embedding similarity, which MemStrata argues is unreliable for contradictions. We have not measured how often the radius demotes a note of a different rule.
- **Comparison with SAGE.** Their gains are against the same model without their method (+5.0 to +48.0 points on open models). Ours against no memory is +22 points first attempt (+45 after the retry): in range, but not the same benchmark.
- **Trust as a defence** was shown on Gemini (planted wrong notes), not on the local model under a fooled checker.
- **No weights were trained.** "Self-evolving" here means memory-level only.
- **Not done:** LLM-judge trust baseline; MINJA-style injection; held-out 30 questions (built, unused); fine-tuning.

## 13. Recommended wording (replaces §7)

- **Use:** "executable re-verification of stored solutions to detect stale memory after a rule change"; "learns project conventions from verified failures on a small local model"; "pre-registered comparison against a plain-memory control"; "to the best of our (limited) search, October 2026".
- **Avoid:** "novel", "first", "state of the art", "trust makes learning safe", "forgetting improves accuracy", "multi-agent", "self-evolving" without "memory-level", and any claim that ATMC beats plain memory on stable data.
- **Cite as nearest prior work:** SAGE (forgetting curve), Memory Worth (outcome scores and staleness), the validated-repair paper (coding memory), STALE/CUPMem (propagation of invalidation), Evo-Memory (protocol), ReMe (small models).

## 14. Corrections after the citation chase and full-text reads (2026-10-15)

Full analysis: [LITERATURE_GAP_ANALYSIS.md](LITERATURE_GAP_ANALYSIS.md). Changes to statements above:

- **§10 said Memory Worth was shown "in simulation over thousands of episodes".** After reading the paper: it also has a small text experiment, but its outcomes are synthetic or keyword-matched, it has **no coding agent and no verified outcomes**, and its own future-work list names a live agent with natural outcome signals and a non-stationary variant. Our experiments are that live validation.
- **§10 described CUPMem's propagation as "learned or LLM-based".** More exactly: an **LLM adjudicator** labels old entries keep, stale, replace or unknown; the "propagation" step only widens the candidate set the adjudicator sees. No code or tests are involved.
- **New nearest neighbours not in §10:** MEMCoder for private-library code generation ([2604.24222](https://arxiv.org/html/2604.24222), same Qwen2.5-Coder-7B, sequential stream), TEPA ([2608.07429](https://arxiv.org/abs/2608.07429), executable drift, keyed revocation), PACMI ([2610.05732](https://arxiv.org/abs/2610.05732), cascading invalidation), Causal Memory Policy ([2610.02070](https://arxiv.org/abs/2610.02070), why associational utility fails on unseen queries) and "Suggested Is Not Superseded" ([2610.15710](https://arxiv.org/abs/2610.15710), premature retirement).
- **§11 item 1 must be worded narrowly:** keyed revocation under executable drift (TEPA) and cascading invalidation (PACMI) exist; what remains is re-running a stored solution against changed tests and demoting the related neighbourhood in a coding assistant, which a plain keyed-revocation control did not reproduce in our stream.
