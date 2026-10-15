# Literature gap analysis: citation chase and close reading (2026-10-15)

Purpose: find out where our findings stand against the closest published work, which of our claims are really new, where we are weaker, and what to fix in the system and the paper.

## 0. What was done, and its limits

| Step | What | Limits |
|---|---|---|
| Citation chase | Semantic Scholar "cited by" lists: **SAGE 83** papers, **Memory Worth 2**, **STALE 44**, **Evo-Memory 100** (the service caps at 100, so more exist) | The service rate-limited me; lists are what it returned. Venue and year are as listed there |
| Full-text reads | Memory Worth (whole), MEMCoder 2604.24222 (whole), STALE (first 100k of 187k characters), SAGE arXiv v2 (100k of 101k), Evo-Memory (100k of 108k) | Read through a summarising tool, not by me directly. **SAGE journal version unreadable** (publisher blocked it) |
| Abstract-only | The NeurIPS-workshop validated-repair paper (no arXiv id found), PACMI, TEPA, PROJECTMEM, How Memory Management Impacts LLM Agents, Causal Memory Policy, Suggested Is Not Superseded, COVE, EDV, AgentStream; SWE-ContextBench read as a summary | Anything below attributed to these is from abstracts or summaries. Read the full text before quoting |
| Not read at all | Dependency-Guided Rollback Repair (2608.10502), MemTX, ERRAND, "Do Self-Evolving Agents Forget?" (2605.09315), "Is One Score Enough?" (2605.15384), "Harness Updating Is Not Harness Benefit" (2605.30621), and about 200 other citing papers | Titles only. A real systematic review would triage these |

Every number is the authors' claim. None was independently checked.

## 1. The papers that matter most

| Paper | What it shows | Against us |
|---|---|---|
| **MEMCoder**, [arXiv 2604.24222](https://arxiv.org/html/2604.24222) (not the other "MemCoder") | Training-free evolving memory for **private-library code generation**; sequential stream; uses the **same model we use, Qwen2.5-Coder-7B**; memory has API, cross-API and task levels; task memories reranked by similarity times (Pass+1)/(Pass+Fault+2). Pass@1 vs naive RAG: Qwen2.5-Coder **25.2 → 41.8** (NdonnxEval) and **23.2 → 32.5** (NumbaEval); other 7B models +17 to +40. Self-evolving baselines ReMem and DC-RS did **not** beat naive RAG on 7B models (about 21 to 22) | **The nearest neighbour to our project.** Same model, similar gain size (ours +22 points vs no memory). They use an **LLM** to attribute errors and rewrite guidelines (audited accuracy about 80.8%); we use the checker's output with no LLM. They have **no plain-raw-episode ablation, no learning curve, no staleness test** |
| **Memory Worth**, [arXiv 2604.12007](https://arxiv.org/html/2604.12007) | Two counters per memory; converges to the success rate given retrieval (rank correlation 0.89 in simulation, 10,000 episodes); a stale memory's score fell from 0.97 to 0.17 over 3,000 episodes; **association, not causation** | Prior art for our utility ranking. **No coding agent and no verified outcomes** (synthetic or keyword outcomes). Its own future-work list names **"a live agent with natural outcome signals"** and a **non-stationary variant**: our v4 utility ranking and drift test are exactly that validation |
| **Causal Memory Policy**, [arXiv 2610.02070](https://arxiv.org/abs/2610.02070) | Utility estimates are **not identifiable** for memories that are rarely retrieved (fails for 54% of required memories on LongMemEval); fix: reserve retrieval slots filled with known probabilities; "no aggregation predicts a memory's value on unseen queries" | **Explains why our learned-utility ranking (v4) gained nothing**: it is associational |
| **TEPA**, [arXiv 2608.07429](https://arxiv.org/abs/2608.07429) | Keyed memories; when new evidence contradicts an active entry **under the same key**, it is revoked. **Real file-backed executable drift**: TEPA 0.950, append-only 0.203, no memory 0.298 | **Reproduces our central failure**: append-only memory collapses below no memory under drift (ours 4.6% vs 44.8%). Their fix is **per-key**. Our plain-memory-with-detector control is the same idea and **did not help**, because in our stream each question is asked once after the change |
| **STALE / CUPMem**, [arXiv 2605.06527](https://arxiv.org/html/2605.06527) | 400 conflict scenarios; best model 55.2%; CUPMem 68.0%; an **LLM adjudicator** retires stale entries; "propagation" only widens the candidate search; **"recognition does not imply application"**; no code or tests | Closest on "a change in one place invalidates related memories". Theirs is LLM-judged and conversational; ours is executable and in code. Their conclusion that only **active** items should ground the answer matches why our retrieval-level demotion works |
| **PACMI**, [arXiv 2610.05732](https://arxiv.org/abs/2610.05732) | Provenance graph with typed dependency edges; invalidation **cascades** to dependents; 100 cases; removing cascading raised errors 3 → 11 (not significant) | Prior art for propagating invalidation. Needs explicit dependency edges; ours infers neighbours by similarity |
| **Suggested Is Not Superseded**, [arXiv 2610.15710](https://arxiv.org/abs/2610.15710) | A binary retire gate retires **39.2%** of claims whose change was only suggested; three-way (superseded / needs confirmation / compatible) cut that to 15.5% at a cost in detection | **A risk to our neighbourhood demotion**: we never measured how many demoted notes were still valid |
| **SWE-ContextBench**, [arXiv 2602.08316](https://arxiv.org/html/2602.08316) | Unfiltered retrieved experience gives limited or negative benefit; compact oracle summaries help (26.3% → 34.3% for Claude Sonnet 4.5), free retrieval hurt (22.2%); unfiltered context cost about 27% more | Agrees with our stale-note collapse. **They report cost; we mostly don't** |
| **SAGE**, [arXiv 2409.00872](https://arxiv.org/html/2409.00872v2) / [Neurocomputing 647 (2025) 130470](https://sciencedirect.com/science/article/abs/pii/S0925231225011427) | Reflection + checker + Ebbinghaus forgetting; gains over the same model without SAGE (journal abstract: 2.26x on one task, +5.0 to +48.0 points on open models); only memory optimisation ablated, **no plain-memory control and no forgetting-only ablation**; text and tables disagree in places (arXiv v2) | Their forgetting claim has no control; **our measurement says forgetting cost 7.6 points**. In absolute points our gain over no memory is in their range |
| **Evo-Memory**, [arXiv 2511.20857](https://arxiv.org/html/2511.20857v1) | Task-stream benchmark; a simple retrieval baseline (ExpRAG) "outperforms several more complex designs"; gains correlate with task similarity (r about 0.56 to 0.72); failures unfiltered hurt baselines | Same protocol shape as ours; our tie with plain memory agrees with it. No coding dataset |
| [What Should Coding Agents Remember?](https://neurips.cc/virtual/2026/169590) (abstract only) | Validated repair episodes; abstains when nothing passes; 20.2% → 30.9%; contrastive 28.1% → 42.7% | Closest coding-memory paper; gains are against **no memory**. Our abstention test (v5) did not reproduce a benefit |
| COVE ([2608.01234](https://arxiv.org/abs/2608.01234)), EDV ([2606.24428](https://arxiv.org/abs/2606.24428)), AgentStream ([2608.00155](https://arxiv.org/abs/2608.00155)), PROJECTMEM ([2606.12329](https://arxiv.org/abs/2606.12329)) | Coordinating memory with weights; avoiding a "self-confirmation trap" by independent verification; "no method dominates across models and scenarios"; a local-first coding-agent memory with no controlled evaluation | Context. COVE supports our plan to consolidate memory into weights; EDV supports checking with an independent verifier (ours is the test suite) |

## 2. What this means for novelty

**Still defensible (narrowly):**
1. **Rule-level stale-memory handling in a coding assistant, triggered by executing stored code.** TEPA does keyed revocation with executable drift; CUPMem and PACMI propagate invalidation; Memory Worth detects staleness from outcomes. I found none that **re-runs a stored solution against the changed tests and then demotes the whole neighbourhood of related notes**. Our controlled finding is that revoking the proven-stale note alone changes nothing; the neighbourhood step is what helps (+21.8 and +13.3 points).
2. **Deterministic, evidence-only learning on a small local model** (no LLM reflector, no failed code shown, fresh retry stored as a verified example). MEMCoder reaches a similar gain size with LLM reflection of about 81% audited accuracy.
3. **A stricter evaluation**: pre-registered rules, a plain-memory control, paired statistics, negative results. MEMCoder, SAGE and most others report gains over weaker baselines.
4. **A conventions benchmark with rule-change streams.** Related: MEMCoder's private-library proxies, Learning to Commit, Naturalize. Mine is self-written and small.

**Not defensible:** trust or utility as an accuracy gain on stable data (ties; Memory Worth, MemRL and MEMCoder's utility rerank are prior art), forgetting (SAGE, MemoryBank, FadeMem; and it hurt here), merging (the "useful memories become faulty" result), and "first" or "novel" in any unqualified form.

**Usefulness.** Real: assistants face changing conventions, and the literature shows append-only memory fails badly there (TEPA: 0.20 vs 0.30 for no memory; ours: 4.6% vs 44.8%). Modest: the unseen-stream effect is +13.3 points with a lower interval bound of +1.7 and exact p = 0.057, and even our best system stays below having no memory after the change.

## 3. Gaps, in order of importance, with a concrete fix for each

| # | Gap | Evidence | Fix |
|---|---|---|---|
| 1 | **Only a self-written benchmark, one 7B model** | MEMCoder uses two libraries and five models; AgentStream finds benefit is non-monotonic in model strength | Run our system on a public or related benchmark with the same model (MEMCoder's NdonnxEval is the best match; BigCodeBench as backup) and add a second local model. Needs your approval for any download |
| 2 | **Missing baselines** | TEPA uses last-write-wins and append-only; Causal Memory Policy randomises retrieval slots | Add a **TEPA-style keyed-revocation** and a **last-write-wins** baseline, and a **randomised-slot exploration** variant of the utility estimate (tests whether causal estimation rescues utility ranking) |
| 3 | **Premature or wrong demotion is unmeasured** | "Suggested Is Not Superseded" (39.2% premature retirement) | Log every demotion, then report the share of demoted notes that were actually still valid; try a **pending state** (half penalty) for notes not themselves proven stale |
| 4 | **Rule-level versus fact-level drift is not separated** | TEPA works per key; ours needs neighbours because new questions follow the changed rule | Add a stream variant where questions **recur** after the change (shows keyed revocation helps there) and the current variant (new questions), to show where each mechanism applies |
| 5 | **Small drift samples** | 60 paired questions, p = 0.057 | More questions and seeds, plus a third stream; report effect sizes with intervals |
| 6 | **No cost or latency table** | SWE-ContextBench and SAGE report it | Add tokens per task and wall time per condition (the result files already hold them; memory runs use about 4x the prompt tokens of no memory) |
| 7 | **Memory-only; weights not changed** | COVE, Memento, ReMe discuss memory plus parameters | The planned LoRA step, tested on unseen questions with memory off |
| 8 | **Similarity radius is unvalidated** | MemStrata (summary only) warns similarity cannot separate contradictions | Measure radius purity on the benchmark and compare against a same-family oracle |
| 9 | **Several claims rest on abstracts** | see section 0 | Read the five closest papers in full before writing the final related-work section |

## 4. Suggested paper structure

1. **Problem:** self-evolving coding assistants learn project conventions, and conventions change.
2. **Method:** deterministic evidence notes, verified worked examples from an evidence-only retry, executable staleness detection with neighbourhood demotion.
3. **Benchmark:** hidden-convention questions and rule-change streams (validated by execution).
4. **Results, honestly ordered:** learning works (24% to 46%, about 69% after a retry); a plain-memory control is hard to beat on stable data (v1 to v5); **memory collapses under rule change (4.6% vs 44.8% with no memory) and the detector recovers it (+21.8, +13.3)**.
5. **Related work and comparisons:** MEMCoder, Memory Worth, Causal Memory Policy, TEPA, STALE/CUPMem, PACMI, SAGE, Evo-Memory, validated-repair paper.
6. **Limits:** self-written benchmark, one model, small drift samples, similarity-based neighbourhood, no weight updates yet.

## 5. Things to ask your guide
Which venues they expect; whether a self-written benchmark is acceptable if a public one is added; which papers on code-assistant memory or continual learning they know that a web search would miss.
