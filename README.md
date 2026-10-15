# Adaptive-Trust-Based-Memory-Consolidation-for-Self-Evolving-Multi-Agent-LLM-Systems

## Project Goal

This project builds a **self-evolving multi-agent LLM system**: an agent that autonomously plans, acts, checks its own work, and improves over time by curating what it remembers and trusts.

The core of that agent, validated, is its **memory subsystem — ATMC (Adaptive Trust-based Memory Consolidation)**; the surrounding loop (planner, sandboxed code actor and tests, reflection, curriculum, persistence) is now implemented and being evaluated. ATMC is the component responsible for one specific job: deciding which past experiences an agent should trust, keep, and retrieve, versus discard — using dynamic Trust Scoring ($T$), Importance Prioritization ($I$), and Adaptive Forgetting ($F$) instead of storing every reflection unweighted forever.

**The relationship, stated plainly:** ATMC is the memory-governance piece a self-evolving agent needs. It is one component, not the whole agent — the agent also needs planning, tool use, and autonomous goal-setting (implemented, see Part B below, but not yet validated to the same standard). Do not read "ATMC" and "the self-evolving agent" as the same thing anywhere in this document — where they overlap, it's called out explicitly.

> **Naming note:** this project was originally developed under the working name "TrustMem-Agent." It has been renamed to ATMC to avoid collision with an unrelated prior-art paper, *TRUSTMEM: Learning Trustworthy Memory Consolidation for LLM Agents with Long-Term Memory* (arXiv 2606.25161), which targets a different problem (RL-trained verification of memory-editing operations) but shares the name and the general subfield.

---

## Implementation Status

### Part A — ATMC (the memory subsystem): validated, working

**A real, reproducible experimental result:**
On GSM8K with deliberately planted wrong notes, trust-gated retrieval cut exposure to those notes by about **10×** versus a static "remember everything" baseline (3 seeds, real sentence embeddings, no accuracy loss). Two caveats matter: (1) ATMC's remaining exposures are a structural cold-start floor — in the first three tasks the poison notes are the only memories, so nothing can distinguish them yet; after warm-up ATMC's exposure was 0 versus ~51 for static. (2) A note that is cited *and* ends up in a correct answer still gains trust, so a wrong note the model happens to cite on a right answer is not penalised. Logs: [`pilot/results/`](pilot/results/).

**Code domain (HumanEval/MBPP, 40 tasks, planted notes that are verified to fail their tests):** the model already passes ~89% with no memory, so accuracy does not separate the systems (all McNemar p = 1.0). What differs is poison: static memory had ~53 poison exposures per run after warm-up and relied on a planted note ~9 times; ATMC had 0 after warm-up on all 3 seeds. Accuracy is not separated (all paired McNemar p = 1.0; ATMC 88.3% vs no-memory 89.2%), so the code-domain benefit is poison protection, not pass rate.

**Built and working, all of it ATMC-scoped (Sections 2, 3, 5 below):**
- Core scoring math — Trust, Importance, Decay, Composite Score (Section 2) → [`pilot/memory.py`](pilot/memory.py)
- Memory store with the hard verification gate, trust-gated composite retrieval, and citation-conditioned trust updates (a memory's trust only moves if the model says it actually relied on it — closes a credit-assignment gap found during testing) → [`pilot/stores.py`](pilot/stores.py)
- A deterministic Checker (ground-truth comparison, one of the mechanisms Section 1.2 allows) and a Reasoning Agent (Gemini) calling into ATMC → [`pilot/llm.py`](pilot/llm.py)
- Pilot scripts reproducing the result above across multiple seeds → [`pilot/experiment.py`](pilot/experiment.py) (`poison` preset)
- A full-stack demo app (FastAPI backend + React frontend) running ATMC live, interactively, so the result above can be watched rather than just read → [`app/`](app/)

**Also built since the first pilot:** Memory Compression (the fourth pillar, independently switchable), the 4-way ablation runner (Full / w/o Trust / w/o Forgetting / w/o Compression), real sentence embeddings, a central config file, and SQLite persistence for the demo.

**Not yet done, still ATMC-scoped:**
- **Ablation (math 5 seeds, code 3 seeds):** removing trust lets ~111 planted wrong notes through on every seed versus 0 for the full system; removing forgetting is smaller and inconsistent (math 14.8 ± 20.3, code 26.0 ± 19.1); compression cuts memory 22% with no extra exposure. Pass rate does not separate any condition (p = 1.0). The earlier TF-IDF-era claim that "forgetting matters more than trust" did **not** hold under real embeddings; the mid-semester review slides still contain it. Forgetting acts through decay in the ranking, not deletion — pruning needs roughly 70 steps to fire, longer than these runs.
- Compression merges notes (memory 42 vs 54 in the ablation) without raising poison exposure, but merging can mix a poisoned note into a clean summary (`n_poison_laundered` counts this).
- **Irrelevant notes can cost a little accuracy (diagnosed, not fixed).** ATMC always injects the top-k notes however dissimilar. On the hardest code sample (seed 456) it failed two tasks (MBPP/444, 501) that no-memory solved, with three irrelevant notes in context and `USED: none` — the model ignored them but its answer still changed. An opt-in relevance floor (`retrieval.min_similarity`, tested at 0.75) halves prompt tokens (624 to 342 per task) but does not recover accuracy (88.3% either way) and lets ~7.7 planted notes through per run versus 0, so it is off by default. One threshold, tuned partly on that failure; the accuracy gap to no-memory (88.3% vs 89.2%) is within noise.
- Baselines beyond static and a Reflexion-style buffer (SAGE-style, A-MEM — Section 9).
- Full-dataset evaluation is not done, and the held-out sets are small (20 tasks per seed); the Gemini free tier allows ~500 calls/day, which limits seed count. Run-to-run noise at temperature 0 is a couple of tasks per 40, so differences of a few points are within noise.

### Part B — The self-evolving agent (built around ATMC): implemented and evaluated, extra components not shown to help

One shared loop ([`pilot/agent.py`](pilot/agent.py)) serves two domains (math, code):
- **Planner** ([`pilot/planner.py`](pilot/planner.py)) — writes a plan without seeing memory.
- **Actor + Checker** — code is run against the real tests in a subprocess sandbox ([`pilot/sandbox.py`](pilot/sandbox.py): macOS `sandbox-exec` denying network, writes outside a temp dir and reads of secrets; scrubbed environment; resource limits). It is **not** a hard security boundary against malicious code.
- **Reflection** ([`pilot/reflection.py`](pilot/reflection.py)) — turns failures into lessons stored in ATMC memory.
- **Curriculum** ([`pilot/curriculum.py`](pilot/curriculum.py)) — fixed, random, or weakness-targeted task choice.
- **Persistence** — memory and history are saved in SQLite and survive restarts.
- **Domains** ([`pilot/domains/`](pilot/domains/)) — GSM8K and HumanEval/MBPP (four MBPP tasks with known-broken tests are excluded).

**Agent-component evaluation (code domain, 3 seeds, 40 training + 20 held-out tasks each, preset `agent`):**

| Condition | Held-out pass rate | Tokens/task |
|---|---|---|
| ATMC alone | 88.3% ± 2.9 | 1164 |
| + planner | 95.0% ± 5.0 | 1526 |
| + planner + reflection | 95.0% ± 5.0 | 1561 |
| + reflection only | 88.3% ± 2.9 | 1193 |
| random task choice | 90.0% | 1178 |
| weakness-targeted task choice | 90.0% | 900 |

No component shows a significant benefit. The planner is the only lead: +6.7 points held-out (better on 4 tasks, worse on 0; bootstrap CI [+1.7, +13.3]) but exact McNemar p = 0.125, since only 4 tasks changed — suggestive, not established — and it costs ~31% more tokens and twice the calls. Reflection changed nothing (identical to plain ATMC on every task): it runs only after a failure and the model fails too rarely at a ~90% baseline. Neither curriculum beat plain ATMC held-out (+1.7 points, p = 1.0); the weakness-targeted curriculum's lower token use comes from the tasks it picks, not from efficiency. A larger held-out set (~60 tasks per seed) would be needed to confirm the planner effect.

"Implemented" is not "validated": the loop works and is reliable, but its extra components have not been shown to improve accuracy at this benchmark's ceiling. Still absent: a reusable skill library — memories are raw text, not consolidated procedures. The system is training-free: the LLM's weights never change; "self-evolving" here means the memory and curriculum change.

> **Want to understand or rebuild this project from scratch?** Start with the developer guide: [`docs/guide/README.md`](docs/guide/README.md) (14 chapters, in build order, with checkpoints).

## Getting Started

**Setup (from the repo root):**
```bash
python3 -m venv pilot/.venv && source pilot/.venv/bin/activate
pip install -r requirements.txt
echo "GEMINI_API_KEY=your_key_here" > .env
```

**Run the experiments** (each run is saved on its own and skipped if already done, so a run stopped by the API quota resumes when you re-run the same command):
```bash
python3 pilot/experiment.py --list                                  # presets and conditions
python3 pilot/experiment.py poison    --domain math                 # static vs ATMC under planted wrong notes
python3 pilot/experiment.py ablation  --domain math                 # Full / w/o Trust / w/o Forgetting / w/o Compression
python3 pilot/experiment.py baselines_poison --domain code          # none / static / Reflexion-style / ATMC
python3 pilot/experiment.py agent     --domain code --heldout 20    # planner, reflection, curriculum
python3 pilot/report.py baselines_poison --domain code              # tables, bootstrap CIs, exact McNemar
pilot/.venv/bin/python -m pytest tests -q                           # 148 tests
```
Results go to `pilot/results/eval/`. Quick overrides: `PILOT_SEED`, `PILOT_N_TASKS`, `PILOT_EMBEDDING` (`fastembed` | `gemini` | `tfidf`).

**Configuration:** every tunable (trust reward/penalty and minimum, decay rates, composite
weights, compression schedule, model and temperature, embedder, seeds) lives in
[`configs/default.yaml`](configs/default.yaml). Each results JSON stores a snapshot of the exact
config it ran with, so any run can be reproduced from its own output. Invalid values
(e.g. composite weights not summing to 1) are rejected at load time.

**Embeddings:** relevance is measured with a local sentence-embedding model
(`BAAI/bge-small-en-v1.5` via `fastembed` -- no API quota, no PyTorch). The original TF-IDF
backend is kept (`embedding.backend: tfidf`) so the two can be compared directly.

**Run the assistant (chat UI):** ask your own Python questions at http://localhost:5173. Add `assert` tests and the
sandbox checks the answer; without tests the answer is marked *unverified* and the assistant learns from it only after you
click "It worked" / "It didn't". Messages in a conversation are follow-ups (the last 3 exchanges, including whether each
passed or was rejected, go into the prompt); "+ New chat" starts a clean context but keeps memory. The sidebar shows the
assistant's own **next practice task**, chosen by the weakness curriculum (a benchmark task close to something it got wrong,
or a new one while it has no mistakes) with the reason; "Let me practise it" solves it against its real tests, so memory
learns verified results. Python only. Needs `GEMINI_API_KEY` in `.env`.

**What the assistant does beyond answering** (all in `pilot/` and the Assistant / Memory / Evolution tabs):

- **Auto-verification** (`pilot/autoverify.py`). With no tests supplied, it writes edge-case tests plus two alternative
  solutions with different approaches and runs everything in the sandbox. A generated test counts only if at least two
  independent solutions agree it is valid, and a verdict needs a solid body of agreed tests (otherwise "inconclusive");
  a wrong first answer is replaced by an alternative that passes them. If the request is too vague to test it says
  "underspecified" instead of guessing. Limits: this checks consistency with the request *as the model understood it*,
  not your unstated intent; generated-test evidence moves memory trust at half strength (`autoverify.evidence_weight`),
  and you can dispute an auto-verified answer afterwards, which retires its note. It costs about 3 extra model calls per question.
- **Strategy evolution** (`pilot/strategy.py`, Evolution tab). Three ways of solving (plan first, direct, write-then-review)
  are tracked per problem type (nearest-prototype matching with the memory embedder, no extra LLM call). It explores while
  *practising* (tasks with real tests), and live questions use the champion. A strategy is promoted only when it has enough
  attempts and the lower bound of its success rate reaches the default's; the champion is recomputed from the evidence each
  time, so a promoted strategy that starts failing is rolled back automatically. This is a mechanism, not a measured
  improvement: with a ~90% baseline and small samples per type, it has not been shown to raise accuracy.
- **Memory management** (Memory tab). Each note records its evidence (times relied on in a correct / wrong answer), who verified
  it (tests, you, generated tests, planted, merged, imported), when it was last used and its version. You can search, filter,
  pin (protected from forgetting and merging), delete, export and import notes. Imported notes restart at the initial trust
  level and must earn more, so an import file cannot smuggle in a highly trusted wrong note; an import from a different
  embedder is rejected. Every retrieval explains itself (rank, similarity, trust, score) and reports how many notes were
  skipped for low trust.

**Run the demo app:**
```bash
# backend (from repo root)
pilot/.venv/bin/uvicorn app.backend.main:app --port 8000

# frontend (separate terminal)
cd app/frontend && npm install && npm run dev
```

## Roadmap

| Status | Item |
|---|---|
| Done | Real embeddings + central config; coding domain with sandboxed actor and test checker; planner, reflection, curriculum, persistence; unified experiment harness with paired statistics; 4-page app plus Assistant page |
| Done | Evaluations: code poison baselines (3 seeds), math poison (3 seeds), ablation on math (5 seeds) and code (3 seeds) |
| Done | Held-out agent-component comparison (planner, reflection, curriculum): no significant effect; planner is a lead only |
| In progress | Optional larger held-out run (~60 tasks/seed) to confirm or reject the planner effect |
| Remaining | Paper draft, correcting the stale "forgetting > trust" claim in the review slides, demo video, a skill library (optional) |

---

> Everything from here on (Sections 1-15) is the full target design — Part A (ATMC) and Part B (the rest of the self-evolving agent) combined. Section headings below aren't separately marked as "done" vs. "not done" again; refer back to the Implementation Status above for that breakdown.

# Adaptive Memory Prioritization for Self-Evolving Agentic AI

> **Adaptive Memory Prioritization using Dynamic Reflection and Trust Scoring**

A research-oriented memory architecture for agentic AI systems that dynamically evaluates, prioritizes, consolidates, and forgets memories based on **trust, importance, relevance, and temporal decay**.

The methodology is designed to address two major limitations of existing memory-augmented agents:

* **Hallucination propagation** — incorrect reflections or experiences becoming persistent knowledge.
* **Context explosion** — uncontrolled memory accumulation increasing retrieval cost and LLM context size.

---

# 1. System Architecture

## Dual-Loop Agentic Memory Pipeline

The proposed architecture separates **task execution** from **memory governance**.

```text
                         ┌─────────────────────────┐
                         │       USER QUERY        │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │     PLANNER AGENT       │
                         └────────────┬────────────┘
                                      │
                 ┌────────────────────┼────────────────────┐
                 │                    │                    │
                 ▼                    ▼                    ▼
        ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
        │ REASONING     │   │ RETRIEVAL      │   │ CHECKER        │
        │ AGENT         │   │ AGENT          │   │ AGENT          │
        └───────┬────────┘   └───────┬────────┘   └───────┬────────┘
                │                    │                    │
                └────────────────────┼────────────────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │   ADAPTIVE MEMORY MANAGER      │
                    ├────────────────────────────────┤
                    │ • Trust Evaluator              │
                    │ • Importance Scorer            │
                    │ • Decay & Forgetting Engine    │
                    │ • Memory Compression           │
                    └───────────────┬────────────────┘
                                    │
                                    ▼
                    ┌────────────────────────────────┐
                    │     CONSOLIDATED MEMORY BANK   │
                    └────────────────────────────────┘
```

---

## 1.1 Inference Loop — Execution

The **Inference Loop** is responsible for solving the current task.

### Step 1 — Query

The user provides a task to the agent.

### Step 2 — Planning

The Planner Agent decomposes the task into executable subtasks.

### Step 3 — Retrieval

The Retrieval Agent searches the vector memory store for relevant historical experiences.

Each memory is associated with dynamic metadata:

```text
Memory
├── Embedding
├── Trust Score (T)
├── Importance Score (I)
├── Decay Factor (F)
├── Query Hit Count
└── Timestamp
```

### Step 4 — Reasoning

The Reasoning Agent attempts to solve the task using:

```text
Current Query
      +
Retrieved Memories
      +
Task Context
      ↓
Reasoning Agent
      ↓
Candidate Solution
```

---

# 1.2 Governance Loop — Memory Maintenance

The **Governance Loop** operates independently from the main reasoning context.

The Checker Agent evaluates the generated solution using an appropriate verification mechanism:

* Ground-truth comparison
* Unit tests
* Code execution
* Symbolic verification
* Benchmark evaluation
* External validation

The resulting feedback updates the memory's:

* **Trust (T)**
* **Importance (I)**
* **Decay / Forgetting (F)**

The Adaptive Memory Manager then decides whether the memory should be:

```text
             Generated Reflection
                      │
                      ▼
              ┌───────────────┐
              │ Staging Buffer│
              └───────┬───────┘
                      │
                 Verification
                      │
          ┌───────────┴───────────┐
          │                       │
        Valid                  Invalid
          │                       │
          ▼                       ▼
 Consolidated Bank            Discard /
                              Low Trust
```

This prevents unverified reflections from immediately becoming persistent knowledge.

---

# 2. Mathematical Formalization

The Adaptive Memory Manager is governed by explicit mathematical functions rather than relying solely on heuristic prompt instructions.

---

## 2.1 Trust Score Update

Trust represents the reliability of a memory based on downstream task performance.

The trust score is updated as:

$$
T_{t+1}
=======

\min
\left(
1,
\max
\left(
0,
T_t + \eta R_{\text{checker}}
-----------------------------

(1-R_{\text{checker}})\delta
\right)
\right)
$$

where:

* $T_t$ = current trust score
* $T_{t+1}$ = updated trust score
* $R_{\text{checker}}$ = checker feedback
* $\eta$ = positive reward increment
* $\delta$ = penalty multiplier
* $T \in [0,1]$

The system applies a stronger penalty to incorrect memories:

$$
\delta > \eta
$$

This enables incorrect or **toxic reflections** to lose trust more rapidly than correct memories gain trust.

### Checker Signal

A binary checker can be represented as:

$$
R_{\text{checker}} \in {-1,+1}
$$

where:

```text
+1 → Verified / successful
-1 → Failed / incorrect
```

---

# 2.2 Importance Score

Importance estimates how useful a memory is for the current query.

The proposed formulation combines semantic relevance with historical retrieval frequency:

$$
I(m)
====

\cos(e_q,e_m)
\cdot
\left(1+\log(1+\text{hits}(m))\right)
$$

where:

* $e_q$ = embedding of the current query
* $e_m$ = embedding of memory $m$
* $\cos(e_q,e_m)$ = semantic similarity
* $\text{hits}(m)$ = number of previous successful retrievals of memory $m$

The logarithmic term prevents frequently accessed memories from dominating the ranking indefinitely.

---

# 2.3 Trust-Aware Adaptive Decay

Instead of applying a fixed temporal decay, the proposed architecture dynamically adjusts the decay rate according to memory trust and importance.

The forgetting factor is:

$$
F(t)
====

e^{-\lambda(T,I)\Delta t}
$$

where:

* $F(t)$ = memory retention factor
* $\Delta t$ = elapsed time
* $\lambda(T,I)$ = adaptive decay rate

The decay rate is defined as:

$$
\lambda(T,I)
============

\frac{\lambda_0}
{1+\alpha T+\beta I}
$$

where:

* $\lambda_0$ = base decay rate
* $\alpha$ = trust contribution coefficient
* $\beta$ = importance contribution coefficient
* $T$ = trust score
* $I$ = importance score

Therefore:

```text
High Trust + High Importance
            ↓
       Low Decay Rate
            ↓
      Long-Term Memory


Low Trust + Low Importance
            ↓
      High Decay Rate
            ↓
       Fast Forgetting
```

This allows the memory system to preserve valuable knowledge while aggressively removing unreliable or irrelevant information.

---

# 2.4 Composite Retrieval Score

The final retrieval score combines trust, importance, and temporal retention:

$$
S_{\text{composite}}
====================

w_1T
+
w_2I
+
w_3F(t)
$$

where:

* $w_1$ = trust weight
* $w_2$ = importance weight
* $w_3$ = decay weight

The weights satisfy:

$$
w_1+w_2+w_3=1
$$

### Trust-Based Filtering

Before top-$k$ retrieval, memories below a minimum trust threshold can be filtered:

$$
T < T_{\min}
\Rightarrow
\text{Memory Excluded}
$$

For example:

```text
Tmin = 0.30
```

This prevents low-confidence memories from entering the reasoning context.

---

# 3. Memory Lifecycle

Each memory follows a controlled lifecycle:

```text
                ┌─────────────────────┐
                │ Memory Generated    │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   Staging Buffer    │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │     Verification    │
                └──────────┬──────────┘
                           │
                    ┌──────┴──────┐
                    │             │
                 Success        Failure
                    │             │
                    ▼             ▼
              Increase T      Decrease T
                    │             │
                    └──────┬──────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Importance Update   │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Adaptive Decay      │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Prune / Consolidate │
                └─────────────────────┘
```

---

# 4. Experimental Methodology

To produce reliable empirical results, the evaluation protocol explicitly addresses common experimental failure modes in agentic AI research.

## 4.1 Data Leakage and Overfitting

The offline memory-warming and online evaluation datasets must remain strictly separated.

Example:

```text
Memory Warming
      │
      ▼
GSM8K Training Set
      │
      ▼
Memory Initialization
```

while:

```text
Online Evaluation
      │
      ▼
GSM8K Test Set
```

The test set must not be used for memory warming, reflection generation, parameter tuning, or threshold selection.

---

## 4.2 Prompt Instability and Non-Determinism

To reduce variability:

* Use temperature = `0.0` for deterministic reasoning where supported.
* Fix random seeds.
* Run experiments across multiple seeds.
* Report:

$$
\text{Mean} \pm \text{Standard Deviation}
$$

A recommended configuration is:

```text
Seeds = {42, 123, 456}
```

---

## 4.3 Vector Store Scalability

Dynamic memory metadata should be stored alongside vector embeddings.

Example:

```json
{
  "trust": 0.87,
  "importance": 0.72,
  "decay": 0.91,
  "hits": 12,
  "timestamp": "..."
}
```

Filtering should occur through vector-database metadata operations instead of repeatedly re-indexing embeddings.

This reduces unnecessary computational overhead during memory maintenance.

---

# 5. Hallucination Propagation Control

A major design principle is the **Hard Checker Gate**.

Unverified reflections are not immediately promoted to the consolidated memory bank.

```text
                    Agent Reflection
                           │
                           ▼
                  ┌─────────────────┐
                  │ Staging Buffer   │
                  └────────┬────────┘
                           │
                           ▼
                    Checker Agent
                           │
              ┌────────────┴────────────┐
              │                         │
          Verified                  Unverified
              │                         │
              ▼                         ▼
     Consolidated Memory          Remain Isolated
              │
              ▼
       Available for Retrieval
```

This provides an explicit barrier against **hallucination propagation**.

---

# 6. Benchmark Strategy

The evaluation should measure the system across four major dimensions.

---

## Track 1 — Reasoning Accuracy

### Pass@1

Measure whether the first generated solution successfully solves the task.

Recommended benchmarks include:

* GSM8K
* HumanEval
* MBPP

Evaluation should be performed over sequential tasks to observe how the memory system evolves.

Example:

```text
Task 1 → Task 2 → Task 3 → ... → Task 500
```

This allows comparison between:

```text
Initial Performance
        vs.
Long-Horizon Performance
```

---

## Task Failure Recovery Rate

Measure how effectively the agent recovers after an incorrect attempt.

A useful metric is:

$$
FRR =
\frac{\text{Failures Corrected}}
{\text{Total Failures}}
$$

The evaluation should additionally measure the number of interactions required to recover.

```text
Failure
  ↓
Reflection
  ↓
Memory Update
  ↓
Next Attempt
  ↓
Success
```

This directly evaluates whether trust-aware memory helps the agent learn from failure without permanently storing incorrect reasoning.

---

# 7. Memory Efficiency and Scalability

## 7.1 Token Cost

Track total LLM tokens consumed per task as the number of accumulated interactions increases.

Plot:

```text
Tokens / Query
      vs.
Number of Tasks
```

The desired behavior is a controlled token-growth curve despite increasing historical experience.

---

## 7.2 Retrieval Latency

Measure retrieval latency in milliseconds as memory size increases.

Plot:

```text
Retrieval Latency (ms)
          vs.
Number of Stored Memories
```

The goal is to demonstrate that metadata filtering and vector indexing maintain practical retrieval performance as the memory bank grows.

> Complexity claims should be supported by actual implementation and profiling rather than assumed from the use of a vector database.

---

## 7.3 Memory Footprint

Track the number of stored memory vectors over time.

```text
Memory Count
     vs.
Number of Tasks
```

A successful adaptive forgetting mechanism should prevent uncontrolled memory growth.

Expected behavior:

```text
Without Forgetting
      ↗
     ↗
    ↗
   ↗
  ↗

Adaptive Forgetting
      ───────────
     /           \
    /             \
   /               ───
```

The objective is to demonstrate that the memory store eventually stabilizes rather than increasing indefinitely.

---

# 8. Critical Ablation Study

Ablation experiments are essential for demonstrating that each component of the proposed architecture contributes to performance.

| Configuration               | Trust (T) | Importance (I) | Adaptive Forgetting (F) | Compression |
| --------------------------- | --------: | -------------: | ----------------------: | ----------: |
| **Full System**             |         ✓ |              ✓ |                       ✓ |           ✓ |
| **w/o Trust Scoring**       |         ✗ |              ✓ |                       ✓ |           ✓ |
| **w/o Adaptive Forgetting** |         ✓ |              ✓ |                       ✗ |           ✓ |
| **w/o Memory Compression**  |         ✓ |              ✓ |                       ✓ |           ✗ |

---

## 8.1 Full System

The complete proposed architecture:

```text
Trust
  +
Importance
  +
Adaptive Forgetting
  +
Memory Compression
```

This serves as the primary method.

---

## 8.2 Without Trust Scoring

Remove the trust-based correctness mechanism.

```text
Importance
    +
Decay
    ↓
Memory Retrieval
```

This tests whether explicit correctness-aware trust scoring contributes to preventing bad memories from persisting.

---

## 8.3 Without Adaptive Forgetting

Disable the forgetting mechanism.

```text
Memory
  ↓
Accumulation
  ↓
Accumulation
  ↓
Accumulation
```

This evaluates the impact of uncontrolled memory growth on:

* Retrieval quality
* Token consumption
* Latency
* Memory footprint

---

## 8.4 Without Memory Compression

Store raw verbal reflections without compression or consolidation.

This provides a comparison against memory systems that retain increasingly large textual experiences.

The experiment evaluates whether compression improves:

* Token efficiency
* Retrieval quality
* Storage efficiency
* Long-horizon performance

---

# 9. Recommended Evaluation Matrix

The final experiments should compare the proposed method against relevant memory-agent baselines.

| System              | Accuracy | Recovery | Tokens/Query | Retrieval Latency | Memory Size |
| ------------------- | -------: | -------: | -----------: | ----------------: | ----------: |
| No Memory           |        — |        — |            — |                 — |           — |
| Static Memory       |        — |        — |            — |                 — |           — |
| Reflexion-style     |        — |        — |            — |                 — |           — |
| SAGE-style          |        — |        — |            — |                 — |           — |
| A-MEM               |        — |        — |            — |                 — |           — |
| **Proposed Method** |        — |        — |            — |                 — |           — |

All methods should be evaluated under equivalent:

* LLM models
* Prompt budgets
* Number of trials
* Dataset splits
* Retrieval budgets
* Temperature settings
* Evaluation procedures

---

# 10. Experimental Reproducibility

Each experiment should record:

```text
Experiment Configuration
├── Model
├── Model Version
├── Dataset
├── Dataset Split
├── Random Seed
├── Temperature
├── Top-k Retrieval
├── Trust Threshold
├── Decay Parameters
├── Importance Weights
├── Composite Score Weights
└── Number of Tasks
```

A complete experiment should be reproducible from a configuration file.

Example:

```yaml
model: <model-name>

temperature: 0.0

retrieval:
  top_k: 5

trust:
  initial: 0.5
  minimum: 0.3
  reward: 0.05
  penalty: 0.15

decay:
  lambda_0: 0.1
  alpha: 1.0
  beta: 1.0

evaluation:
  seeds:
    - 42
    - 123
    - 456
```

---

# 11. Research Hypotheses

The experimental evaluation should test the following hypotheses.

### H1 — Trust improves reliability

> Explicit trust scoring reduces the persistence and retrieval of incorrect memories.

### H2 — Adaptive forgetting improves scalability

> Trust- and importance-aware decay prevents uncontrolled memory growth while preserving useful long-term knowledge.

### H3 — Memory compression improves efficiency

> Consolidating raw reflections reduces token consumption and storage requirements without significantly reducing reasoning performance.

### H4 — Adaptive memory improves long-horizon learning

> The proposed memory mechanism improves task performance over sequential interactions compared with static or unregulated memory systems.

### H5 — The complete system provides the best trade-off

> Combining Trust, Importance, Adaptive Forgetting, and Memory Compression provides a superior accuracy–efficiency trade-off compared with individual components.

---

# 12. Expected Contributions

The proposed research aims to contribute:

1. **A dual-loop architecture** separating task execution from memory governance.
2. **A mathematically defined trust mechanism** for evaluating memory reliability.
3. **A query-aware importance mechanism** combining semantic relevance and historical utility.
4. **Trust-aware adaptive forgetting** for dynamic long-term memory management.
5. **A hard verification gate** preventing unverified reflections from immediately becoming persistent memories.
6. **A unified retrieval score** combining trust, importance, and temporal retention.
7. **A rigorous evaluation framework** covering accuracy, recovery, token efficiency, latency, and memory footprint.
8. **A comprehensive ablation study** isolating the contribution of each proposed component.

---

# 13. End-to-End Methodology

The complete system can be summarized as:

```text
                         USER QUERY
                              │
                              ▼
                        PLANNER AGENT
                              │
                              ▼
                  ┌───────────────────────┐
                  │   MEMORY RETRIEVAL    │
                  └───────────┬───────────┘
                              │
                    Trust Filtering
                              │
                              ▼
                     Composite Ranking
                              │
                              ▼
                     Top-k Memories
                              │
                              ▼
                     REASONING AGENT
                              │
                              ▼
                        CANDIDATE
                         SOLUTION
                              │
                              ▼
                       CHECKER AGENT
                              │
                     ┌────────┴────────┐
                     │                 │
                  SUCCESS            FAILURE
                     │                 │
                     ▼                 ▼
                 Trust ↑            Trust ↓
                     │                 │
                     └────────┬────────┘
                              │
                              ▼
                  ADAPTIVE MEMORY MANAGER
                              │
                 ┌────────────┼────────────┐
                 │            │            │
                 ▼            ▼            ▼
              Trust      Importance     Decay
                 │            │            │
                 └────────────┼────────────┘
                              │
                              ▼
                     MEMORY CONSOLIDATION
                              │
                              ▼
                     PRUNING / COMPRESSION
                              │
                              ▼
                  CONSOLIDATED MEMORY BANK
                              │
                              └──────► Next Task
```

---

# 14. Repository Structure

## 14.1 Current Structure

What actually exists in this repository right now:

```text
.
├── README.md
├── requirements.txt
├── configs/default.yaml          # every tunable: trust, decay, weights, compression, model, embedder, seeds
│
├── pilot/                        # engine + experiments
│   ├── memory.py                 # Section 2: trust/importance/decay/composite math
│   ├── stores.py                 # StaticMemoryStore, AdaptiveTrustStore (ATMC, 4 toggleable pillars)
│   ├── embeddings.py             # text -> vector: fastembed (default) | gemini | tfidf
│   ├── config.py, factory.py     # config loading/validation; builds stores from config
│   ├── tasks.py                  # task sampling and poison construction (shared)
│   ├── agent.py, planner.py, reflection.py, curriculum.py   # the self-evolving loop
│   ├── sandbox.py                # isolated execution of generated code
│   ├── domains/                  # gsm8k (math) and coding (HumanEval/MBPP)
│   ├── report.py                 # tables + paired statistics from saved runs
│   ├── llm.py                    # Reasoning Agent (Gemini) + deterministic Checker
│   ├── experiment.py             # all experiments: poison, ablation, baselines, agent (Section 8)
│   ├── data/                     # gsm8k_test, HumanEval, mbpp
│   └── results/                  # experiment logs (tfidf_baseline/ = results under the original TF-IDF embedder)
│
├── tests/                        # 148 pytest tests (scoring math, stores, sandbox, domains, agent, API)
│
└── app/                          # full-stack demo
    ├── backend/main.py, db.py    # FastAPI (assistant, memory, eval endpoints); SQLite persistence
    └── frontend/                 # React (Vite): chat Assistant, Benchmark lab, Memory, Results, Architecture
```

## 14.2 Target Structure (by end of Month 3)

The fuller structure this project is converging toward, as the roadmap above closes each gap:

```text
adaptive-memory-agent/
│
├── README.md
│
├── agents/
│   ├── planner.py
│   ├── reasoning.py
│   ├── retrieval.py
│   └── checker.py
│
├── memory/
│   ├── manager.py
│   ├── trust.py
│   ├── importance.py
│   ├── decay.py
│   ├── compression.py
│   └── consolidation.py
│
├── retrieval/
│   ├── vector_store.py
│   ├── ranking.py
│   └── filters.py
│
├── benchmarks/
│   ├── gsm8k/
│   ├── humaneval/
│   └── mbpp/
│
├── experiments/
│   ├── full_system/
│   ├── ablations/
│   └── baselines/
│
├── evaluation/
│   ├── accuracy.py
│   ├── recovery.py
│   ├── latency.py
│   ├── token_cost.py
│   └── memory_footprint.py
│
├── configs/
│   ├── default.yaml
│   └── ablations.yaml
│
├── results/
│   ├── tables/
│   ├── figures/
│   └── logs/
│
├── requirements.txt
└── LICENSE
```

---

# 15. Research Standard

The objective of this methodology is not merely to demonstrate that the proposed memory mechanism works on a small number of examples.

The evaluation should establish that the system:

```text
             Reliable
                │
                ▼
        ┌───────────────┐
        │ Adaptive      │
        │ Memory        │
        └───────────────┘
          │     │     │
          ▼     ▼     ▼
       Accuracy Efficiency Scalability
          │     │     │
          └─────┼─────┘
                ▼
       Long-Horizon Agentic
            Reasoning
```

The central research claim is therefore:

> **An agentic AI system can improve long-horizon reasoning by treating memory as a dynamically governed resource rather than a static collection of retrieved experiences.**

The proposed architecture continuously determines **what should be remembered, how strongly it should be trusted, how relevant it remains, and when it should be forgotten**.
