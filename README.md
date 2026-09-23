# Adaptive-Trust-Based-Memory-Consolidation-for-Self-Evolving-Multi-Agent-LLM-Systems
ATMC (Adaptive Trust-based Memory Consolidation) is an advanced multi-agent architecture designed for lifelong self-evolving LLMs. By replacing unweighted memory storage with dynamic Trust Scoring ($T$), Importance Prioritization ($I$), and Adaptive Forgetting ($F$), the system eliminates noisy or unverified reflections, prevents hallucination propagation, and maintains $O(1)$ memory retrieval efficiency across sequential reasoning tasks (GSM8K, HumanEval).

> **Naming note:** this project was originally developed under the working name "TrustMem-Agent." It has been renamed to ATMC to avoid collision with an unrelated prior-art paper, *TRUSTMEM: Learning Trustworthy Memory Consolidation for LLM Agents with Long-Term Memory* (arXiv 2606.25161), which targets a different problem (RL-trained verification of memory-editing operations) but shares the name and the general subfield.

---

## Implementation Status

Everything below this point (Sections 1-15) is the full research design. This section states, plainly, how much of it currently exists in code.

**Validated — a real, reproducible experimental result:**
Trust-gated memory retrieval reduces exposure to deliberately planted incorrect memories by **8.0× (± 0.54 across 3 seeds)** versus a static "remember everything" baseline, with no accuracy cost, on GSM8K. This is direct support for Hypothesis H1 (Section 11). Full logs: [`pilot/results/`](pilot/results/).

**Built and working:**
- Core scoring math — Trust, Importance, Decay, Composite Score (Section 2) → [`pilot/memory.py`](pilot/memory.py)
- Memory store with the hard verification gate, trust-gated composite retrieval, and citation-conditioned trust updates (i.e. a memory's trust only moves if the model says it actually relied on it — closes a credit-assignment gap found during testing) → [`pilot/stores.py`](pilot/stores.py)
- A deterministic Checker Agent (ground-truth comparison, one of the mechanisms Section 1.2 allows) and a Reasoning Agent (Gemini) → [`pilot/llm.py`](pilot/llm.py)
- Pilot scripts reproducing the result above across multiple seeds → [`pilot/run_pilot.py`](pilot/run_pilot.py), [`pilot/run_poison_pilot.py`](pilot/run_poison_pilot.py)
- A full-stack demo app (FastAPI backend + React frontend) running this mechanism live, interactively → [`app/`](app/)

**Not yet built** (tracked in the Roadmap below):
- A separate Planner Agent — the Reasoning Agent currently solves directly, so the 4-agent architecture in Section 1 is not yet fully realized as distinct agents
- Memory Compression — the fourth pillar (Section 8.4) has no implementation yet
- Benchmarks beyond GSM8K (HumanEval/MBPP planned)
- The full ablation matrix (Section 8) — only 2 of 4 conditions are currently comparable
- Baselines beyond static memory (Reflexion-style, SAGE-style, A-MEM — Section 9)
- Persistent storage, tool/code execution, and autonomous curriculum generation — needed for the "self-evolving agent" framing in this repository's title, which is distinct from ATMC (the memory-consolidation mechanism) itself. **ATMC is the validated core memory subsystem a self-evolving agent needs — it is not, on its own, a complete self-evolving agent.**

## Getting Started

**Run the pilot experiments:**
```bash
cd pilot
python3 -m venv .venv && source .venv/bin/activate
pip install google-genai python-dotenv scikit-learn numpy
echo "GEMINI_API_KEY=your_key_here" > ../.env
python3 run_poison_pilot.py
```

**Run the demo app:**
```bash
# backend (from repo root)
pilot/.venv/bin/uvicorn app.backend.main:app --port 8000

# frontend (separate terminal)
cd app/frontend && npm install && npm run dev
```

## Roadmap

Three-month plan from the validated core (ATMC) to the full self-evolving agent described in this document.

| Phase | Focus | Key additions |
|---|---|---|
| **Month 1** | Harden the core | Full ablation matrix, more seeds, self-reflection generation, a real Planner Agent |
| **Month 2** | Give the agent something to act in | Second benchmark + code-execution Checker, sandboxed tool execution, persistent storage |
| **Month 3** | Autonomy | Self-directed curriculum, a reusable skill library (closes the Compression gap), final full evaluation, paper write-up |

---

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
├── .gitignore
│
├── pilot/                        # engine + experiments
│   ├── memory.py                 # Section 2: trust/importance/decay/composite math
│   ├── stores.py                 # StaticMemoryStore, AdaptiveTrustStore (ATMC)
│   ├── llm.py                    # Reasoning Agent (Gemini) + deterministic Checker
│   ├── run_pilot.py              # organic-mistake pilot (ceiling-effect finding)
│   ├── run_poison_pilot.py       # poison-memory pilot (the validated 8.0x result)
│   ├── data/gsm8k_test.jsonl     # benchmark data
│   └── results/                  # experiment logs, by seed, incl. pre/post-fix history
│
└── app/                           # full-stack demo
    ├── backend/main.py            # FastAPI wrapper around pilot/ engine
    └── frontend/                  # React (Vite) — live static-vs-ATMC comparison UI
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
