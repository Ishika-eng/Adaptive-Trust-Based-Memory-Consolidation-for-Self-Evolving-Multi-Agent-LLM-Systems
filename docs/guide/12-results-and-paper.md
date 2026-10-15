# 12 · Running the experiments, the results, and the paper

## 12.1 What was run, and the exact commands

All results live in `pilot/results/eval/` (one JSON per run). Each command below is resumable; run **one at a time**.

| Experiment | Command | Runs | Approx. model calls* |
|---|---|---|---|
| Poison, math | `experiment.py poison --domain math --seeds 42,123,456` | 6 | ~250 |
| Baselines + poison, code | `experiment.py baselines_poison --domain code --seeds 42,123,456` | 18 | ~700 |
| Ablation, math (5 seeds) | `experiment.py ablation --domain math --seeds 42,123,456,789,1011` | 20 | ~900 |
| Ablation, code | `experiment.py ablation --domain code --seeds 42,123,456` | 12 | ~540 |
| Agent components, code | `experiment.py agent --domain code --seeds 42,123,456` | 18 | ~1,600 (cache cut real calls to about 400 on the first pass) |

\* The script prints its own estimate first. With a free key (about 500 calls/day) these take several days; with billing enabled, a few hours. After running: `pilot/.venv/bin/python pilot/report.py <experiment> --domain <d> --ref <reference condition>`.

## 12.2 The results (from the saved files)

**Setup:** 40 tasks per run, top-3 retrieval, planted wrong note for every 3rd task (13–14 per run), seeds 42/123/456 unless noted, `gemini-3.5-flash-lite` at temperature 0, `BAAI/bge-small-en-v1.5` embeddings.

### Poison resistance (planted notes shown to the model per run)

| Condition | Pass rate | Shown (total / after warm-up) | Relied on | Tokens/task |
|---|---|---|---|---|
| GSM8K static | 98.3 ± 1.4 | 60.0 / 51.0 | 5.0 | 614 |
| GSM8K ATMC | 100.0 ± 0.0 | **6.0 / 0** | 0 | 828 |
| Code no memory | 89.2 ± 5.8 | 0 | 0 | 104 |
| Code static | 89.2 ± 5.8 | 61.3 / 52.7 | 9.3 | 638 |
| Code Reflexion-style | 89.2 ± 6.3 | 53.3 / 44.3 | 0.3 | 736 |
| Code ATMC | 88.3 ± 7.2 | **6.0 / 0** | 0 | 624 |
| Code ATMC + relevance floor 0.75 | 88.3 ± 5.2 | 8.3 / 7.7 | 1.3 | 342 |

Reading it: ATMC cuts exposure about **tenfold**, and to zero after the 3-task warm-up (the 6 remaining are a *structural* cold-start floor: in the first three tasks the planted notes are the only notes). **Accuracy is not separated** (every paired McNemar p = 1.0 on code; GSM8K +1.7 points, CI [0.0, +4.2], p = 0.50). The relevance floor halves tokens but lets planted notes through again and does not recover accuracy, so it is off by default.

### Ablation (planted notes in context after warm-up, mean ± sd)

| Variant | GSM8K (5 seeds) | Code (3 seeds) | Notes kept (GSM8K) |
|---|---|---|---|
| Full ATMC | 0 | 0 | 42 |
| w/o Trust | **111 ± 0** | **111 ± 0** | 42 |
| w/o Forgetting | 14.8 ± 20.3 | 26.0 ± 19.1 | 42 |
| w/o Merging | 0 | 0 | 54 |

Trust is the decisive pillar (without it every note has equal weight and only similarity ranks them). Forgetting has a smaller, **inconsistent** effect (at our run lengths it only lowers ranks; deletion needs ~76 steps). Merging shrinks the store 22% (54 → 42) at no cost in exposure. Pass rates are indistinguishable.

### Agent components (code, 20 held-out tasks per seed, 3 seeds)

| Condition | Held-out pass | Δ vs ATMC [95% CI] | McNemar p | Tokens/task |
|---|---|---|---|---|
| ATMC alone | 88.3 ± 2.9 | – | – | 1164 |
| + Planner | 95.0 ± 5.0 | +6.7 [+1.7, +13.3] | 0.125 | 1526 |
| + Reflection | 88.3 ± 2.9 | +0.0 | 1.000 | 1193 |
| + Planner + Reflection | 95.0 ± 5.0 | +6.7 [+1.7, +13.3] | 0.125 | 1561 |
| Random task choice | 90.0 ± 0.0 | +1.7 [+0.0, +5.0] | 1.000 | 1178 |
| Weakness-targeted choice | 90.0 ± 0.0 | +1.7 [−3.3, +6.7] | 1.000 | 900 |

**No component shows a significant benefit.** The planner is a lead (4 tasks better, 0 worse, but only 4 changed), costs ~31% more tokens, and would need a larger held-out set to confirm.

## 12.3 What the results do and do not say

- **Supported:** trust-gated memory keeps planted wrong notes out of the model's context; trust is the pillar responsible.
- **Not shown:** higher accuracy. The base model already solves ~90% of these tasks, so benefit is invisible at this ceiling. In the static code runs the model cited a planted note on 28 steps and failed on only one.
- **A threat-model caveat:** planted notes were recorded as *failed attempts* (labelled `INCORRECT`, stored by ATMC at trust 0.35). A stronger test inserts wrong notes recorded as **correct**. We did not run it (our attempt was blocked by quota). That experiment, or a weaker model/harder tasks where memory errors change accuracy, is the most valuable next step.
- Small samples (40 tasks × 3–5 seeds), one model family, Python only.

## 12.4 The paper pipeline (`paper/`)

The paper is generated from the result files, so numbers cannot drift from data.

```bash
pilot/.venv/bin/pip install matplotlib python-docx pypdf     # plus LibreOffice for the PDF
pilot/.venv/bin/python paper/figures.py                       # writes paper/figures/*.png
pilot/.venv/bin/python paper/build_paper.py                   # writes paper/ATMC_Paper.docx and .pdf
```

### `figures.py`

Six matplotlib figures in a serif font, drawn **from the JSON results**: (1) workflow boxes, (2) architecture boxes, (3) poison exposure bars with seed dots, (4) ablation bars, (5) trust over verified failures (**computed by calling `memory.update_trust`**, so it is the real rule, not a drawing), (6) held-out pass rate by component.

### `build_paper.py`

Builds an IEEE-style two-column Word file with `python-docx`:

- A **title section** with one column (title, a 2×2 borderless author table), then a **continuous section break** into a **two-column** section — columns are set by writing a `<w:cols w:num="2">` element into the section properties.
- Page: US Letter, 0.75" margins, Times New Roman 10 pt; headings in small caps; the IEEE footer line only on the first page (`different_first_page_header_footer`).
- **Tables** use a fixed layout with explicit column widths summing to less than the column width (otherwise they spill over the text), captions above; **figures** are captioned below.
- **Equations** are paragraphs with a centre tab stop and a right tab stop for the number, using italic and sub/superscript runs.
- Numbers in the text are f-strings computed from `report.build_report` and the raw JSON; a final script compares quoted numbers against the raw files.
- The PDF is produced with `soffice --headless --convert-to pdf`.
- **Author names are placeholders** to fill in.

References were verified one by one against search results (titles, authors, arXiv numbers); recheck the whole list before submitting.

## Checkpoint

You can regenerate `paper/ATMC_Paper.pdf` from the committed results and it matches the tables above.
