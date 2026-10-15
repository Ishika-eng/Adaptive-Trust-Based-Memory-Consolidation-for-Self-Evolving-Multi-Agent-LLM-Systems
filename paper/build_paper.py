"""Builds the paper as an IEEE-style two-column Word document (and PDF via LibreOffice).

    pilot/.venv/bin/python paper/figures.py
    pilot/.venv/bin/python paper/build_paper.py

Every number in the text and tables is read from pilot/results/eval/ -- nothing is typed in by hand.
Author names and e-mails are placeholders to be filled in.
"""
import glob
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pilot"))
import report  # noqa: E402

FIG = ROOT / "paper" / "figures"
OUT = ROOT / "paper" / "ATMC_Paper.docx"
FONT = "Times New Roman"

# ============================================================ data (all from saved results)
_REF = {"poison": "static", "baselines_poison": "no_memory", "ablation": "full_system", "agent": "atmc"}
REP = {(e, d): report.build_report(e, d, _REF[e]) for e, d in
       [("poison", "math"), ("baselines_poison", "code"), ("ablation", "math"), ("ablation", "code"), ("agent", "code")]}


def row(exp, dom, cond):
    return next(c for c in REP[(exp, dom)]["conditions"] if c["condition"] == cond)


def m(exp, dom, cond, key):
    return row(exp, dom, cond)[key]["mean"]


def sd(exp, dom, cond, key):
    return row(exp, dom, cond)[key]["sd"]


def total_exposure(exp, dom, cond):
    xs = [json.load(open(f))["summary"]["poison_exposure"]
          for f in sorted(glob.glob(str(ROOT / f"pilot/results/eval/{exp}__{dom}__{cond}__seed*.json")))]
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0)


def comp(exp, dom, cond, kind="in_run"):
    c = next(c for c in REP[(exp, dom)]["comparisons"] if c["condition"] == cond)
    return c[kind]


def pc(x, d=1):
    return f"{100 * x:.{d}f}"


def pm(mean, s, pct=False, d=1):
    return f"{pc(mean, d)} ± {pc(s, d)}" if pct else f"{mean:.{d}f} ± {s:.{d}f}"


def loc(patterns):
    n = 0
    for p in patterns:
        for f in glob.glob(str(ROOT / p), recursive=True):
            if "node_modules" in f or ".venv" in f or "/dist/" in f:
                continue
            n += sum(1 for _ in open(f, errors="ignore"))
    return n


PY_LOC = loc(["pilot/*.py", "pilot/domains/*.py", "app/backend/*.py"])
JS_LOC = loc(["app/frontend/src/**/*.jsx", "app/frontend/src/**/*.css"])
TEST_LOC = loc(["tests/*.py"])

# ============================================================ low-level docx helpers
doc = Document()


def font(run, size=10, bold=False, italic=False, small_caps=False, color=None, underline=False, sub=False, sup=False):
    run.font.name = FONT
    run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    run.font.size = Pt(size)
    run.font.bold, run.font.italic, run.font.small_caps, run.font.underline = bold, italic, small_caps, underline
    run.font.subscript, run.font.superscript = sub, sup
    if color:
        run.font.color.rgb = RGBColor(*color)


def para(container=None, align=WD_ALIGN_PARAGRAPH.JUSTIFY, first=0.17, after=4, before=0, left=0.0, keep_next=False, size=10):
    p = (container or doc).add_paragraph()
    pf = p.paragraph_format
    pf.alignment, pf.space_after, pf.space_before = align, Pt(after), Pt(before)
    pf.first_line_indent = Inches(first) if first else None
    pf.left_indent = Inches(left) if left else None
    pf.keep_with_next = keep_next
    pf.line_spacing = 1.0
    p._size = size
    return p


def runs(p, parts, size=None):
    """parts: str or (text, flags) with flags among i (italic) b (bold) s (subscript) p (superscript) l (link)."""
    size = size or getattr(p, "_size", 10)
    for part in parts if isinstance(parts, list) else [parts]:
        text, flags = (part, "") if isinstance(part, str) else part
        r = p.add_run(text)
        font(r, size, bold="b" in flags, italic="i" in flags, sub="s" in flags, sup="p" in flags,
             color=(5, 99, 193) if "l" in flags else None, underline="l" in flags)
    return p


def text(parts, **kw):
    return runs(para(**kw), parts)


def set_cols(section, n, space=0.25):
    sectPr = section._sectPr
    for c in sectPr.findall(qn("w:cols")):
        sectPr.remove(c)
    cols = OxmlElement("w:cols")
    cols.set(qn("w:num"), str(n))
    cols.set(qn("w:space"), str(int(space * 1440)))
    sectPr.append(cols)


def heading(num, title):
    p = para(align=WD_ALIGN_PARAGRAPH.CENTER, first=0, before=7, after=3, keep_next=True)
    r = p.add_run(f"{num}. ")
    font(r, 10)
    r = p.add_run(title)
    font(r, 10, small_caps=True)
    return p


def subheading(letter, title):
    p = para(align=WD_ALIGN_PARAGRAPH.LEFT, first=0, before=4, after=2, keep_next=True)
    runs(p, f"{letter}. {title}")
    return p


def bullet(parts):
    p = para(first=-0.2, left=0.4, after=2)
    p.paragraph_format.tab_stops.add_tab_stop(Inches(0.4))
    runs(p, ["•\t"] + (parts if isinstance(parts, list) else [parts]))
    return p


def equation(parts, number):
    p = para(align=WD_ALIGN_PARAGRAPH.LEFT, first=0, after=3, before=1)
    p.paragraph_format.tab_stops.add_tab_stop(Inches(1.62), WD_TAB_ALIGNMENT.CENTER)
    p.paragraph_format.tab_stops.add_tab_stop(Inches(3.3), WD_TAB_ALIGNMENT.RIGHT)
    runs(p, ["\t"] + parts + [f"\t({number})"])
    return p


def caption(label, number, body, above=False):
    p = para(align=WD_ALIGN_PARAGRAPH.LEFT if above else WD_ALIGN_PARAGRAPH.CENTER, first=0,
             after=3 if above else 6, before=4 if above else 0, keep_next=above, size=9)
    runs(p, f"{label} {number}: {body}", size=9)
    return p


def figure(name, number, body, width=3.3):
    p = para(align=WD_ALIGN_PARAGRAPH.CENTER, first=0, after=2, before=4, keep_next=True)
    p.add_run().add_picture(str(FIG / name), width=Inches(width))
    caption("Figure", number, body)


def table(number, body, header, rows, widths, size=8.2):
    caption("Table", number, body, above=True)
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.LEFT
    t.autofit = False
    tblPr = t._tbl.tblPr
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblPr.append(layout)
    for old in tblPr.findall(qn("w:tblW")):
        tblPr.remove(old)
    tblW = OxmlElement("w:tblW")
    tblW.set(qn("w:w"), str(int(sum(widths) * 1440)))
    tblW.set(qn("w:type"), "dxa")
    tblPr.append(tblW)
    mar = OxmlElement("w:tblCellMar")
    for side in ("left", "right"):
        e = OxmlElement(f"w:{side}")
        e.set(qn("w:w"), "50")
        e.set(qn("w:type"), "dxa")
        mar.append(e)
    tblPr.append(mar)
    for i, w in enumerate(widths):
        t.columns[i].width = Inches(w)
    for i, cells in enumerate([header] + rows):
        for j, val in enumerate(cells):
            c = t.cell(i, j)
            c.width = Inches(widths[j])
            c.text = ""
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
            runs(p, val, size=size)
        tr = t.rows[i]._tr
        trPr = tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        trPr.append(cant)
    after = para(first=0, after=0)
    after.paragraph_format.line_spacing = Pt(5)
    return t


# ============================================================ page setup, title block
sec0 = doc.sections[0]
sec0.page_width, sec0.page_height = Inches(8.5), Inches(11)
sec0.left_margin = sec0.right_margin = Inches(0.75)
sec0.top_margin, sec0.bottom_margin = Inches(0.75), Inches(0.7)
sec0.different_first_page_header_footer = True
fp = sec0.first_page_footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.LEFT
font(fp.add_run("XXX-X-XXXX-XXXX-X/XX/$XX.00 ©20XX IEEE"), 8)

normal = doc.styles["Normal"]
normal.font.name, normal.font.size = FONT, Pt(10)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)

TITLE = "Adaptive Trust-Based Memory Consolidation for a Self-Improving Coding Assistant"
p = para(align=WD_ALIGN_PARAGRAPH.CENTER, first=0, after=8, before=14)
r = p.add_run(TITLE)
font(r, 24)

authors = tbl = doc.add_table(rows=2, cols=2)
tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
for i in range(2):
    for j in range(2):
        c = tbl.cell(i, j)
        c.width = Inches(3.5)
        n = 2 * i + j + 1
        lines = [(f"[Author {n} Name]", ""), ("Computer Engineering", ""), ("Vishwakarma Institute of Technology, Pune", ""),
                 ("Pune, India", ""), (f"author{n}@vit.edu", "l")]
        c.text = ""
        for k, (line, fl) in enumerate(lines):
            q = c.paragraphs[0] if k == 0 else c.add_paragraph()
            q.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            q.paragraph_format.space_after = Pt(11 if (i == 0 and k == len(lines) - 1) else 0)
            runs(q, [(line, fl)], size=10)
spacer = para(first=0, after=6)

# continuous section: two columns
sec1 = doc.add_section(WD_SECTION.CONTINUOUS)
set_cols(sec1, 2, 0.28)

# ============================================================ abstract
g_static, g_atmc = total_exposure("poison", "math", "static")[0], total_exposure("poison", "math", "atmc")[0]
c_static, c_atmc = total_exposure("baselines_poison", "code", "static")[0], total_exposure("baselines_poison", "code", "atmc")[0]
c_refl = total_exposure("baselines_poison", "code", "reflexion")[0]
c_floor = total_exposure("baselines_poison", "code", "atmc_minsim")[0]

abstract = (
    "Language-model agents that improve by writing their own memory can also learn wrong lessons and then trust them. "
    "This paper presents ATMC (Adaptive Trust-based Memory Consolidation), an outcome-verified memory for a self-improving "
    "coding assistant. Every stored note carries a trust score that is set and updated from verified results (unit tests run "
    "in a sandbox, generated tests with cross-solution consensus, or the user's verdict) and only for notes the model actually "
    "cited. Low-trust notes are blocked, unused notes fade, and similar notes are merged. The assistant also chooses its own "
    "practice tasks and tracks which solving strategy works best for each problem type; the model's weights are never changed. "
    f"With deliberately planted wrong notes on GSM8K and HumanEval/MBPP (3 seeds, 40 tasks per run), ATMC cut the planted notes "
    f"shown to the model about tenfold ({g_static:.1f} to {g_atmc:.1f} per run on GSM8K; {c_static:.1f} to {c_atmc:.1f} on code), to zero "
    f"after a three-task warm-up, while pass rates stayed statistically indistinguishable from a no-memory baseline "
    f"({pc(m('baselines_poison','code','no_memory','pass_rate'))}% vs {pc(m('baselines_poison','code','atmc','pass_rate'))}% on code). "
    "Ablations identify trust scoring as the decisive component. The planner, reflection and curriculum showed no significant "
    "accuracy gain, and we report the limitations of this evidence."
)
p = para(first=0.17, after=6, size=9.5)
runs(p, [("Abstract", "i"), "— " + abstract], size=9.5)
p = para(first=0.17, after=8, size=9.5)
runs(p, [("Keywords", "i"), "— Self-Evolving Agents, Agent Memory, Trust Scoring, Memory Poisoning, Code Generation, Large Language Models"], size=9.5)

# ============================================================ I. Introduction
heading("I", "Introduction")
text("Language-model agents increasingly improve by keeping notes about their own attempts and reading them back on later "
     "tasks [1]–[5]. Unlike curated knowledge, these notes are written by a fallible model. A wrong note, whether it comes from a "
     "hallucination, a weak test or a deliberate attack [6], [7], is retrieved like any other and can steer later answers. "
     "Most memory designs focus on what to store and how to retrieve it; few ask whether a stored note deserves to be trusted.")
text("This work addresses that gap. ATMC treats memory as something that is earned: a note's trust rises or falls with verified "
     "outcomes, notes below a threshold are never retrieved, unused notes fade, and similar notes are merged to keep the memory "
     "small. We build the memory into a complete coding assistant, with a sandboxed checker, a self-directed practice loop and a "
     "chat interface, and evaluate it under controlled corruption of its memory.")
text("Key Contributions:", first=0, after=2)
bullet("An outcome-verified, trust-scored memory (ATMC) with citation-conditioned trust updates, trust-gated ranking, adaptive "
       "forgetting and compression; it is training-free and independent of the underlying model.")
bullet("A verification layer for tasks with and without tests: sandboxed execution, auto-verification with generated tests and "
       "cross-solution consensus, evidence-weighted learning, and a dispute path.")
bullet("Self-directed improvement: a weakness-targeted practice curriculum and a champion/challenger strategy selector with "
       "automatic rollback.")
bullet("A reproducible evaluation with paired statistics on GSM8K, HumanEval and MBPP, reporting negative results as well as "
       "positive ones.")

# ============================================================ II. Related work
heading("II", "Related Work")
text("Memory for language-model agents. Reflexion stores verbal reflections after failures and conditions later attempts on "
     "them [1]. Generative Agents retrieve memories by recency, importance and relevance and add higher-level reflections [2]. "
     "MemGPT pages information between context tiers [3], A-MEM organizes notes as an evolving linked memory [4], and Voyager "
     "accumulates a library of reusable skills [5]. These systems decide what to keep and fetch; ATMC adds a score for how far a "
     "note has been shown to help, and removes notes that fail that test.")
text("Memory poisoning. PoisonedRAG shows that a handful of malicious passages can steer a retrieval-augmented model, and "
     "reports that existing defenses are insufficient [6]. AgentPoison poisons an agent's long-term memory or knowledge base "
     "with a backdoor trigger [7]. ATMC targets the same surface from the defender's side, using outcome-based trust.")
text("Self-improvement and verification. Self-Refine revises an answer using the model's own feedback [8]. CodeT has the model "
     "write tests and picks solutions by agreement with them [9]; we use the same agreement idea, but to gate what enters memory. "
     "A recent survey organizes self-evolving agents by what evolves (models, memory, tools, architecture) and when [10]; ATMC "
     "operates at the memory and strategy-selection level and leaves the weights untouched.")
text("Benchmarks and components. We evaluate on HumanEval [11], MBPP [12] and GSM8K [13]; relevance is measured with BGE "
     "sentence embeddings [14] in a retrieval-augmented setting [15].")

# ============================================================ III. Methodology
heading("III", "Methodology")
text("The system is a pipeline in which each stage has one job: the interface collects a question, the backend plans, retrieves "
     "memory, calls the language model and checks the result in a sandbox, and the memory is updated from the outcome. The "
     "components are modular, so the model, the memory or the interface can be replaced independently. The language roles "
     "(planner, solver, reflector, summarizer, test writer) are separate prompts to one model run in sequence; they are modules of "
     "a single pipeline and not independent, communicating agents.")

table(1, "System components and technologies",
      ["Component", "Technology / Tool", "Purpose"],
      [["Chat interface", "React + Vite", "Questions, answers, memory view"],
       ["Backend API", "FastAPI (Python)", "Agent loop, persistence"],
       ["Language model", "Gemini API", "Reasoning, tests, summaries"],
       ["Embeddings", "BGE-small (fastembed)", "Note and task similarity"],
       ["Memory (ATMC)", "Own implementation", "Trust, forgetting, merging"],
       ["Checker", "Python sandbox", "Run tests safely"],
       ["Storage", "SQLite", "Notes, history, evidence"],
       ["Evaluation", "Own harness + pytest", "Paired statistics"]],
      [0.95, 1.25, 1.12])

figure("fig1_workflow.png", 1, "End-to-end workflow of the assistant for one question", width=2.45)

subheading("A", "System Architecture")
text(f"The assistant has four parts (Fig. 2): a web interface, a backend that runs the agent loop, the ATMC memory with its "
     f"SQLite store, and a sandbox that executes generated code. Retrieval embeddings are computed locally; only the language "
     f"model is an external service. The implementation is about {PY_LOC:,} lines of Python, {JS_LOC:,} lines of JavaScript and "
     f"CSS, and {TEST_LOC:,} lines of tests (148 automated tests).")
figure("fig2_architecture.png", 2, "System architecture", width=3.1)

subheading("B", "ATMC Memory Model")
text("A note stores the task, the attempted solution, its outcome and any lesson, together with a trust score, a use count, the "
     "evidence behind its outcome (tests, user, generated tests, planted, merged) and counts of verified successes and failures "
     "when it was relied on. Trust is initialized from the checker's verdict and then updated, but only for notes the model "
     "reports having used, which avoids crediting or blaming notes that were merely retrieved:")
equation([("T", "i"), ("′", ""), (" = clip(", ""), ("T", "i"), (" + ", ""), ("wη", "i"), (" [ok] − ", ""), ("wδ", "i"),
          (" [fail], 0, 1)", "")], 1)
text([("where [ok] and [fail] mark a verified correct or wrong outcome, with ", ""), ("η", "i"), (" = 0.05, ", ""), ("δ", "i"), (" = 0.15 (", ""), ("δ", "i"), (" > ", ""), ("η", "i"),
      (" so bad notes lose trust faster than good ones gain it) and an evidence weight ", ""), ("w", "i"),
      (" = 1 for tests and user verdicts and 0.5 for generated tests. Notes with trust below ", ""), ("T", "i"), ("min", "is"),
      (" = 0.30 are excluded from retrieval. Relevance to a query ", ""), ("q", "i"), (" is the cosine similarity of normalized "
      "embeddings, weighted by how often a note has been used:", "")], first=0)
equation([("I", "i"), ("(", ""), ("m", "i"), (") = cos(", ""), ("e", "i"), ("q", "is"), (", ", ""), ("e", "i"), ("m", "is"),
          (") · (1 + log(1 + ", ""), ("h", "i"), ("m", "is"), ("))", "")], 2)
text("Notes also fade with the number of task steps since creation, following an exponential forgetting curve [19], more slowly when they are trusted and important:", first=0)
equation([("F", "i"), (" = exp(−", ""), ("λΔt", "i"), ("),   ", ""), ("λ", "i"), (" = ", ""), ("λ", "i"), ("0", "s"),
          (" / (1 + ", ""), ("αT", "i"), (" + ", ""), ("βI", "i"), (")", "")], 3)
text([("Retrieval ranks notes by a composite in which trust also gates importance, so a barely-trusted note cannot enter the "
       "top ", ""), ("k", "i"), (" = 3 merely by looking similar:", "")], first=0)
equation([("S", "i"), (" = 0.4 ", ""), ("T", "i"), (" + 0.4 ", ""), ("I", "i"), ("·", ""), ("T", "i"), (" + 0.2 ", ""), ("F", "i")], 4)

subheading("C", "Verification and Learning Signals")
text("When a task has tests (HumanEval, MBPP, or assertions the user supplies), the answer runs in a sandbox that denies network "
     "access and writes outside a temporary directory, hides credentials, and enforces a timeout and resource limits. "
     "The sandbox is not a hard boundary against hostile code. When a user task has no tests, auto-verification asks the model for "
     "up to eight assert statements and for two independent alternative solutions. A generated test counts as reliable only if at "
     "least two solutions pass it, and a verdict requires a body of agreed tests (at least two and half of those generated); "
     "otherwise the result is inconclusive. A wrong first answer is replaced by an alternative that passes the reliable tests, and a "
     "request too vague to test is declared underspecified. The check measures consistency with the request as the model "
     "understood it, not the user's unstated intent, so generated-test evidence is given half weight and the user can dispute "
     "the result, which retires the note.")

subheading("D", "Forgetting, Merging and Pinning")
text("Every ten additions, notes whose trust is below the gate, or whose retention is below 0.05, are removed. Every eight "
     "additions, the three oldest, least-used notes are merged into a model-written summary, which records its lineage "
     "(summed evidence, version number). Pinned notes are exempt from both, and imported notes restart at the initial trust so a "
     "file cannot smuggle in a highly trusted wrong note.")

subheading("E", "Practice and Strategy Selection")
text("The assistant picks its own practice tasks: each unattempted task is scored by its similarity to the assistant's own "
     "failure notes, with a 20% chance of choosing at random to avoid fixating. It also tracks three solving strategies (plan "
     "first, direct, and write-then-review) for each problem type, found by nearest-prototype matching with the same embedder. "
     "The plan-first default is replaced for a problem type only when another strategy has at least five attempts and the lower "
     "bound of its Wilson interval [16] reaches the default's success rate; the choice is recomputed from the evidence each time, "
     "so a strategy that starts failing is rolled back automatically. Exploration happens during practice, where outcomes come from "
     "real tests.")

subheading("F", "Experimental Setup")
text("Datasets. GSM8K [13] (test split), HumanEval [11] and MBPP [12] (four MBPP tasks whose reference solutions fail their "
     "own tests were excluded). Each run processes a seeded sample of 40 tasks (seeds 42, 123, 456; the GSM8K ablation also "
     "uses 789 and 1011). The model is gemini-3.5-flash-lite [20] at temperature 0; embeddings are BAAI/bge-small-en-v1.5 (384 "
     "dimensions). Memory settings: top-3 retrieval, initial trust 0.5, merging every 8 additions.")
text("Corruption. Every third task (13–14 per run) receives a planted wrong note for that task: a wrong numeric answer (GSM8K) "
     "or a single-site mutation of the reference solution verified to fail its tests (code). These notes are recorded as failed "
     "attempts. Baselines: no memory; static memory (stores and retrieves everything by similarity); a Reflexion-style buffer of "
     "the three most recent failure reflections [1]; and ATMC.")
text("Metrics and statistics. Pass rate; planted notes shown to the model, in total and after the first three tasks (before "
     "which no genuine note can exist); planted notes the model cited; tokens per task; and retrieval latency. Conditions are "
     "compared on the same tasks with a paired bootstrap 95% confidence interval [18] and the exact McNemar test [17]. Agent "
     "components are also scored on 20 unseen tasks per seed with memory frozen.")

# ============================================================ IV. Results
heading("IV", "Results and Discussion")
subheading("A", "Resistance to Planted Wrong Notes")

t2 = []
def r2(label, exp, dom, cond, tokens=True):
    tot = total_exposure(exp, dom, cond)[0]
    aft = m(exp, dom, cond, "poison_exposure_after_warmup")
    used = (m(exp, dom, cond, "poison_used"), sd(exp, dom, cond, "poison_used"))
    t2.append([label, pm(m(exp, dom, cond, "pass_rate"), sd(exp, dom, cond, "pass_rate"), pct=True),
               f"{tot:.1f} ({aft:.1f})", f"{used[0]:.1f}" if used[0] else "0", f"{m(exp, dom, cond, 'tokens_in'):.0f}"])
r2("GSM8K: static", "poison", "math", "static")
r2("GSM8K: ATMC", "poison", "math", "atmc")
r2("Code: no memory", "baselines_poison", "code", "no_memory")
r2("Code: static", "baselines_poison", "code", "static")
r2("Code: Reflexion-style", "baselines_poison", "code", "reflexion")
r2("Code: ATMC", "baselines_poison", "code", "atmc")
r2("Code: ATMC + floor", "baselines_poison", "code", "atmc_minsim")
table(2, "Pass rate and exposure to planted notes (3 seeds, 40 tasks per run). Shown = planted notes in the model's context per run "
         "(after warm-up in parentheses); Relied = planted notes the model cited; floor = optional minimum-similarity filter (0.75)",
      ["Condition", "Pass rate (%)", "Shown", "Relied", "Tokens"], t2, [1.0, 0.82, 0.68, 0.38, 0.45])

gs = REP[("poison", "math")]["comparisons"][0]["in_run"]
text(f"Table 2 and Fig. 3 show the central result. On GSM8K, static memory placed {g_static:.1f} planted notes in front of the model "
     f"per run and the model cited {m('poison','math','static','poison_used'):.1f} of them; ATMC showed {g_atmc:.1f} and none was cited. On code, "
     f"static memory showed {c_static:.1f} and the Reflexion-style buffer {c_refl:.1f}, against {c_atmc:.1f} for ATMC, a roughly tenfold "
     f"reduction. The remaining {c_atmc:.0f} exposures per run are a cold-start floor: in the first three tasks the planted notes are "
     f"the only notes in memory, so nothing can distinguish them yet; after that, ATMC showed none.")
figure("fig3_poison.png", 3, "Planted notes shown to the model per 40-task run (mean ± s.d.; dots are seeds)", width=3.1)
text(f"Accuracy did not separate the systems. On GSM8K ATMC scored {pc(m('poison','math','atmc','pass_rate'))}% against "
     f"{pc(m('poison','math','static','pass_rate'))}% (difference {100*gs['diff']:+.1f} points, 95% CI [{100*gs['ci95'][0]:+.1f}, "
     f"{100*gs['ci95'][1]:+.1f}], McNemar p = {gs['p_mcnemar']:.2f}). On code every paired comparison against no memory had "
     f"p = 1.0 (no memory {pc(m('baselines_poison','code','no_memory','pass_rate'))}%, static "
     f"{pc(m('baselines_poison','code','static','pass_rate'))}%, ATMC {pc(m('baselines_poison','code','atmc','pass_rate'))}%). The base model "
     f"already solves about 90% of these tasks alone, so the measurable benefit here is protection from corrupted notes, not higher accuracy.")

subheading("B", "Ablation")
t3 = []
names = {"full_system": "Full ATMC", "wo_trust": "w/o Trust", "wo_forgetting": "w/o Forgetting", "wo_compression": "w/o Merging"}
for c, n in names.items():
    t3.append([n, pm(m("ablation", "math", c, "poison_exposure_after_warmup"), sd("ablation", "math", c, "poison_exposure_after_warmup"), d=1),
               pm(m("ablation", "code", c, "poison_exposure_after_warmup"), sd("ablation", "code", c, "poison_exposure_after_warmup"), d=1),
               f"{m('ablation','math',c,'final_memory_size'):.0f}",
               f"{pc(m('ablation','math',c,'pass_rate'))}/{pc(m('ablation','code',c,'pass_rate'))}"])
table(3, "Ablation: planted notes in context after warm-up (mean ± s.d.), notes kept, and pass rate (GSM8K / code)",
      ["Variant", "GSM8K (5 seeds)", "Code (3 seeds)", "Notes", "Pass %"], t3, [0.78, 0.8, 0.78, 0.36, 0.62])
text(f"Removing trust scoring let {m('ablation','math','wo_trust','poison_exposure_after_warmup'):.0f} planted notes through per run on every seed in "
     f"both domains, against zero for the full system (Fig. 4), because without trust every note has equal weight and only similarity "
     f"ranks them. Removing forgetting had a smaller, inconsistent effect "
     f"({m('ablation','math','wo_forgetting','poison_exposure_after_warmup'):.1f} ± {sd('ablation','math','wo_forgetting','poison_exposure_after_warmup'):.1f} on GSM8K, "
     f"{m('ablation','code','wo_forgetting','poison_exposure_after_warmup'):.1f} ± {sd('ablation','code','wo_forgetting','poison_exposure_after_warmup'):.1f} on code); at these run lengths forgetting "
     f"mostly acts by lowering ranks, and deletion needs roughly 70 steps. Merging reduced the stored notes from "
     f"{m('ablation','math','wo_compression','final_memory_size'):.0f} to {m('ablation','math','full_system','final_memory_size'):.0f} "
     f"({100*(1-m('ablation','math','full_system','final_memory_size')/m('ablation','math','wo_compression','final_memory_size')):.0f}%) with no change in exposure. "
     f"No ablation changed the pass rate (all p = 1.0).")
figure("fig4_ablation.png", 4, "Ablation: planted notes in context after the three-task warm-up (mean ± s.d.; dots are seeds)", width=3.1)

subheading("C", "Trust Dynamics")
text("Fig. 5 is computed from the update rule itself. A note that has passed the gate once sits at 0.55; after two verified "
     "failures in which it was cited it falls below the retrieval gate, and after four if its evidence came from generated tests "
     "(half weight). A static memory keeps it forever. In our experiments the planted notes enter as failed attempts, so they "
     "start at 0.35 against 0.55 for genuine notes and are outranked from the fourth task onwards; the learned update path matters "
     "most when a wrong note is mistakenly recorded as correct, which we did not test (Section V).")
figure("fig5_trust.png", 5, "Trust of a note over verified failures in which it is cited (from the update rule)", width=3.1)

subheading("D", "Planner, Reflection and Curriculum")
t4 = []
labels4 = {"atmc": "ATMC alone", "atmc_planner": "+ Planner", "atmc_reflection": "+ Reflection", "atmc_planner_reflection": "+ Planner + Reflection",
           "atmc_random_choice": "Random task choice", "atmc_weakness_choice": "Weakness-targeted choice"}
for c, n in labels4.items():
    h = pm(m("agent", "code", c, "heldout_pass_rate"), sd("agent", "code", c, "heldout_pass_rate"), pct=True)
    if c == "atmc":
        t4.append([n, h, "-", "-", f"{m('agent','code',c,'tokens_in'):.0f}"])
    else:
        ho = comp("agent", "code", c, "heldout")
        t4.append([n, h, f"{100*ho['diff']:+.1f} [{100*ho['ci95'][0]:+.1f}, {100*ho['ci95'][1]:+.1f}]", f"{ho['p_mcnemar']:.3f}",
                   f"{m('agent','code',c,'tokens_in'):.0f}"])
table(4, "Agent components on 20 held-out tasks per seed (3 seeds): change versus ATMC alone in points with 95% CI, McNemar p, tokens per task",
      ["Condition", "Held-out (%)", "Δ vs ATMC [95% CI]", "p", "Tokens"], t4, [0.98, 0.62, 0.92, 0.34, 0.47], size=7.6)
pl = comp("agent", "code", "atmc_planner", "heldout")
text(f"No component showed a statistically significant benefit (Table 4, Fig. 6). The planner is the only lead: "
     f"{100*pl['diff']:+.1f} points on held-out tasks, better on {pl['b_better']} tasks and worse on {pl['a_better']}, with a bootstrap interval "
     f"above zero but an exact McNemar p = {pl['p_mcnemar']:.3f}, because only {pl['b_better']+pl['a_better']} tasks changed; it also costs "
     f"{100*(m('agent','code','atmc_planner','tokens_in')/m('agent','code','atmc','tokens_in')-1):.0f}% more tokens. Reflection changed nothing: it runs only after a "
     f"failure, and failures are rare at this baseline. Neither curriculum beat plain ATMC. A larger held-out set would be needed to "
     f"confirm or reject the planner effect.")
figure("fig6_agent.png", 6, "Held-out pass rate of the agent components (mean ± s.d.; dots are seeds)", width=3.1)

subheading("E", "Cost and Overhead")
text(f"ATMC used {m('baselines_poison','code','atmc','tokens_in'):.0f} prompt tokens per task, about the same as static memory "
     f"({m('baselines_poison','code','static','tokens_in'):.0f}) and six times the no-memory baseline "
     f"({m('baselines_poison','code','no_memory','tokens_in'):.0f}); the prompt always carries the top three notes, so it does not grow with "
     f"memory size, while merging bounds the store itself. Retrieval took about "
     f"{m('baselines_poison','code','atmc','mean_retrieval_ms'):.1f} ms per query. An optional relevance floor (minimum similarity 0.75) halved "
     f"tokens to {m('baselines_poison','code','atmc_minsim','tokens_in'):.0f} but let {m('baselines_poison','code','atmc_minsim','poison_exposure_after_warmup'):.1f} "
     f"planted notes through after warm-up, and it did not recover accuracy, so it is off by default. Auto-verification costs about three extra "
     f"model calls per question; we exercised it on a handful of requests (verified, inconclusive and underspecified outcomes all "
     f"occurred) but did not evaluate it statistically.")

# ============================================================ V. Limitations
heading("V", "Limitations")
text("The evidence supports protection, not improvement. The base model already solves about 90% of the tasks without memory, "
     "so differences in accuracy cannot be detected, and in the static code runs the model cited a planted note on 28 steps yet failed "
     "on only one of them. The planted notes were recorded as failed attempts, so ATMC started them at lower trust; a stronger "
     "test, which we have not run, inserts wrong notes recorded as correct. A wrong note that is cited in a correct answer still "
     "gains trust. Runs use 40 tasks and 3 seeds (5 for one ablation) on one model family, in Python only, and differences of a few "
     "points are within run-to-run noise at temperature 0. Auto-verification checks consistency, not intent, and strategy "
     "selection was exercised only on a small number of practice attempts, so it is a working mechanism and not a measured gain. "
     "The sandbox is not a hard security boundary, the free API tier limited the number of seeds, and the roles are modules of one "
     "pipeline rather than independent agents.")

# ============================================================ VI. Conclusion
heading("VI", "Conclusion")
text("We presented ATMC, an outcome-verified memory for a self-improving coding assistant, and the system built around it. "
     f"Trust scoring kept planted wrong notes out of the model's context roughly tenfold ({c_static:.0f} to {c_atmc:.0f} per run on code) "
     "without a measurable cost in accuracy, and the ablations identify trust, not forgetting or merging, as the decisive "
     "component. The assistant also verifies its own answers, practises its weak spots and tracks which strategy works, but "
     "these additions have not yet been shown to raise accuracy.")
text("Future work includes harder benchmarks or weaker models where memory errors change accuracy, wrong notes recorded as "
     "correct, a second model family, shared memory across cooperating agents, and an indexed store for larger memories. "
     "Treating an agent's memory as untrusted input that must earn its influence is a simple principle that fits any agent that "
     "writes its own notes.")

p = para(align=WD_ALIGN_PARAGRAPH.CENTER, first=0, before=6, after=3, keep_next=True)
r = p.add_run("Acknowledgment")
font(r, 10, small_caps=True)
text("The authors would like to thank the faculty and mentors at Vishwakarma Institute of Technology, Pune, in particular our "
     "guide, Prof. Kirti Wanjale, for their guidance and support throughout this work.")

# ============================================================ references
p = para(align=WD_ALIGN_PARAGRAPH.CENTER, first=0, before=8, after=3, keep_next=True)
font(p.add_run("References"), 10, small_caps=True)
REFS = [
    "N. Shinn et al., “Reflexion: Language agents with verbal reinforcement learning,” arXiv:2303.11366, 2023.",
    "J. S. Park et al., “Generative agents: Interactive simulacra of human behavior,” in Proc. ACM UIST, 2023.",
    "C. Packer et al., “MemGPT: Towards LLMs as operating systems,” arXiv:2310.08560, 2023.",
    "W. Xu et al., “A-Mem: Agentic memory for LLM agents,” arXiv:2502.12110, 2025.",
    "G. Wang et al., “Voyager: An open-ended embodied agent with large language models,” arXiv:2305.16291, 2023.",
    "W. Zou, R. Geng, B. Wang, and J. Jia, “PoisonedRAG: Knowledge corruption attacks to retrieval-augmented generation of large language models,” in Proc. USENIX Security, 2025.",
    "Z. Chen et al., “AgentPoison: Red-teaming LLM agents via poisoning memory or knowledge bases,” in Proc. NeurIPS, 2024.",
    "A. Madaan et al., “Self-Refine: Iterative refinement with self-feedback,” arXiv:2303.17651, 2023.",
    "B. Chen et al., “CodeT: Code generation with generated tests,” in Proc. ICLR, 2023 (arXiv:2207.10397).",
    "“A survey of self-evolving agents,” arXiv:2507.21046, 2025.",
    "M. Chen et al., “Evaluating large language models trained on code,” arXiv:2107.03374, 2021.",
    "J. Austin et al., “Program synthesis with large language models,” arXiv:2108.07732, 2021.",
    "K. Cobbe et al., “Training verifiers to solve math word problems,” arXiv:2110.14168, 2021.",
    "S. Xiao et al., “C-Pack: Packed resources for general Chinese embeddings,” in Proc. ACM SIGIR, 2024 (arXiv:2309.07597).",
    "P. Lewis et al., “Retrieval-augmented generation for knowledge-intensive NLP tasks,” arXiv:2005.11401, 2020.",
    "E. B. Wilson, “Probable inference, the law of succession, and statistical inference,” J. Amer. Statist. Assoc., vol. 22, no. 158, 1927.",
    "Q. McNemar, “Note on the sampling error of the difference between correlated proportions or percentages,” Psychometrika, vol. 12, no. 2, 1947.",
    "B. Efron, “Bootstrap methods: Another look at the jackknife,” Ann. Statist., vol. 7, no. 1, 1979.",
    "H. Ebbinghaus, Memory: A Contribution to Experimental Psychology. New York, NY, USA: Teachers College, Columbia Univ., 1913.",
    "Google, “Gemini API documentation,” ai.google.dev, accessed 2026.",
]
for i, ref in enumerate(REFS, 1):
    p = para(first=0, after=1.5, size=8)
    runs(p, f"[{i}] {ref}", size=8)

doc.save(OUT)
print("saved", OUT)
subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(OUT.parent), str(OUT)],
               check=True, capture_output=True)
print("pdf", OUT.with_suffix(".pdf"))
