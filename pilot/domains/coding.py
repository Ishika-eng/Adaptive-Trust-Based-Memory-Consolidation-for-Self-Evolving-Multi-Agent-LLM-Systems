"""Coding problems (HumanEval, MBPP): the answer is a program, verified by
actually running the problem's unit tests inside the sandbox -- the objective,
execution-based ground truth ATMC's trust signal needs."""
from __future__ import annotations

import ast
import json
import random
import re
import warnings
from pathlib import Path

from config import ROOT
from sandbox import run_python

from .base import Domain, Verdict, memory_block, plan_block

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_RETURN = re.compile(r"^[ \t]*return[ \t]+(.+?)[ \t]*$", re.MULTILINE)
_USED_LINE = re.compile(r"^\s*USED:.*$", re.IGNORECASE | re.MULTILINE)

# MBPP tasks whose own reference solution fails its own tests (verified by running
# all 974): two assert wrong values, two use a `Node` class the dataset never defines.
KNOWN_BROKEN_MBPP = {180, 367, 493, 927}

# (pattern, replacement): single-site edits used to build plausible-but-wrong code
_MUTATIONS = [
    ("==", "!="), ("<=", "<"), (">=", ">"), ("<", "<="), (">", ">="),
    (" + ", " - "), (" - ", " + "), (" and ", " or "), (" or ", " and "),
    ("True", "False"), ("False", "True"), (" // ", " / "), (" % ", " // "),
    ("+ 1", "+ 2"), ("- 1", "- 2"), ("[0]", "[1]"), ("[1]", "[0]"), ("min(", "max("), ("max(", "min("),
]


class CodingDomain(Domain):
    name = "code"

    def __init__(self, cfg):
        self.cfg = cfg
        self.timeout_s = cfg.coding.timeout_s
        self._poison_cache: dict[str, str | None] = {}

    # ---- tasks -------------------------------------------------------
    def _load(self, dataset: str) -> list[dict]:
        tasks = []
        if dataset == "humaneval":
            for line in open(ROOT / "pilot/data/HumanEval.jsonl"):
                d = json.loads(line)
                tasks.append({
                    "id": d["task_id"], "dataset": "humaneval", "question": d["prompt"],
                    "entry_point": d["entry_point"], "test": d["test"],
                    "reference": d["canonical_solution"],
                })
        elif dataset == "mbpp":
            for line in open(ROOT / "pilot/data/mbpp.jsonl"):
                d = json.loads(line)
                if d["task_id"] in KNOWN_BROKEN_MBPP:
                    continue
                tests = d["test_list"]
                tasks.append({
                    "id": f"MBPP/{d['task_id']}", "dataset": "mbpp",
                    "question": f"{d['text']}\nYour code should pass this test:\n{tests[0]}",
                    "entry_point": None, "tests": tests, "setup": d.get("test_setup_code", ""),
                    "reference": d["code"].replace("\r\n", "\n"),
                })
        elif dataset in ("conventions", "conventions_heldout", "conventions_drift", "conventions_drift2", "conventions_drift3", "conventions_drift4"):   # pilot/benchmarks/build_*.py
            for line in open(ROOT / f"pilot/data/{dataset}.jsonl"):
                d = json.loads(line)
                tasks.append({k: d[k] for k in ("id", "dataset", "family", "role", "question", "entry_point", "tests", "setup",
                                                "reference", "phase") if k in d})
        else:
            raise ValueError(f"unknown coding dataset {dataset!r}")
        return tasks

    def load_tasks(self, cfg, n, seed):
        tasks = []
        for name in cfg.coding.datasets:
            tasks += self._load(name)
        rng = random.Random(seed)
        if list(cfg.coding.datasets) in (["conventions_drift3"],):
            # recurring rule change: old rule (A), new rule (B), the old rule again (A2)
            parts = [[t for t in tasks if t["phase"] == ph] for ph in ("A", "B", "A2")]
            for p_ in parts:
                rng.shuffle(p_)
            return (parts[0] + parts[1] + parts[2])[:n]
        if list(cfg.coding.datasets) in (["conventions_drift"], ["conventions_drift2"], ["conventions_drift4"]):
            # a rule-change stream: every question first under the OLD rule (phase A), then under the NEW one (phase B)
            a = [t for t in tasks if t["phase"] == "A"]
            b = [t for t in tasks if t["phase"] == "B"]
            rng.shuffle(a); rng.shuffle(b)
            return (a + b)[:n]
        rng.shuffle(tasks)
        return tasks[:n]

    # ---- prompting / extraction -------------------------------------
    def build_prompt(self, task, memories, plan=None):
        prefix, used = memory_block(memories)
        if task.get("dataset") == "user":
            tests = f"\n\nYour code must pass these tests:\n{task['tests'][0]}" if task.get("tests") else ""
            if task.get("context"):
                request = (f"Earlier in this conversation:\n{task['context']}\n\n"
                           f"The user now says (a follow-up to the above):\n{task['request']}")
            else:
                request = f"Request:\n{task['question']}"
            return (
                f"{prefix}{plan_block(plan)}You are a careful Python coding assistant. First give a brief explanation "
                f"(2-4 sentences) of your approach, then the complete working code in a single ```python code block "
                f"(include any imports; do not add example usage or tests).{used}\n\n"
                f"{request}{tests}"
            )
        return (
            f"{prefix}{plan_block(plan)}Solve this Python programming task. Reply with the complete, working function "
            f"(including any imports it needs) in a single ```python code block.{used}\n\n"
            f"Task:\n{task['question']}"
        )

    def revise_prompt(self, task, draft):
        request = task.get("request") or task["question"]
        return (
            "Below are a coding request and a draft solution. Review the draft carefully for bugs, missed edge cases "
            "(empty input, single elements, duplicates, boundaries) and anything the request asks for that it misses. "
            "Then reply with a brief explanation (2-4 sentences) followed by the final, complete, corrected code in a "
            "single ```python code block (unchanged if it was already correct; do not add example usage or tests).\n\n"
            f"Request:\n{request}\n\nDraft solution:\n```python\n{draft}\n```"
        )

    def extract_answer(self, raw, task):
        blocks = _FENCE.findall(raw)
        if blocks:
            ep = task.get("entry_point")
            for b in blocks:  # prefer a block that defines the function we need
                if (ep is None and task.get("dataset") != "user") or (ep and f"def {ep}" in b):
                    return b.strip("\n")
            if task.get("dataset") == "user":
                for b in blocks:
                    if re.search(r"^\s*(def|class) ", b, re.MULTILINE):
                        return b.strip("\n")
            return blocks[-1].strip("\n")
        text = _USED_LINE.sub("", raw).strip()
        return text or None

    # ---- verification ------------------------------------------------
    def _program(self, task, code: str) -> str:
        if task["dataset"] == "humaneval":
            ep = task["entry_point"]
            # The prompt supplies imports and helper functions; a docstring-only
            # stub is valid Python, so the model's own definition simply overrides
            # it, and a body-only reply continues the stub.
            return f"{task['question']}\n\n{code}\n\n{task['test']}\n\ncheck({ep})\n"
        return f"{task['setup']}\n{code}\n\n" + "\n".join(task["tests"]) + "\n"

    def check(self, task, answer):
        if not answer:
            return Verdict(False, "no_answer", "no code found in the reply")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # datasets contain non-raw regex strings
                ast.parse(answer if "def " in answer else task["question"] + answer)
        except SyntaxError as e:
            return Verdict(False, "syntax_error", f"line {e.lineno}: {e.msg}")
        res = run_python(self._program(task, answer), timeout_s=self.timeout_s)
        if task.get("dataset") == "user" and not task.get("tests"):
            return self._check_without_tests(res)
        if res.timed_out:
            return Verdict(False, "timeout", f"exceeded {self.timeout_s}s")
        if res.returncode == 0:
            return Verdict(True, "ok")
        lines = [l for l in res.stderr.strip().splitlines() if l.strip()]
        last = lines[-1] if lines else f"exit code {res.returncode}"
        stage = "wrong_answer" if "AssertionError" in res.stderr else "runtime_error"
        return Verdict(False, stage, last[:200])

    def _check_without_tests(self, res) -> Verdict:
        """A user task with no tests has no ground truth. The code can still fail objectively
        (crash, hang), which is a verified failure; if it merely runs, the result is *unverified*
        and memory waits for the user's verdict."""
        if res.timed_out:
            return Verdict(False, "timeout", f"exceeded {self.timeout_s}s")
        if res.returncode != 0:
            lines = [l for l in res.stderr.strip().splitlines() if l.strip()]
            return Verdict(False, "runtime_error", (lines[-1] if lines else f"exit code {res.returncode}")[:200])
        return Verdict(False, "unverified", res.stdout.strip()[:500], verified=False)

    def explanation(self, raw: str) -> str:
        """The prose of a reply: everything outside the code fences and the USED line."""
        text = _FENCE.sub("", raw)
        return _USED_LINE.sub("", text).strip()

    @staticmethod
    def user_task(question: str, tests: str = "", idx: int = 0, context: str = "", previous: str = "") -> dict:
        """A task typed by the user. For a follow-up, `context` (the earlier exchanges) goes into the
        prompt, while `question` also names the earlier request so retrieval and the saved memory note
        make sense on their own ("make it faster" alone would mean nothing later)."""
        tests = tests.strip()
        request = question.strip()
        full = f"{previous.strip()}\nFollow-up: {request}" if context and previous else request
        return {
            "id": f"user/{idx}", "dataset": "user", "question": full, "request": request, "context": context,
            "entry_point": None, "tests": [tests] if tests else [], "setup": "", "reference": "",
        }

    # ---- memory notes ------------------------------------------------
    @staticmethod
    def _typed_evidence(detail: str) -> str:
        """'f(1): expected 5, got 5.0' -> 'f(1) -> expected 5 (int) but got 5.0 (float)'. Pure parsing of what the checker
        printed (no model involved), so the type information that usually reveals a convention cannot be hallucinated."""
        m = re.search(r"(.+?): expected (.+), got (.+)$", detail)
        if not m:
            return detail
        def kind(x):
            try:
                return type(ast.literal_eval(x)).__name__
            except Exception:
                return "?"
        return f"{m.group(1)} -> expected {m.group(2)} ({kind(m.group(2))}) but got {m.group(3)} ({kind(m.group(3))})"

    def note_solution(self, text):
        if "Verified solution (" in text:
            body = text.split("Verified solution (", 1)[1]
            if "):\n" not in body or "\nOutcome: CORRECT" not in body:
                return None
            return body.split("):\n", 1)[1].split("\nOutcome: CORRECT", 1)[0]
        if "\nSolution:\n" in text and "\nOutcome: CORRECT" in text:
            return text.split("\nSolution:\n", 1)[1].split("\nOutcome: CORRECT", 1)[0]
        return None

    def reverify(self, task, code):
        ep = task.get("entry_point")
        if not ep or f"def {ep}(" not in code:
            return None
        return bool(self.check(task, code).correct)

    def failure_signature(self, verdict):
        from consolidate import detail_signature
        return detail_signature(verdict.detail) if verdict.detail else set()

    def repair_prompt(self, task, verdict, memories=None):
        if verdict.stage not in ("wrong_answer", "runtime_error") or not verdict.detail:
            return None
        what = self._typed_evidence(verdict.detail) if verdict.stage == "wrong_answer" else verdict.detail
        return (self.build_prompt(task, memories or []) +
                f"\n\nA first attempt at this task FAILED the project's own check: {what}\n"
                "The project follows conventions the task description does not state. Work out what the check expects "
                "from that result, then write a new, complete solution that satisfies the project's conventions.")

    def worked_example(self, task, answer, verdict):
        code = (answer or "").strip()[:1200]
        if not code:
            return None
        what = self._typed_evidence(verdict.detail) if verdict.stage == "wrong_answer" else verdict.detail
        return (f"Task: {task['question'][:700]}\nVerified solution (it passes the project's check; an earlier attempt "
                f"failed with: {what}):\n{code}\nOutcome: CORRECT")

    def note_claim_verified(self, verdict) -> bool:
        return (not verdict.correct and self.cfg.to_dict().get("coding", {}).get("note_style") == "evidence"
                and verdict.stage in ("wrong_answer", "runtime_error"))

    def reflection(self, task, raw, answer, verdict):
        if (not verdict.correct and self.cfg.to_dict().get("coding", {}).get("note_style") == "evidence"
                and verdict.stage in ("wrong_answer", "runtime_error")):
            # the failed code is left out: it anchors a small model on the very mistake the note should prevent
            what = self._typed_evidence(verdict.detail) if verdict.stage == "wrong_answer" else verdict.detail
            return f"Task: {task['question'][:700]}\nA solution to this task FAILED the project's check: {what}\nOutcome: INCORRECT"
        code = (answer or "").strip()[:1200]
        outcome = "CORRECT" if verdict.correct else f"INCORRECT ({verdict.stage}: {verdict.detail})"
        return f"Task: {task['question'][:700]}\nSolution:\n{code}\nOutcome: {outcome}"

    # ---- poisoning ---------------------------------------------------
    def _reference_code(self, task) -> str:
        if task["dataset"] == "humaneval":
            return task["question"] + task["reference"]
        return task["reference"]

    def poison_text(self, task):
        """A plausible-looking solution that really fails the tests: a single-site
        edit of the reference solution, verified by actually running it."""
        if task["id"] in self._poison_cache:
            return self._poison_cache[task["id"]]
        ref = self._reference_code(task)
        # Only edit the solution, never the docstring (edits there change nothing).
        lo = len(task["question"]) if task["dataset"] == "humaneval" else 0
        edits = [(i, i + len(pat), rep) for pat, rep in _MUTATIONS for i in _find_all(ref, pat) if i >= lo]
        random.Random(task["id"]).shuffle(edits)
        # Fallback for solutions with no flippable operator: return None instead.
        edits += [(m.start(1), m.end(1), "None") for m in _RETURN.finditer(ref) if m.start(1) >= lo]
        wrong = None
        for start, end, rep in edits[:40]:
            cand = ref[:start] + rep + ref[end:]
            v = self.check(task, cand)
            if v.stage in ("wrong_answer", "runtime_error"):
                wrong = cand
                break
        text = None
        if wrong:
            text = (f"Task: {task['question'][:700]}\nSolution:\n{wrong[:1200]}\n"
                    f"Outcome: INCORRECT (wrong_answer: assertion failed)")
        self._poison_cache[task["id"]] = text
        return text


def _find_all(s: str, sub: str):
    start = 0
    while (i := s.find(sub, start)) != -1:
        yield i
        start = i + 1
