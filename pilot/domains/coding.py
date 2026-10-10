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
        else:
            raise ValueError(f"unknown coding dataset {dataset!r}")
        return tasks

    def load_tasks(self, cfg, n, seed):
        tasks = []
        for name in cfg.coding.datasets:
            tasks += self._load(name)
        random.Random(seed).shuffle(tasks)
        return tasks[:n]

    # ---- prompting / extraction -------------------------------------
    def build_prompt(self, task, memories, plan=None):
        prefix, used = memory_block(memories)
        return (
            f"{prefix}{plan_block(plan)}Solve this Python programming task. Reply with the complete, working function "
            f"(including any imports it needs) in a single ```python code block.{used}\n\n"
            f"Task:\n{task['question']}"
        )

    def extract_answer(self, raw, task):
        blocks = _FENCE.findall(raw)
        if blocks:
            ep = task.get("entry_point")
            for b in blocks:  # prefer a block that defines the function we need
                if ep is None or f"def {ep}" in b:
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
        if res.timed_out:
            return Verdict(False, "timeout", f"exceeded {self.timeout_s}s")
        if res.returncode == 0:
            return Verdict(True, "ok")
        lines = [l for l in res.stderr.strip().splitlines() if l.strip()]
        last = lines[-1] if lines else f"exit code {res.returncode}"
        stage = "wrong_answer" if "AssertionError" in res.stderr else "runtime_error"
        return Verdict(False, stage, last[:200])

    # ---- memory notes ------------------------------------------------
    def reflection(self, task, raw, answer, verdict):
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
