"""Auto-verification for coding tasks that arrive without tests.

With no tests there is no ground truth, so the assistant builds some -- carefully, because a test
written by the same model can be wrong. Three safeguards:

  1. Several *independent* solutions are written (the answer plus alternatives with different approaches).
  2. A generated test is only trusted ("reliable") if at least two solutions pass it, so a mistaken
     test that nothing agrees with cannot condemn a correct answer.
  3. Whatever it concludes, it is weaker evidence than real tests: memory updates move trust less for it
     (autoverify.evidence_weight) and the user can dispute the result afterwards.

It checks consistency with the request as understood, never the user's unstated intent -- hence the
"underspecified" outcome. Outcomes: "verified" (the answer passes every reliable test), "switched" (it did not, but an alternative
did, and that one is returned), "failed" (no solution passes every reliable test), "underspecified" (the request is too vague to test) or "inconclusive"
(too few tests that independent solutions agree on, so it says nothing). Only the first two let memory learn without the user.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field

import llm
from sandbox import run_python

MARK = "__ATMC_RESULT__"
_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_VARIANTS = [
    "Use a different approach from the most obvious one, while keeping the behaviour identical.",
    "Write it in a different style from the most obvious solution (for example iterative vs recursive, or built-ins vs manual loops), keeping the behaviour identical.",
    "Prefer the simplest possible implementation, handling every edge case explicitly.",
]


@dataclass
class AutoCheck:
    status: str                 # verified | switched | failed | inconclusive | unavailable
    n_tests: int = 0
    n_reliable: int = 0
    n_candidates: int = 0
    passed: int = 0             # reliable tests the returned answer passes
    answer: str | None = None   # the answer to use (may be an alternative)
    failing: list[str] = field(default_factory=list)   # reliable tests the original answer fails
    note: str = ""

    def to_json(self) -> dict:
        return {k: getattr(self, k) for k in ("status", "n_tests", "n_reliable", "n_candidates", "passed", "failing", "note")}


def function_name(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    return names[0] if names else None


def parse_tests(text: str, limit: int) -> list[str]:
    """One-line `assert ...` statements that are valid Python, deduplicated, at most `limit`."""
    blocks = _FENCE.findall(text)
    lines = "\n".join(blocks).splitlines() if blocks else text.splitlines()
    tests, seen = [], set()
    for line in lines:
        line = line.strip()
        if not line.startswith("assert ") or line in seen:
            continue
        try:
            ast.parse(line)
        except SyntaxError:
            continue
        seen.add(line)
        tests.append(line)
    return tests[:limit]


def run_tests(code: str, tests: list[str], timeout_s: float) -> list[bool]:
    """Pass/fail of each test, one sandbox run per solution. A crash, hang or exit fails every test."""
    program = (
        f"{code}\n\nimport json as __j\n__T = {tests!r}\n__R = []\nfor __t in __T:\n    try:\n"
        f"        exec(__t, globals())\n        __R.append(True)\n    except BaseException:\n        __R.append(False)\n"
        f"print({MARK!r} + __j.dumps(__R))\n"
    )
    res = run_python(program, timeout_s=timeout_s)
    if res.timed_out or res.returncode != 0:
        return [False] * len(tests)
    for line in reversed(res.stdout.splitlines()):
        if line.startswith(MARK):
            try:
                out = json.loads(line[len(MARK):])
                if len(out) == len(tests):
                    return [bool(x) for x in out]
            except ValueError:
                pass
    return [False] * len(tests)


class AutoVerifier:
    def __init__(self, domain, cfg):
        self.domain = domain
        self.n_candidates = cfg.autoverify.candidates
        self.n_tests = cfg.autoverify.tests
        self.timeout = cfg.coding.timeout_s

    def _tests_prompt(self, task: dict, fname: str) -> str:
        if task.get("context"):        # a follow-up only makes sense with the conversation it follows
            request = (f"Earlier in this conversation:\n{task['context']}\n\n"
                       f"The user now says (a follow-up to the above):\n{task['request']}")
        else:
            request = task.get("request") or task["question"]
        return (
            f"Write up to {self.n_tests} Python `assert` statements that test a function named `{fname}` for the "
            f"request below. Cover normal cases AND edge cases (empty input, a single element, duplicates, "
            f"boundaries, negative or large values where relevant). Output only the assert statements, one per "
            f"line, inside a single ```python block. Do not define functions, import anything, or add comments. "
            f"Only assert behaviour the request clearly specifies. If the request is too vague to define one clear "
            f"correct behaviour (for example 'clean up a string' without saying what cleaning means), output only "
            f"the single word UNDERSPECIFIED instead of guessing.\n\nRequest:\n{request}"
        )

    def _alternative(self, task: dict, fname: str, i: int) -> str | None:
        prompt = (f"{self.domain.build_prompt(task, [], None)}\n\nName the main function `{fname}`. "
                  f"{_VARIANTS[i % len(_VARIANTS)]}")
        code = self.domain.extract_answer(llm.complete(prompt), task)
        return code if code and function_name(code) else None

    def run(self, task: dict, answer: str | None) -> AutoCheck:
        fname = function_name(answer or "")
        if not fname:
            return AutoCheck("unavailable", note="no function to check")
        raw_tests = llm.complete(self._tests_prompt(task, fname))
        if "UNDERSPECIFIED" in raw_tests.upper() and not parse_tests(raw_tests, 1):
            return AutoCheck("underspecified", note="the request is open to interpretation, so it cannot be checked automatically")
        tests = parse_tests(raw_tests, self.n_tests)
        if not tests:
            return AutoCheck("inconclusive", note="could not generate usable tests")
        candidates = [answer]
        for i in range(self.n_candidates - 1):
            alt = self._alternative(task, fname, i)
            if alt and alt not in candidates:
                candidates.append(alt)
        results = [run_tests(c, tests, self.timeout) for c in candidates]
        # a test is reliable only if at least two independent solutions agree it is valid
        reliable = [j for j in range(len(tests)) if sum(r[j] for r in results) >= 2]
        base = dict(n_tests=len(tests), n_reliable=len(reliable), n_candidates=len(candidates))
        # Two wrong solutions can both pass one trivial test, so a verdict needs a solid body of agreed tests.
        if len(candidates) < 2 or len(reliable) < max(2, (len(tests) + 1) // 2):
            return AutoCheck("inconclusive", **base, note="the independent solutions did not agree on enough tests")
        passes_all = [all(r[j] for j in reliable) for r in results]
        failing = [tests[j] for j in reliable if not results[0][j]]
        if passes_all[0]:
            return AutoCheck("verified", **base, passed=len(reliable), answer=answer)
        for c, ok in zip(candidates[1:], passes_all[1:]):
            if ok:
                return AutoCheck("switched", **base, passed=len(reliable), answer=c, failing=failing,
                                 note="the first answer failed generated tests that another solution passed")
        return AutoCheck("failed", **base, passed=sum(results[0][j] for j in reliable), answer=answer, failing=failing,
                         note="no solution passed every reliable test")
