"""The Domain interface: everything that differs between task types (math word
problems, coding problems) lives behind it, so the agent loop is written once.

A domain knows how to: load tasks, turn a task + retrieved memories into a
prompt, pull the answer out of the model's reply, verify it against ground
truth, write the memory note describing the attempt, and build a deliberately
wrong note for poisoning experiments.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Verdict:
    correct: bool
    stage: str    # "ok" | "wrong_answer" | "syntax_error" | "runtime_error" | "timeout" | "no_answer"
    detail: str = ""   # short, human-readable failure detail (fed into reflections)
    evidence: str = "tests"   # where a verified verdict came from: "tests" | "user" | "generated" (weaker)
    verified: bool = True   # False = no ground truth (a user task with no tests): memory must not learn from it yet


def memory_block(memories: list[str]) -> tuple[str, str]:
    """(prefix, used_instruction): the retrieved-memory context shown to the
    model, and the instruction to cite which memories it actually relied on."""
    if not memories:
        return "", ""
    context = "\n\n".join(f"Past experience {i+1}: {m}" for i, m in enumerate(memories))
    prefix = (
        "Here are some past problem-solving experiences that may or may not be relevant:\n"
        f"{context}\n\n"
    )
    used = (
        " After that, add a line of the exact form 'USED: <comma-separated numbers>' "
        "listing which of the numbered past experiences above you actually relied on to "
        "solve this problem, or 'USED: none' if you didn't use any of them."
    )
    return prefix, used


def plan_block(plan: str | None) -> str:
    return f"Plan to follow:\n{plan.strip()}\n\n" if plan else ""


class Domain:
    name: str = "base"

    def load_tasks(self, cfg, n: int, seed: int) -> list[dict]:
        raise NotImplementedError

    def build_prompt(self, task: dict, memories: list[str], plan: str | None = None) -> str:
        raise NotImplementedError

    def extract_answer(self, raw: str, task: dict):
        raise NotImplementedError

    def revise_prompt(self, task: dict, draft) -> str:
        """Ask the model to review its own draft and return a corrected final answer (the `revise` strategy)."""
        raise NotImplementedError(f"the {self.name} domain does not support self-revision")

    def check(self, task: dict, answer) -> Verdict:
        raise NotImplementedError

    def reflection(self, task: dict, raw: str, answer, verdict: Verdict) -> str:
        """The memory note stored after an attempt."""
        raise NotImplementedError

    def poison_text(self, task: dict) -> str | None:
        """A plausible but wrong note for this task, or None if one can't be built."""
        raise NotImplementedError

    def poison_specs(self, tasks: list[dict], embedder, every: int) -> list[tuple[str, object]]:
        specs = []
        for i, task in enumerate(tasks):
            if i % every == 0:
                text = self.poison_text(task)
                if text is not None:
                    specs.append((text, embedder.embed(task["question"])))
        return specs
