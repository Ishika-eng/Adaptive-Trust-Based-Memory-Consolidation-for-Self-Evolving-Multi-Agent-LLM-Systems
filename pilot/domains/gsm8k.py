"""GSM8K math word problems: one numeric answer, checked by comparison."""
from __future__ import annotations

import llm
from tasks import build_poison_text, load_tasks, make_wrong_answer

from .base import Domain, Verdict, memory_block, plan_block


class MathDomain(Domain):
    name = "math"

    def load_tasks(self, cfg, n, seed):
        return load_tasks(cfg, n, seed)

    def build_prompt(self, task, memories, plan=None):
        prefix, used = memory_block(memories)
        return (
            f"{prefix}{plan_block(plan)}Solve this grade-school math problem step by step. "
            f"End your response with a line of the exact form '#### <final numeric answer>'.{used}\n\n"
            f"Problem: {task['question']}"
        )

    def extract_answer(self, raw, task):
        return llm.extract_final_number(raw)

    def check(self, task, answer):
        if llm.numbers_match(answer, task["gold"]):
            return Verdict(True, "ok")
        if answer is None:
            return Verdict(False, "no_answer", "no final number found")
        return Verdict(False, "wrong_answer", f"answered {answer}, expected {task['gold']}")

    def reflection(self, task, raw, answer, verdict):
        return (
            f"Q: {task['question']}\n"
            f"A: {raw.strip()[:400]}\n"
            f"Outcome: {'CORRECT' if verdict.correct else 'INCORRECT'} (gold={task['gold']})"
        )

    def poison_text(self, task):
        return build_poison_text(task["question"], make_wrong_answer(task["gold"]))
