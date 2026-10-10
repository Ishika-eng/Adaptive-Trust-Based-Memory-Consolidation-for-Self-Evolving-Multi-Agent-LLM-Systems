"""Planner: decomposes a task into steps *before* the reasoning step.

It deliberately sees only the task, never the retrieved memories, so a
poisoned memory cannot bend the plan -- planning and memory stay independent
failure domains.
"""
from __future__ import annotations

import llm


class Planner:
    def plan(self, task: dict) -> str:
        prompt = (
            "Break the following task into a short numbered plan (at most 5 steps) describing HOW to "
            "solve it. Do NOT solve it: give no final answer and write no code.\n\n"
            f"Task:\n{task['question']}"
        )
        return llm.complete(prompt).strip()
