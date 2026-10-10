"""Reflector: turns a failed attempt into a short lesson that is stored with
the memory, instead of a bare CORRECT/INCORRECT tag.

Only failures are reflected on (one extra LLM call each). The lesson is an
*unverified claim*, so it is stored inside a note whose trust is still set by
the objective checker -- a wrong lesson loses trust like any other bad memory.
"""
from __future__ import annotations

import llm
from domains.base import Verdict


class Reflector:
    def lesson(self, task: dict, answer, verdict: Verdict) -> str:
        prompt = (
            "A previous attempt at the task below failed.\n\n"
            f"Task:\n{task['question'][:900]}\n\n"
            f"Attempt:\n{str(answer)[:900]}\n\n"
            f"What the checker reported: {verdict.stage}: {verdict.detail}\n\n"
            "In 2-3 sentences, state the most likely cause of the failure and what to do differently "
            "next time. Do not restate the full solution."
        )
        return llm.complete(prompt).strip()
