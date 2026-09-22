"""Thin wrapper around Gemini for the reasoning agent. The checker is
deterministic (regex-extracted numeric answer vs ground truth) -- no need
to burn a second LLM call verifying grade-school arithmetic."""
from __future__ import annotations

import os
import re
import time

from google import genai

MODEL_NAME = "gemini-3.5-flash-lite"

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment.")
    _client = genai.Client(api_key=key)
    return _client


def solve(question: str, retrieved_memories: list[str], max_retries: int = 4) -> str:
    """Ask the model to solve a GSM8K problem, optionally with retrieved
    past reflections as few-shot context. Returns raw model text."""
    client = _get_client()

    if retrieved_memories:
        context = "\n\n".join(f"Past experience {i+1}: {m}" for i, m in enumerate(retrieved_memories))
        prefix = (
            "Here are some past problem-solving experiences that may or may not be relevant:\n"
            f"{context}\n\n"
        )
        used_instruction = (
            " After that, add a line of the exact form 'USED: <comma-separated numbers>' "
            "listing which of the numbered past experiences above you actually relied on to "
            "solve this problem, or 'USED: none' if you didn't use any of them."
        )
    else:
        prefix = ""
        used_instruction = ""

    prompt = (
        f"{prefix}Solve this grade-school math problem step by step. "
        f"End your response with a line of the exact form '#### <final numeric answer>'.{used_instruction}\n\n"
        f"Problem: {question}"
    )

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            return resp.text
        except Exception as e:  # rate limit / transient errors
            last_err = e
            time.sleep(2 ** attempt * 2)
    raise RuntimeError(f"Gemini call failed after {max_retries} retries: {last_err}")


_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_final_number(text: str) -> str | None:
    if "####" in text:
        tail = text.split("####")[-1]
    else:
        tail = text
    m = _NUM_RE.search(tail)
    if not m:
        return None
    return m.group(0).replace(",", "")


_USED_RE = re.compile(r"USED:\s*(.*)", re.IGNORECASE)


def extract_used_mask(text: str, n_memories: int) -> list[bool]:
    """Parse the model's self-reported 'USED: 1,3' citation line into a
    boolean mask over the retrieved memories (in the order they were listed
    as 'Past experience 1', 'Past experience 2', ...). Defaults to "used
    everything" if the model didn't follow the format, so a parse failure
    degrades to the old (pre-fix) behavior rather than silently zeroing
    every memory's trust update."""
    if n_memories == 0:
        return []
    m = _USED_RE.search(text)
    if not m:
        return [True] * n_memories
    body = m.group(1).strip().lower()
    if body.startswith("none"):
        return [False] * n_memories
    indices = set()
    for tok in re.findall(r"\d+", body):
        idx = int(tok) - 1
        if 0 <= idx < n_memories:
            indices.add(idx)
    if not indices:
        return [True] * n_memories  # unparseable -> fall back, don't silently zero out
    return [i in indices for i in range(n_memories)]


def numbers_match(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold.replace(",", ""))) < 1e-4
    except ValueError:
        return pred.strip() == gold.strip()
