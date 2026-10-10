"""Thin wrapper around Gemini: one retrying `complete()` call, token accounting,
and the small parsers shared by all domains (numeric answer, USED citations).
Prompts live in the domains; verification is deterministic, never an LLM judge."""
from __future__ import annotations

import os
import re
import time

from google import genai
from google.genai import types

MODEL_NAME = "gemini-3.5-flash-lite"
TEMPERATURE: float | None = None  # None = the API's own default
MAX_RETRIES = 4
MIN_INTERVAL_S = 0.0   # minimum spacing between calls; keeps every caller under the rate limit
_last_call = 0.0

_client = None

# Running totals across all calls in this process (for the efficiency metrics).
USAGE = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0}


class QuotaExhausted(RuntimeError):
    """The API's *daily* quota is used up. Retrying is pointless -- it resets in hours."""


def usage_snapshot() -> dict:
    return dict(USAGE)


def configure(cfg) -> None:
    """Apply model settings from configs/default.yaml (model.*)."""
    global MODEL_NAME, TEMPERATURE, MAX_RETRIES, MIN_INTERVAL_S
    MODEL_NAME = cfg.model.name
    TEMPERATURE = cfg.model.temperature
    MAX_RETRIES = cfg.model.max_retries
    MIN_INTERVAL_S = cfg.model.request_interval_s


def _get_client():
    global _client
    if _client is not None:
        return _client
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment.")
    _client = genai.Client(api_key=key)
    return _client


def complete(prompt: str) -> str:
    """One LLM call with exponential-backoff retries (rate limits, transient errors).
    Raises QuotaExhausted immediately on a daily-quota error instead of retrying."""
    global _last_call
    client = _get_client()
    gen_config = types.GenerateContentConfig(temperature=TEMPERATURE) if TEMPERATURE is not None else None
    last_err = None
    for attempt in range(MAX_RETRIES):
        wait = MIN_INTERVAL_S - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
        try:
            resp = client.models.generate_content(model=MODEL_NAME, contents=prompt, config=gen_config)
            USAGE["calls"] += 1
            meta = getattr(resp, "usage_metadata", None)
            if meta is not None:
                USAGE["prompt_tokens"] += getattr(meta, "prompt_token_count", 0) or 0
                USAGE["output_tokens"] += getattr(meta, "candidates_token_count", 0) or 0
            return resp.text or ""
        except Exception as e:
            if "PerDay" in str(e):
                raise QuotaExhausted(str(e)[:300]) from e
            last_err = e
            m = re.search(r"retry in ([\d.]+)s", str(e))
            time.sleep(float(m.group(1)) + 1 if m else 2 ** attempt * 2)
    raise RuntimeError(f"Gemini call failed after {MAX_RETRIES} retries: {last_err}")


def summarize(texts: list[str]) -> str:
    """Memory Compression (README 8.4): merge several raw memory entries into
    one short consolidated note. Used by AdaptiveTrustStore._compress()."""
    joined = "\n\n".join(f"Note {i+1}:\n{t}" for i, t in enumerate(texts))
    prompt = (
        "Merge the following past problem-solving notes into a single short "
        "consolidated note (3-4 sentences) that preserves the key facts, "
        "problem types, and outcomes from all of them, so it can replace "
        "them in a memory bank without losing important information.\n\n"
        f"{joined}\n\nConsolidated note:"
    )
    return complete(prompt).strip()


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
