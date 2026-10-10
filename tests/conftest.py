import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pilot"))
sys.path.insert(0, str(ROOT / "app" / "backend"))

import llm  # noqa: E402
from config import load_config  # noqa: E402
from embeddings import Embedder  # noqa: E402


class WordHashEmbedder(Embedder):
    """Deterministic bag-of-words embedder: similar wording -> similar vectors.
    Fast and offline, so tests never load the real model."""
    name = "test:wordhash64"

    def _embed_uncached(self, text):
        v = np.zeros(64)
        for w in text.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
        return v


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def embedder():
    return WordHashEmbedder()


class FakeLLM:
    """Replaces llm.complete. `script` is a function (prompt) -> reply; every
    prompt is recorded so tests can assert on what the agent actually sent."""

    def __init__(self, script):
        self.script = script
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        llm.USAGE["calls"] += 1
        llm.USAGE["prompt_tokens"] += len(prompt.split())
        llm.USAGE["output_tokens"] += 10
        return self.script(prompt)


@pytest.fixture
def fake_llm(monkeypatch):
    def install(script):
        fake = FakeLLM(script)
        monkeypatch.setattr(llm, "complete", fake)
        return fake
    return install
