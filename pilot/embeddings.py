"""Text -> vector, behind one interface so the retrieval code never cares which
backend produced the vectors.

- fastembed (default): local ONNX sentence-embedding model. Free, no API quota,
  handles paraphrases and code text, which TF-IDF cannot.
- gemini: Gemini embedding API (uses the API quota).
- tfidf: the original bag-of-words baseline, kept so "real embeddings vs TF-IDF"
  stays a runnable comparison. Needs a corpus to fit on.

All backends return L2-normalised float32 vectors, so cosine similarity is a
plain dot product regardless of backend.
"""
from __future__ import annotations

import numpy as np


def _normalise(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class Embedder:
    name: str = "base"

    def __init__(self):
        self._cache: dict[str, np.ndarray] = {}

    def _embed_uncached(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def embed(self, text: str) -> np.ndarray:
        # The same question is embedded once per condition per task; caching
        # makes that free after the first call.
        if text not in self._cache:
            self._cache[text] = _normalise(self._embed_uncached(text))
        return self._cache[text]


class FastEmbedder(Embedder):
    def __init__(self, model: str):
        super().__init__()
        from fastembed import TextEmbedding

        self.name = f"fastembed:{model}"
        self._model = TextEmbedding(model)

    def _embed_uncached(self, text: str) -> np.ndarray:
        return next(iter(self._model.embed([text])))


class GeminiEmbedder(Embedder):
    def __init__(self, model: str):
        super().__init__()
        import llm  # reuses the configured Gemini client

        self.name = f"gemini:{model}"
        self._model = model
        self._llm = llm

    def _embed_uncached(self, text: str) -> np.ndarray:
        resp = self._llm._get_client().models.embed_content(model=self._model, contents=text)
        return np.array(resp.embeddings[0].values)


class TfidfEmbedder(Embedder):
    def __init__(self, corpus: list[str]):
        super().__init__()
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.name = "tfidf:500"
        self._vec = TfidfVectorizer(max_features=500).fit(corpus)

    def _embed_uncached(self, text: str) -> np.ndarray:
        return self._vec.transform([text]).toarray()[0]


def make_embedder(cfg, corpus: list[str] | None = None) -> Embedder:
    backend = cfg.embedding.backend
    if backend == "fastembed":
        return FastEmbedder(cfg.embedding.model)
    if backend == "gemini":
        return GeminiEmbedder(cfg.embedding.gemini_model)
    if backend == "tfidf":
        if not corpus:
            raise ValueError("tfidf backend needs a corpus to fit on")
        return TfidfEmbedder(corpus)
    raise ValueError(f"unknown embedding backend {backend!r}")
