"""Builds memory stores from the config -- the only place constructor
arguments are mapped from config keys, so scripts and the demo app cannot
drift apart."""
from __future__ import annotations

from stores import AdaptiveTrustStore, NoMemoryStore, ReflexionStyleStore, StaticMemoryStore


def build_static(min_similarity: float = 0.0) -> StaticMemoryStore:
    return StaticMemoryStore(min_similarity=min_similarity)


def build_atmc(cfg, embedder, summarize_fn, label: str | None = None, **pillar_overrides) -> AdaptiveTrustStore:
    """`pillar_overrides` takes enable_trust / enable_forgetting /
    enable_compression, used by the ablation runner to switch one pillar off."""
    pillars = {
        "enable_trust": cfg.pillars.trust,
        "enable_forgetting": cfg.pillars.forgetting,
        "enable_compression": cfg.pillars.compression,
    }
    pillars.update(pillar_overrides)
    return AdaptiveTrustStore(
        t_min=cfg.trust.minimum,
        eta=cfg.trust.reward,
        delta=cfg.trust.penalty,
        initial_trust=cfg.trust.initial,
        lambda_0=cfg.decay.lambda_0,
        alpha=cfg.decay.alpha,
        beta=cfg.decay.beta,
        prune_every=cfg.decay.prune_every,
        prune_min_retention=cfg.decay.prune_min_retention,
        w1=cfg.composite.w_trust,
        w2=cfg.composite.w_importance,
        w3=cfg.composite.w_decay,
        compress_every=cfg.compression.every,
        compress_batch_size=cfg.compression.batch_size,
        compress_min_age_steps=cfg.compression.min_age_steps,
        embed_fn=embedder.embed,
        summarize_fn=summarize_fn,
        label=label,
        min_similarity=cfg.retrieval.to_dict().get("min_similarity", 0.0),
        **pillars,
    )


STORE_KINDS = ("none", "static", "reflexion", "atmc")


def build_store(kind: str, cfg, embedder, summarize_fn, **pillar_overrides):
    """One entry point for every memory system compared in the evaluation."""
    if kind == "none":
        return NoMemoryStore()
    if kind == "static":
        return build_static(cfg.retrieval.to_dict().get("min_similarity", 0.0))
    if kind == "reflexion":
        return ReflexionStyleStore(buffer_size=cfg.retrieval.top_k)
    if kind == "atmc":
        return build_atmc(cfg, embedder, summarize_fn, **pillar_overrides)
    raise ValueError(f"unknown store kind {kind!r}; expected one of {STORE_KINDS}")
