"""Loads configs/default.yaml into an attribute-accessible object.

One file defines every tunable (trust, decay, composite weights, compression,
seeds, model, embedder). Experiments save `cfg.to_dict()` into their results
JSON, so a run can be reproduced from its own output.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "configs" / "default.yaml"


class Config:
    """Read-only-ish wrapper: cfg.trust.minimum, cfg['trust']['minimum'], cfg.to_dict()."""

    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, name):
        try:
            value = self._data[name]
        except KeyError:
            raise AttributeError(name) from None
        return Config(value) if isinstance(value, dict) else value

    def __getitem__(self, name):
        value = self._data[name]
        return Config(value) if isinstance(value, dict) else value

    def to_dict(self) -> dict:
        return copy.deepcopy(self._data)


def _deep_update(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _validate(d: dict) -> None:
    c = d["composite"]
    total = c["w_trust"] + c["w_importance"] + c["w_decay"]
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"composite weights must sum to 1.0, got {total}")
    if d["trust"]["penalty"] <= d["trust"]["reward"]:
        raise ValueError("trust.penalty must exceed trust.reward (bad memories lose trust faster)")
    if not 0.0 <= d["trust"]["minimum"] < 1.0:
        raise ValueError("trust.minimum must be in [0, 1)")
    if d["embedding"]["backend"] not in {"fastembed", "gemini", "tfidf"}:
        raise ValueError(f"unknown embedding backend {d['embedding']['backend']!r}")
    if d["agent"]["domain"] not in {"math", "code"}:
        raise ValueError(f"unknown agent.domain {d['agent']['domain']!r}")
    if d["agent"]["curriculum"] not in {"fixed", "random", "weakness"}:
        raise ValueError(f"unknown agent.curriculum {d['agent']['curriculum']!r}")
    if not 0.0 <= d["agent"]["curriculum_explore"] <= 1.0:
        raise ValueError("agent.curriculum_explore must be in [0, 1]")
    if d["retrieval"]["top_k"] < 1:
        raise ValueError("retrieval.top_k must be >= 1")
    if not 0.0 <= d["retrieval"].get("min_similarity", 0.0) < 1.0:
        raise ValueError("retrieval.min_similarity must be in [0, 1)")


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    """Load the YAML config; `overrides` is a nested dict merged on top,
    e.g. {"evaluation": {"n_tasks": 10}}."""
    with open(path or DEFAULT_PATH) as f:
        data = yaml.safe_load(f)
    if overrides:
        _deep_update(data, overrides)
    _validate(data)
    return Config(data)


def load_experiment_config() -> tuple[Config, int]:
    """Config for an experiment run, plus the seed to use.

    Environment overrides (handy for quick smoke tests and one-off comparisons):
      PILOT_SEED       seed for this run (default: first entry of evaluation.seeds)
      PILOT_N_TASKS    number of tasks in the sequence
      PILOT_EMBEDDING  fastembed | gemini | tfidf
    """
    overrides: dict = {}
    if os.environ.get("PILOT_N_TASKS"):
        overrides.setdefault("evaluation", {})["n_tasks"] = int(os.environ["PILOT_N_TASKS"])
    if os.environ.get("PILOT_EMBEDDING"):
        overrides.setdefault("embedding", {})["backend"] = os.environ["PILOT_EMBEDDING"]
    cfg = load_config(overrides=overrides)
    seed = int(os.environ.get("PILOT_SEED", cfg.evaluation.seeds[0]))
    return cfg, seed
