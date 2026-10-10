from .base import Domain, Verdict
from .gsm8k import MathDomain
from .coding import CodingDomain


def get_domain(name: str, cfg) -> Domain:
    if name == "math":
        return MathDomain()
    if name == "code":
        return CodingDomain(cfg)
    raise ValueError(f"unknown domain {name!r} (expected 'math' or 'code')")
