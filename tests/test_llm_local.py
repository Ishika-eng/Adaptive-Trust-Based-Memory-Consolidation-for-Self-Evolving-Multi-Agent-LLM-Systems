import pytest

import llm
from config import Config, load_config


def _setup(monkeypatch, tmp_path, provider="ollama"):
    d = load_config().to_dict()
    d["model"].update(provider=provider, name="local-test", temperature=0.0, cache=True, request_interval_s=0, max_retries=2)
    monkeypatch.setattr("config.ROOT", tmp_path)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    llm.configure(Config(d))
    for k in llm.USAGE:
        llm.USAGE[k] = 0


def test_local_provider_uses_the_local_server_and_the_cache(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    seen = []
    monkeypatch.setattr(llm, "_ollama", lambda p: (seen.append(p), ("hi", 11, 4))[1])
    monkeypatch.setattr(llm, "_get_client", lambda: pytest.fail("the cloud client must not be used"))
    assert llm.complete("p") == llm.complete("p") == "hi"
    assert seen == ["p"] and llm.USAGE["api_calls"] == 1 and llm.USAGE["cache_hits"] == 1
    assert llm.USAGE["prompt_tokens"] == 22 and llm.USAGE["output_tokens"] == 8


def test_cache_does_not_mix_providers(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, "ollama")
    a = llm._cache_key("p")
    _setup(monkeypatch, tmp_path, "gemini")
    assert llm._cache_key("p") != a


def test_unreachable_local_server_gives_a_clear_error(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    def boom(p):
        raise ConnectionRefusedError("no server")
    monkeypatch.setattr(llm, "_ollama", boom)
    with pytest.raises(RuntimeError, match="ollama serve"):
        llm.complete("p")


def test_unknown_provider_is_rejected():
    d = load_config().to_dict()
    d["model"]["provider"] = "openai"
    from config import _validate
    with pytest.raises(ValueError):
        _validate(d)
