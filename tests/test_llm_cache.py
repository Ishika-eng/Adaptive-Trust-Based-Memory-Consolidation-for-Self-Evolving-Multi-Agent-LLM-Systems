import llm
from config import Config, load_config


class _Resp:
    text = "answer"
    usage_metadata = type("M", (), {"prompt_token_count": 7, "candidates_token_count": 3})()


class _Models:
    def __init__(self):
        self.n = 0

    def generate_content(self, **kw):
        self.n += 1
        return _Resp()


def _setup(monkeypatch, tmp_path, temperature=0.0, cache=True):
    d = load_config().to_dict()
    d["model"].update(temperature=temperature, cache=cache, request_interval_s=0)
    monkeypatch.setattr("config.ROOT", tmp_path)
    llm.configure(Config(d))
    models = _Models()
    monkeypatch.setattr(llm, "_get_client", lambda: type("C", (), {"models": models})())
    for k in llm.USAGE:
        llm.USAGE[k] = 0
    return models


def test_identical_prompt_is_served_from_cache_and_costs_stay_comparable(monkeypatch, tmp_path):
    models = _setup(monkeypatch, tmp_path)
    assert llm.complete("p") == llm.complete("p") == "answer"
    assert models.n == 1
    assert llm.USAGE["calls"] == 2 and llm.USAGE["api_calls"] == 1 and llm.USAGE["cache_hits"] == 1
    assert llm.USAGE["prompt_tokens"] == 14          # logical cost counted for both


def test_different_prompt_misses(monkeypatch, tmp_path):
    models = _setup(monkeypatch, tmp_path)
    llm.complete("a"); llm.complete("b")
    assert models.n == 2


def test_cache_disabled_when_not_deterministic(monkeypatch, tmp_path):
    models = _setup(monkeypatch, tmp_path, temperature=0.7)
    llm.complete("p"); llm.complete("p")
    assert models.n == 2 and llm.CACHE_PATH is None


def test_cache_is_usable_from_several_threads(monkeypatch, tmp_path):
    import threading
    models = _setup(monkeypatch, tmp_path)
    errors = []

    def worker(i):
        try:
            llm.complete(f"prompt {i % 2}")
        except Exception as e:      # noqa: BLE001
            errors.append(e)
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
        t.join()               # one after another, each in a new thread, like a server's worker pool
    assert not errors
