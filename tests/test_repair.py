"""Evidence-guided retry: one fresh attempt after a verified failure; a passing retry becomes a worked example."""
import pytest

import llm
from agent import build_agent
from config import Config, load_config
from domains import get_domain
from factory import build_static


def _agent(repair: bool, fake_llm, replies):
    d = load_config().to_dict()
    d["coding"] = {**d["coding"], "datasets": ["conventions"], "note_style": "evidence"}
    d["agent"] = {**d["agent"], "domain": "code", "curriculum": "fixed", "repair": repair}
    cfg = Config(d)
    dom = get_domain("code", cfg)
    tasks = [t for t in dom.load_tasks(cfg, 110, 0) if t["id"] == "CONV/money_paise/01"]
    from conftest import WordHashEmbedder
    a = build_agent(cfg, build_static(), WordHashEmbedder(), dom, tasks, 0)
    prompts = []
    def script(p):
        prompts.append(p)
        return replies.pop(0)
    fake_llm(script)
    return a, tasks[0], prompts


WRONG = "```python\ndef apply_discount(price, percent):\n    return price * (1 - percent / 100)\n```"
RIGHT = "```python\ndef apply_discount(price, percent):\n    return price * (100 - percent) // 100\n```"


def test_failed_attempt_gets_one_fresh_retry_and_a_passing_retry_is_stored_as_a_worked_example(fake_llm):
    a, task, prompts = _agent(True, fake_llm, [WRONG, RIGHT])
    r = a.step(task)
    assert not r.correct and r.repair == {"attempted": True, "passed": True}   # the REPORTED result is still the first attempt
    assert len(prompts) == 2
    assert "FAILED the project's own check" in prompts[1] and "expected 1799 (int)" in prompts[1]
    assert "price * (1 - percent / 100)" not in prompts[1]                       # the failed code is never shown again
    texts = [m.text for m in a.store.items]
    assert any("Verified solution" in t and "// 100" in t for t in texts) and any("FAILED the project's check" in t for t in texts)
    assert r.to_log()["repair_attempted"] and r.to_log()["repair_passed"]


def test_a_retry_that_also_fails_stores_no_example(fake_llm):
    a, task, _ = _agent(True, fake_llm, [WRONG, WRONG])
    r = a.step(task)
    assert r.repair == {"attempted": True, "passed": False}
    assert not any("Verified solution" in m.text for m in a.store.items)


def test_no_retry_when_off_or_when_the_first_attempt_passes(fake_llm):
    a, task, prompts = _agent(False, fake_llm, [WRONG])
    assert a.step(task).repair is None and len(prompts) == 1
    a, task, prompts = _agent(True, fake_llm, [RIGHT])
    r = a.step(task)
    assert r.correct and r.repair is None and len(prompts) == 1


def test_retry_can_show_retrieved_memory_and_credits_it(fake_llm):
    d = load_config().to_dict()
    d["coding"] = {**d["coding"], "datasets": ["conventions"], "note_style": "evidence"}
    d["agent"] = {**d["agent"], "domain": "code", "curriculum": "fixed", "repair": True, "repair_with_memory": True}
    cfg = Config(d)
    dom = get_domain("code", cfg)
    t = next(x for x in dom.load_tasks(cfg, 110, 0) if x["id"] == "CONV/money_paise/01")
    from conftest import WordHashEmbedder
    a = build_agent(cfg, build_static(), WordHashEmbedder(), dom, [t], 0)
    a.store.add("Task: earlier\nVerified solution (earlier attempt failed with: x): MARKER_NOTE\nOutcome: CORRECT", a.embedder.embed(t["question"]), True)
    prompts = []
    replies = [WRONG, RIGHT]
    fake_llm(lambda p: (prompts.append(p), replies.pop(0))[1])
    a.step(t)
    assert "MARKER_NOTE" in prompts[1] and "FAILED the project's own check" in prompts[1]
