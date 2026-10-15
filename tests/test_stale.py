"""Rule-change detection: a stored solution that the CURRENT checker rejects is proven stale; its neighbourhood goes with it."""
import numpy as np

from agent import build_agent
from config import Config, load_config
from domains import get_domain
from factory import build_static
from stores import AdaptiveTrustStore, StaticMemoryStore

OLD = "def format_date(year, month, day):\n    return f'{day:02d}-{month:02d}-{year}'"
NEW = "def format_date(year, month, day):\n    return f'{year}-{month:02d}-{day:02d}'"


def ex(code):
    return f"Task: Write format_date\nVerified solution (it passes the project's check; an earlier attempt failed with: x):\n{code}\nOutcome: CORRECT"


def _domain():
    d = load_config().to_dict()
    d["coding"] = {**d["coding"], "datasets": ["conventions_drift"], "note_style": "evidence"}
    cfg = Config(d)
    return cfg, get_domain("code", cfg)


def test_note_solution_and_reverify():
    cfg, dom = _domain()
    assert dom.note_solution(ex(OLD)) == OLD
    assert dom.note_solution("Task: x\nSolution:\ncode here\nOutcome: CORRECT") == "code here"
    assert dom.note_solution("Task: x\nA solution to this task FAILED the project's check: y\nOutcome: INCORRECT") is None
    task = next(t for t in dom.load_tasks(cfg, 58, 0) if t["phase"] == "B" and t["entry_point"] == "format_date")
    assert dom.reverify(task, OLD) is False          # the old solution fails the new tests: proven stale
    assert dom.reverify(task, NEW) is True
    assert dom.reverify(task, "def other():\n    return 1") is None   # not about this function: cannot be judged


def test_marking_a_note_stale_demotes_its_older_neighbourhood_but_not_newer_or_distant_notes():
    s = AdaptiveTrustStore(enable_forgetting=False, enable_compression=False)
    near, far = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    for text, v in (("old1", near * 0.99 + far * 0.1), ("old2", near), ("distant", far)):
        s.add(text, v, correct=True)
    stale = s.items[1]
    s.add("newer", near, correct=True)
    n = s.mark_stale(stale)
    t = {m.text: m.trust for m in s.items}
    assert n == 1 and t["old2"] == 0.0 and t["old1"] < s.t_min              # the older neighbour fell below the gate
    assert t["distant"] > 0.5 and t["newer"] > 0.5                          # an unrelated note and a NEWER note survive
    assert "old1" not in [m.text for m in s.retrieve(near, k=5)] and "newer" in [m.text for m in s.retrieve(near, k=5)]


def test_plain_memory_only_drops_the_one_proven_stale_note():
    s = StaticMemoryStore()
    v = np.array([1.0, 0.0])
    s.add("a", v, correct=True); s.add("b", v, correct=True)
    s.mark_stale(s.items[0])
    assert [m.text for m in s.items] == ["b"]


def test_agent_detects_the_rule_change_after_a_failure_and_never_fires_on_a_correct_answer(fake_llm):
    cfg, dom = _domain()
    d = cfg.to_dict()
    d["agent"] = {**d["agent"], "domain": "code", "curriculum": "fixed", "reverify_stale": True}
    cfg = Config(d)
    dom = get_domain("code", cfg)
    task = next(t for t in dom.load_tasks(cfg, 58, 0) if t["phase"] == "B" and t["entry_point"] == "format_date")
    from conftest import WordHashEmbedder
    emb = WordHashEmbedder()
    store = AdaptiveTrustStore(enable_forgetting=False, enable_compression=False)
    a = build_agent(cfg, store, emb, dom, [task], 0)
    store.add(ex(OLD), emb.embed(task["question"]), correct=True)
    fake_llm(lambda p: f"```python\n{OLD}\n```")           # the model follows the stale note and fails the new tests
    r = a.step(task)
    assert not r.correct and r.stale_marked == 1 and store.items[0].trust == 0.0
    # a correct answer triggers no re-verification
    store2 = AdaptiveTrustStore(enable_forgetting=False, enable_compression=False)
    a2 = build_agent(cfg, store2, emb, dom, [task], 0)
    store2.add(ex(OLD), emb.embed(task["question"]), correct=True)
    fake_llm(lambda p: f"```python\n{NEW}\n```")
    r2 = a2.step(task)
    assert r2.correct and r2.stale_marked == 0 and store2.items[0].trust > 0.5
