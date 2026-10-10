import importlib
import sys

import pytest
from fastapi.testclient import TestClient

import db
import embeddings
import llm
from conftest import WordHashEmbedder


@pytest.fixture
def api(tmp_path, monkeypatch, fake_llm):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "api.db")
    monkeypatch.setattr(embeddings, "make_embedder", lambda cfg, corpus=None: WordHashEmbedder())
    sys.modules.pop("main", None)
    main = importlib.import_module("main")
    gold = {t["question"]: t["gold"] for t in main.TASKS}
    code_pool = {t["id"]: t for t in main.ASSISTANTS["code"].pool}

    def script(prompt):
        if "Problem: " in prompt:
            q = prompt.split("Problem: ", 1)[1].strip()
            return f"#### {gold[q]}" + ("\nUSED: none" if "Past experience" in prompt else "")
        if "Break the following task" in prompt:
            return "1. think\n2. write the function"
        if "previous attempt" in prompt:
            return "Check the edge cases."
        if "Python programming task" in prompt:
            tid = next(i for i, t in code_pool.items() if t["question"] in prompt)
            t = code_pool[tid]
            body = t["reference"] if t["dataset"] == "mbpp" else t["question"] + t["reference"]
            if "FAIL" in script.mode:
                body = "def nothing():\n    return None"
            return f"```python\n{body}\n```"
        return "SUMMARY"
    script.mode = ""
    fake_llm(script)
    client = TestClient(main.app)
    client.script, client.main = script, main
    return client


def test_side_by_side_solve_poison_state_reset(api):
    assert api.get("/api/state").json()["static_memory_size"] == 0
    assert api.post("/api/poison").json()["seeded"] > 0
    assert api.post("/api/poison").json()["already_poisoned"] is True
    r = api.post("/api/solve", json={"task_id": 0}).json()
    assert r["static"]["correct"] and r["atmc"]["correct"]
    assert {"retrieved", "memory_size", "pred", "gold"} <= r["atmc"].keys()
    trusts = {x["trust"] for x in r["atmc"]["retrieved"] if x["is_poison"]}
    assert trusts == {0.35}                                    # ATMC distrusts planted notes at once
    assert {x["trust"] for x in r["static"]["retrieved"] if x["is_poison"]} == {0.5}
    api.post("/api/reset")
    assert api.get("/api/state").json()["static_memory_size"] == 0


def test_assistant_runs_a_full_code_step_and_the_agent_chooses_its_own_task(api):
    r = api.post("/api/agent/step", json={"domain": "code"}).json()
    assert r["chosen_by"] == "weakness"
    assert r["plan"] and r["verdict"]["correct"] and r["verdict"]["stage"] == "ok"
    assert "def " in r["answer"] and r["cost"]["tokens_in"] > 0
    s = r["summary"]
    assert s["steps"] == 1 and s["passed"] == 1 and s["memory_size"] == 1 and s["isolation"] in ("sandbox-exec", "process-limits-only")
    assert s["options"] == {"planner": True, "reflection": True, "curriculum": "weakness"}


def test_failed_attempt_is_reflected_on_and_stored_as_a_low_trust_warning(api):
    api.script.mode = "FAIL"
    r = api.post("/api/agent/step", json={"domain": "code"}).json()
    assert not r["verdict"]["correct"] and r["lesson"] == "Check the edge cases."
    mem = api.get("/api/memory/code").json()
    assert len(mem) == 1 and mem[0]["trust"] == 0.35 and mem[0]["correct"] is False


def test_user_can_pick_a_task_and_the_agent_never_repeats_one(api):
    pool = api.get("/api/agent/tasks?domain=code").json()
    first = pool[0]["id"]
    api.post("/api/agent/step", json={"domain": "code", "task_id": first})
    after = api.get("/api/agent/tasks?domain=code").json()
    assert next(t for t in after if t["id"] == first)["done"] is True
    for _ in range(3):
        api.post("/api/agent/step", json={"domain": "code"})
    hist = api.get("/api/agent/state?domain=code").json()["history"]
    ids = [h["task_id"] for h in hist]
    assert len(ids) == len(set(ids)) == 4


def test_unknown_domain_and_task_give_clean_errors(api):
    assert api.get("/api/agent/state?domain=poetry").status_code == 404
    assert api.post("/api/agent/step", json={"domain": "code", "task_id": "nope"}).status_code == 404
    assert api.get("/api/memory/nonsense").status_code == 404


def test_quota_exhaustion_becomes_a_429_and_changes_nothing(api, fake_llm):
    def boom(prompt):
        raise llm.QuotaExhausted("daily")
    fake_llm(boom)
    r = api.post("/api/agent/step", json={"domain": "code"})
    assert r.status_code == 429 and "quota" in r.json()["detail"].lower()
    assert api.get("/api/agent/state?domain=code").json()["steps"] == 0


def test_assistant_poison_is_planted_and_distrusted(api):
    assert api.post("/api/agent/poison", json={"domain": "code"}).json()["seeded"] > 0
    mem = api.get("/api/memory/code").json()
    assert mem and all(m["is_poison"] and m["trust"] == 0.35 for m in mem)
    assert api.post("/api/agent/poison", json={"domain": "code"}).json()["already_poisoned"] is True


def test_assistant_state_survives_a_restart(api, tmp_path, monkeypatch):
    api.post("/api/agent/step", json={"domain": "code"})
    api.post("/api/agent/step", json={"domain": "code"})
    sys.modules.pop("main", None)
    main2 = importlib.import_module("main")                  # a fresh process, same database
    a = main2.ASSISTANTS["code"]
    assert a.store.size() == 2 and len(a.history) == 2
    nxt = a.next_task()
    assert nxt["id"] not in a.done_ids                       # the curriculum remembers what it has done


def test_reset_clears_assistant_memory_and_history(api):
    api.post("/api/agent/step", json={"domain": "code"})
    api.post("/api/agent/reset", json={"domain": "code"})
    s = api.get("/api/agent/state?domain=code").json()
    assert s["steps"] == 0 and s["memory_size"] == 0


def test_eval_endpoint_lists_and_reports(api, tmp_path, monkeypatch):
    import json, report
    monkeypatch.setattr(report, "EVAL_DIR", tmp_path)
    (tmp_path / "demo__math__a__seed1.json").write_text(json.dumps(
        {"condition": "a", "seed": 1, "n": 2, "summary": {"pass_rate": 0.5, "tokens_in": 10, "mean_retrieval_ms": 1,
         "poison_exposure_after_warmup": 0, "poison_used": 0, "final_memory_size": 1, "heldout_pass_rate": None},
         "steps": [{"task_id": "x", "correct": True}, {"task_id": "y", "correct": False}]}))
    assert api.get("/api/eval").json() == [{"experiment": "demo", "domain": "math"}]
    rep = api.get("/api/eval?experiment=demo&domain=math").json()
    assert rep["conditions"][0]["pass_rate"]["mean"] == 0.5
