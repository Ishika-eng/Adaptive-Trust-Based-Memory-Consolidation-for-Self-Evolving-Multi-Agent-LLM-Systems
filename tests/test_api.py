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
        if "Python programming task" in prompt or "Draft solution" in prompt:
            tid = next(i for i, t in code_pool.items() if t["question"] in prompt)
            t = code_pool[tid]
            body = t["reference"] if t["dataset"] == "mbpp" else t["question"] + t["reference"]
            if "FAIL" in script.mode:
                body = "def nothing():\n    return None"
            return f"```python\n{body}\n```"
        return "SUMMARY"
    script.mode = ""
    fake_llm(script)
    main.BOOK.explore = 0.0              # live questions use the champion, so tests don't depend on chance
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


def _chat_script(api, code="def add(a, b):\n    return a + b"):
    base = api.script

    def script(prompt):
        if "careful Python coding assistant" in prompt or "Request:\nwrite add" in prompt:
            return f"Add them.\n```python\n{code}\n```"
        return base(prompt)
    script.mode = ""
    return script


def test_chat_with_tests_verifies_and_learns_immediately(api, fake_llm):
    fake_llm(_chat_script(api))
    t = api.post("/api/chat/ask", json={"prompt": "write add(a, b)", "tests": "assert add(1, 2) == 3"}).json()
    assert t["status"] == "passed" and t["code"].startswith("def add") and t["explanation"] == "Add them."
    assert api.get("/api/chat/summary").json() == {**api.get("/api/chat/summary").json(), "asked": 1, "solved": 1,
                                                  "memory_size": 1, "awaiting_feedback": 0}


def test_chat_failing_tests_are_a_verified_failure_with_a_lesson(api, fake_llm):
    fake_llm(_chat_script(api, "def add(a, b):\n    return a - b"))
    t = api.post("/api/chat/ask", json={"prompt": "write add(a, b)", "tests": "assert add(1, 2) == 3"}).json()
    assert t["status"] == "failed" and t["stage"] == "wrong_answer" and t["lesson"]


def test_chat_without_tests_waits_for_the_users_verdict_before_learning(api, fake_llm):
    fake_llm(_chat_script(api))
    t = api.post("/api/chat/ask", json={"prompt": "write add(a, b)"}).json()
    assert t["status"] == "unverified"
    assert api.get("/api/chat/summary").json()["memory_size"] == 0            # nothing learned yet
    assert api.get("/api/chat/summary").json()["awaiting_feedback"] == 1
    done = api.post("/api/chat/feedback", json={"id": t["id"], "correct": True}).json()
    assert done["feedback"]["correct"] is True
    assert api.get("/api/chat/summary").json()["memory_size"] == 1            # now it learned
    assert api.post("/api/chat/feedback", json={"id": t["id"], "correct": True}).status_code == 409


def test_chat_user_rejection_stores_a_failure_note_with_a_lesson(api, fake_llm):
    fake_llm(_chat_script(api))
    t = api.post("/api/chat/ask", json={"prompt": "write add(a, b)"}).json()
    done = api.post("/api/chat/feedback", json={"id": t["id"], "correct": False, "comment": "off by one"}).json()
    assert done["lesson"]
    notes = api.get("/api/memory/chat").json()
    assert len(notes) == 1 and "INCORRECT" in notes[0]["text"] and "off by one" in notes[0]["text"]


def test_chat_rejects_empty_and_oversized_input_and_resets(api, fake_llm):
    fake_llm(_chat_script(api))
    assert api.post("/api/chat/ask", json={"prompt": "   "}).status_code == 400
    assert api.post("/api/chat/ask", json={"prompt": "x" * 5000}).status_code == 400
    api.post("/api/chat/ask", json={"prompt": "write add(a, b)", "tests": "assert add(1, 2) == 3"})
    api.post("/api/chat/reset")
    assert api.get("/api/chat/history").json() == []
    assert api.get("/api/chat/summary").json()["memory_size"] == 0


def test_unexpected_backend_errors_come_back_as_readable_json_not_a_crash(api, fake_llm):
    def boom(prompt):
        raise RuntimeError("Gemini call failed after 4 retries: API key not valid")
    fake_llm(boom)
    r = api.post("/api/chat/ask", json={"prompt": "write add(a, b)"})
    assert r.status_code == 502 and "API key not valid" in r.json()["detail"]


def _capturing(api, code="def add(a, b):\n    return a + b"):
    prompts = []
    inner = _chat_script(api, code)

    def script(prompt):
        prompts.append(prompt)
        return inner(prompt)
    script.mode = ""
    return script, prompts


def test_followup_sees_the_earlier_exchange_and_a_new_chat_does_not(api, fake_llm):
    script, prompts = _capturing(api)
    fake_llm(script)
    api.post("/api/chat/ask", json={"prompt": "write add(a, b)", "tests": "assert add(1, 2) == 3"})
    t2 = api.post("/api/chat/ask", json={"prompt": "now make it accept three numbers"}).json()
    solve = [p for p in prompts if "careful Python coding assistant" in p][-1]
    assert t2["followup"] is True
    assert "Earlier in this conversation" in solve and "return a + b" in solve and "the tests passed" in solve
    assert "now make it accept three numbers" in solve

    assert api.post("/api/chat/new").json()["conversation"] == 2
    t3 = api.post("/api/chat/ask", json={"prompt": "write a sorter"}).json()
    last = [p for p in prompts if "careful Python coding assistant" in p][-1]
    assert t3["followup"] is False and "Earlier in this conversation" not in last and t3["conversation"] == 2


def test_followup_carries_the_users_rejection_into_the_context(api, fake_llm):
    script, prompts = _capturing(api)
    fake_llm(script)
    t = api.post("/api/chat/ask", json={"prompt": "write add(a, b)"}).json()
    api.post("/api/chat/feedback", json={"id": t["id"], "correct": False, "comment": "must support floats"})
    api.post("/api/chat/ask", json={"prompt": "fix it"})
    solve = [p for p in prompts if "careful Python coding assistant" in p][-1]
    assert "did NOT work: must support floats" in solve


def test_assistant_suggests_a_task_explains_why_and_practises_it_verified(api, fake_llm):
    fake_llm(_chat_script(api))
    s = api.get("/api/chat/suggest").json()
    assert s["id"] and "haven't made a mistake" in s["reason"]
    assert api.get("/api/chat/suggest").json()["id"] == s["id"]            # stable until used or refreshed

    turn = api.post("/api/chat/practise", json={"task_id": s["id"]}).json()
    assert turn["practice"] is True and turn["status"] == "passed"         # real tests, verified
    summary = api.get("/api/chat/summary").json()
    assert summary["practised"] == 1 and summary["practised_passed"] == 1 and summary["asked"] == 0
    assert summary["memory_size"] == 1                                      # it learned from it
    assert api.post("/api/chat/practise", json={"task_id": s["id"]}).status_code == 404   # no repeats
    assert api.get("/api/chat/suggest").json()["id"] != s["id"]


def test_suggestion_after_a_failure_points_at_that_failure(api, fake_llm):
    fake_llm(_chat_script(api, "def add(a, b):\n    return a - b"))
    api.post("/api/chat/ask", json={"prompt": "write add(a, b)", "tests": "assert add(1, 2) == 3"})
    main = api.main
    main.CHAT["agent"].curriculum.explore = 0.0                             # force the weakness branch
    s = api.get("/api/chat/suggest?refresh=true").json()
    assert "close to a task I got wrong" in s["reason"] and "write add(a, b)" in s["reason"]


def _autocheck_script(api, answers):
    """Chat solutions come from `answers` in order; test generation returns three asserts."""
    base = _chat_script(api)
    queue = list(answers)

    def script(prompt):
        if "Python `assert` statements" in prompt:
            return "```python\nassert add(1, 2) == 3\nassert add(0, 4) == 4\nassert add(-1, 1) == 0\n```"
        if "careful Python coding assistant" in prompt:
            return f"Adds them.\n```python\n{queue.pop(0)}\n```"
        return base(prompt)
    script.mode = ""
    return script


GOOD_A = "def add(a, b):\n    return a + b"
GOOD_B = "def add(a, b):\n    return b + a"
GOOD_C = "def add(a, b):\n    return sum([a, b])"
BAD_A = "def add(a, b):\n    return a - b"


def test_chat_without_tests_is_auto_verified_and_memory_learns_at_half_strength(api, fake_llm):
    fake_llm(_autocheck_script(api, [GOOD_A, GOOD_B, GOOD_C]))
    t = api.post("/api/chat/ask", json={"prompt": "write add(a, b)"}).json()
    assert t["status"] == "auto_verified" and t["autocheck"]["status"] == "verified" and t["autocheck"]["n_reliable"] == 3
    assert api.get("/api/chat/summary").json()["memory_size"] == 1      # learned with no user input
    assert api.get("/api/chat/summary").json()["solved"] == 1


def test_chat_switches_to_the_alternative_that_passes_the_agreed_tests(api, fake_llm):
    fake_llm(_autocheck_script(api, [BAD_A, GOOD_B, GOOD_C]))
    t = api.post("/api/chat/ask", json={"prompt": "write add(a, b)"}).json()
    assert t["status"] == "auto_verified" and t["autocheck"]["status"] == "switched"
    assert "b + a" in t["code"] or "sum(" in t["code"]


def test_inconclusive_autocheck_falls_back_to_asking_the_user(api, fake_llm):
    fake_llm(_autocheck_script(api, [GOOD_A, GOOD_A, GOOD_A]))            # identical code is not independent evidence
    t = api.post("/api/chat/ask", json={"prompt": "write add(a, b)"}).json()
    assert t["status"] == "unverified" and t["autocheck"]["status"] == "inconclusive"
    assert api.get("/api/chat/summary").json()["memory_size"] == 0


def test_autocheck_can_be_turned_off_per_question(api, fake_llm):
    fake_llm(_autocheck_script(api, [GOOD_A]))
    t = api.post("/api/chat/ask", json={"prompt": "write add(a, b)", "autocheck": False}).json()
    assert t["status"] == "unverified" and t["autocheck"] is None


def test_disputing_an_auto_verified_answer_retires_its_note_and_stores_a_failure(api, fake_llm):
    fake_llm(_autocheck_script(api, [GOOD_A, GOOD_B, GOOD_C]))
    t = api.post("/api/chat/ask", json={"prompt": "write add(a, b)"}).json()
    d = api.post("/api/chat/dispute", json={"id": t["id"], "comment": "wrong for floats"}).json()
    assert d["status"] == "disputed"
    notes = api.get("/api/memory/chat").json()
    assert len(notes) == 2
    assert [n["trust"] for n in notes if n["correct"] is False and n["trust"] == 0.0]          # the retired note
    assert any("wrong for floats" in n["text"] for n in notes)
    assert api.post("/api/chat/dispute", json={"id": t["id"]}).status_code == 409
    assert api.post("/api/chat/dispute", json={"id": "nope"}).status_code == 404


def test_strategy_is_chosen_recorded_and_resolved_when_the_outcome_is_known(api, fake_llm):
    fake_llm(_chat_script(api))
    t = api.post("/api/chat/ask", json={"prompt": "write add(a, b)", "tests": "assert add(1, 2) == 3"}).json()
    assert t["strategy"]["name"] in ("plan_first", "direct", "revise") and t["strategy"]["mode"] == "serve"
    board = api.get("/api/chat/strategies").json()
    assert board["total"] == 1 and board["events"][0]["success"] is True            # tests decided it at once

    # no tests, auto-check off -> the outcome is unknown until the user answers
    u = api.post("/api/chat/ask", json={"prompt": "write add(a, b)", "autocheck": False}).json()
    assert api.get("/api/chat/strategies").json()["events"][0]["success"] is None
    api.post("/api/chat/feedback", json={"id": u["id"], "correct": False})
    assert api.get("/api/chat/strategies").json()["events"][0]["success"] is False


def test_practising_explores_and_its_verified_result_feeds_the_scoreboard(api, fake_llm):
    fake_llm(_chat_script(api))
    s = api.get("/api/chat/suggest").json()
    turn = api.post("/api/chat/practise", json={"task_id": s["id"]}).json()
    assert turn["strategy"]["mode"] == "practise" and "least" in turn["strategy"]["reason"]
    ev = api.get("/api/chat/strategies").json()["events"][0]
    assert ev["mode"] == "practise" and ev["success"] is True


def test_disputing_an_auto_verified_answer_turns_its_strategy_win_into_a_loss(api, fake_llm):
    fake_llm(_autocheck_script(api, [GOOD_A, GOOD_B, GOOD_C]))
    t = api.post("/api/chat/ask", json={"prompt": "write add(a, b)"}).json()
    assert api.get("/api/chat/strategies").json()["events"][0]["success"] is True
    api.post("/api/chat/dispute", json={"id": t["id"]})
    assert api.get("/api/chat/strategies").json()["events"][0]["success"] is False


def test_strategy_evidence_survives_a_restart_and_clears_on_reset(api, fake_llm):
    fake_llm(_chat_script(api))
    api.post("/api/chat/ask", json={"prompt": "write add(a, b)", "tests": "assert add(1, 2) == 3"})
    assert len(api.main.db.load_events()) == 1
    api.post("/api/chat/reset")
    assert api.main.db.load_events() == [] and api.get("/api/chat/strategies").json()["total"] == 0


def test_memory_listing_exposes_evidence_and_retrieval_explains_itself(api, fake_llm):
    fake_llm(_chat_script(api))
    api.post("/api/chat/ask", json={"prompt": "write add(a, b)", "tests": "assert add(1, 2) == 3"})
    t = api.post("/api/chat/ask", json={"prompt": "write add(a, b) again", "tests": "assert add(1, 2) == 3"}).json()
    note = api.get("/api/memory/chat").json()[0]
    assert {"uid", "source", "successes", "failures", "version", "pinned", "last_used_step", "gated"} <= note.keys()
    assert note["source"] == "tests" and note["uid"]
    first = t["retrieved"][0]
    assert first["why"]["rank"] == 1 and "similarity" in first["why"] and t["filtered"] == 0


def test_pin_delete_export_and_import_roundtrip(api, fake_llm):
    fake_llm(_chat_script(api))
    api.post("/api/chat/ask", json={"prompt": "write add(a, b)", "tests": "assert add(1, 2) == 3"})
    uid = api.get("/api/memory/chat").json()[0]["uid"]
    assert api.post("/api/memory/chat/pin", json={"uid": uid, "pinned": True}).status_code == 200
    assert api.get("/api/memory/chat").json()[0]["pinned"] is True
    assert api.post("/api/memory/chat/pin", json={"uid": "zzz"}).status_code == 404

    export = api.get("/api/memory/chat/export").json()
    assert export["format"] == "atmc-memory/1" and len(export["items"]) == 1

    assert api.delete(f"/api/memory/chat/{uid}").json()["memory_size"] == 0
    assert api.delete(f"/api/memory/chat/{uid}").status_code == 404

    export["items"][0]["trust"] = 1.0                      # a hostile file claiming maximum trust
    done = api.post("/api/memory/chat/import", json=export).json()
    assert done["added"] == 1 and done["memory_size"] == 1
    imported = api.get("/api/memory/chat").json()[0]
    assert imported["source"] == "imported" and imported["trust"] == 0.5 and imported["pinned"] is False
    assert api.post("/api/memory/chat/import", json=export).json()["skipped_duplicates"] == 1


def test_import_rejects_foreign_embedders_and_malformed_notes(api, fake_llm):
    bad_embedder = {"embedder": "other:model", "items": []}
    assert api.post("/api/memory/chat/import", json=bad_embedder).status_code == 400
    wrong_dim = {"embedder": api.main.EMBEDDER.name, "items": [{"text": "x", "embedding": [0.1, 0.2]}]}
    assert api.post("/api/memory/chat/import", json=wrong_dim).status_code == 400
    too_many = {"embedder": api.main.EMBEDDER.name, "items": [{"text": "x", "embedding": [0.0]}] * 2001}
    assert api.post("/api/memory/chat/import", json=too_many).status_code == 400
