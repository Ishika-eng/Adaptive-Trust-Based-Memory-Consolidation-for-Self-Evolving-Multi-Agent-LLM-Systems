"""Learning Lab: two assistants, one task stream; the model switch refuses an unavailable local model."""
import json

import learnlab
from test_api import api  # noqa: F401  (fixture: app with a fake embedder and a scripted fake model)



def _wrong_solver(fake_llm):
    from config import ROOT
    rows = {r["question"]: r for r in (json.loads(l) for l in open(ROOT / "pilot/data/conventions.jsonl"))}

    def script(prompt):
        for q, r in rows.items():
            if q in prompt:
                code = r["naive"] or r["reference"]       # the sensible-but-rule-ignoring answer; controls get the right one
                return f"```python\n{code}\n```"
        return "SUMMARY"
    fake_llm(script)


def test_switching_to_an_unavailable_local_model_is_refused_with_a_clear_message(api, monkeypatch):
    monkeypatch.setattr(learnlab, "_ollama_up", lambda: False)
    r = api.post("/api/model", json={"mode": "local"})
    assert r.status_code == 409 and "ollama" in r.json()["detail"].lower()
    assert api.post("/api/model", json={"mode": "nonsense"}).status_code == 400


def test_a_run_gives_both_assistants_the_same_question_and_only_one_remembers(api, fake_llm, monkeypatch):
    _wrong_solver(fake_llm)
    monkeypatch.setattr(learnlab, "_ollama_up", lambda: True)
    assert api.post("/api/model", json={"mode": "local"}).json()["mode"] == "local"
    assert api.post("/api/learn/start", json={"seed": 42, "n": 12}).json()["n"] == 12
    rows = [api.post("/api/learn/step").json()["row"] for _ in range(12)]
    assert all(r["no_memory"]["notes_used"] == [] for r in rows)            # the control never recalls anything
    assert any(r["memory"]["notes_used"] for r in rows[1:])                   # the other one does, once it has notes
    first_fail = next(r for r in rows if not r["memory"]["correct"] and r["role"] == "rule")
    assert "FAILED the project's check" in first_fail["memory"]["note_saved"]  # evidence note, not the wrong code
    s = api.get("/api/learn/state").json()
    assert s["step"] == 12 and s["done"] and s["memory_size"] > 0
    assert s["no_memory"]["steps"] == s["memory"]["steps"] == 12
    assert api.post("/api/learn/step").status_code == 409                     # finished runs refuse more steps


def test_step_before_start_is_a_clear_error(api):
    learnlab.SESSION.clear()
    assert api.post("/api/learn/step").status_code == 409
