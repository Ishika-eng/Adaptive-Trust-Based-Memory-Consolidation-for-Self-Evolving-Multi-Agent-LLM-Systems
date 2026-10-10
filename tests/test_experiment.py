import json

import pytest

import experiment
import llm
from domains import get_domain


@pytest.fixture
def setup(cfg, embedder, fake_llm, tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "OUT_DIR", tmp_path)
    domain = get_domain("math", cfg)
    gold = {t["question"]: t["gold"] for t in domain.load_tasks(cfg, 200, 1)}

    def script(prompt):
        if "Problem: " in prompt:
            q = prompt.split("Problem: ", 1)[1].strip()
            return f"#### {gold[q]}" + ("\nUSED: 1" if "Past experience" in prompt else "")
        return "SUMMARY"
    fake = fake_llm(script)
    return cfg, embedder, domain, fake, tmp_path


def run(setup, preset, cname, seed=1, n=6, heldout=0, force=False):
    cfg, embedder, domain, _, _ = setup
    p = experiment.PRESETS[preset]
    return experiment.run_one(preset, cname, p.conditions[cname], p, seed, cfg, embedder, domain, n, heldout, force)


def test_run_writes_a_complete_result_file(setup):
    msg = run(setup, "poison", "atmc")
    f = setup[4] / "poison__math__atmc__seed1.json"
    d = json.loads(f.read_text())
    assert d["condition"] == "atmc" and d["seed"] == 1 and len(d["steps"]) == 6
    assert d["config"]["trust"]["minimum"] == 0.30 and d["embedder"] == "test:wordhash64"
    s = d["summary"]
    assert s["pass_rate"] == 1.0 and s["n_poison_seeded"] == 2 and s["llm_calls"] >= 6
    assert {"tokens_in", "tokens_out", "mean_retrieval_ms", "poison_exposure_after_warmup"} <= s.keys()
    assert msg.startswith("pass 1.00")


def test_finished_runs_are_skipped_unless_forced(setup):
    run(setup, "poison", "static")
    calls = setup[3].prompts[:]
    assert run(setup, "poison", "static").startswith("skipped")
    assert setup[3].prompts == calls                        # no new LLM calls
    assert not run(setup, "poison", "static", force=True).startswith("skipped")


def test_heldout_phase_is_frozen_and_scored(setup):
    cfg, embedder, domain, fake, out = setup
    run(setup, "baselines", "atmc", n=5, heldout=3)
    d = json.loads((out / "baselines__math__atmc__seed1.json").read_text())
    assert d["summary"]["heldout_pass_rate"] == 1.0 and d["heldout"] == 3


def test_baselines_poison_flag_and_memory(setup):
    run(setup, "baselines_poison", "no_memory")
    d = json.loads((setup[4] / "baselines_poison__math__no_memory__seed1.json").read_text())
    assert d["summary"]["final_memory_size"] == 0 and d["summary"]["poison_exposure"] == 0


def test_fixed_order_conditions_see_the_same_sequence(setup):
    run(setup, "poison", "static"); run(setup, "poison", "atmc")
    a = json.loads((setup[4] / "poison__math__static__seed1.json").read_text())["steps"]
    b = json.loads((setup[4] / "poison__math__atmc__seed1.json").read_text())["steps"]
    assert [s["task_id"] for s in a] == [s["task_id"] for s in b]


def test_condition_overrides_do_not_leak_between_conditions(setup):
    cfg = setup[0]
    c = experiment.condition_cfg(cfg, experiment.PRESETS["agent"].conditions["atmc_planner"])
    assert c.agent.use_planner and c.agent.curriculum == "fixed"
    assert not cfg.agent.use_planner                      # original config untouched
    w = experiment.condition_cfg(cfg, experiment.PRESETS["agent"].conditions["atmc_weakness_choice"])
    assert w.agent.curriculum == "weakness"


def test_quota_exhaustion_aborts_without_writing_a_partial_file(cfg, embedder, fake_llm, tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "OUT_DIR", tmp_path)
    def boom(prompt):
        raise llm.QuotaExhausted("daily")
    fake_llm(boom)
    p = experiment.PRESETS["poison"]
    with pytest.raises(llm.QuotaExhausted):
        experiment.run_one("poison", "atmc", p.conditions["atmc"], p, 1, cfg, embedder, get_domain("math", cfg), 4, 0, False)
    assert list(tmp_path.iterdir()) == []


def test_every_preset_references_valid_store_kinds():
    from factory import STORE_KINDS
    for name, p in experiment.PRESETS.items():
        for c in p.conditions.values():
            assert c.kind in STORE_KINDS, name
