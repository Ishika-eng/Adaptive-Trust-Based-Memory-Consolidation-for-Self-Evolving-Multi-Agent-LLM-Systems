import json

import pytest

import report


def test_mcnemar_exact_known_values():
    assert report.mcnemar_exact(0, 0) == 1.0
    assert report.mcnemar_exact(5, 5) == 1.0
    assert report.mcnemar_exact(0, 6) == pytest.approx(2 * (1 / 64))        # 0.03125
    assert report.mcnemar_exact(1, 9) == pytest.approx(2 * 11 / 1024)


def test_paired_comparison_detects_a_real_difference():
    pairs = [(False, True)] * 12 + [(True, True)] * 20 + [(False, False)] * 8    # B fixes 12, breaks none
    c = report.paired_comparison(pairs)
    assert c["n"] == 40 and c["diff"] == pytest.approx(0.3)
    assert c["ci95"][0] > 0 and c["p_mcnemar"] < 0.01 and (c["b_better"], c["a_better"]) == (12, 0)


def test_paired_comparison_of_identical_systems_is_null():
    c = report.paired_comparison([(True, True)] * 30 + [(False, False)] * 10)
    assert c["diff"] == 0 and c["p_mcnemar"] == 1.0 and c["ci95"][0] <= 0 <= c["ci95"][1]


def write(tmp_path, cond, seed, outcomes, **summary):
    steps = [{"task_id": f"t{i}", "correct": ok} for i, ok in enumerate(outcomes)]
    base = {"pass_rate": sum(outcomes) / len(outcomes), "tokens_in": 100 * len(outcomes), "mean_retrieval_ms": 1.0,
            "poison_exposure_after_warmup": 0, "poison_used": 0, "final_memory_size": 3, "heldout_pass_rate": None}
    (tmp_path / f"x__math__{cond}__seed{seed}.json").write_text(json.dumps(
        {"condition": cond, "seed": seed, "n": len(outcomes), "summary": {**base, **summary}, "steps": steps}))


def test_build_report_aggregates_seeds_and_pairs_tasks(tmp_path):
    for seed in (1, 2):
        write(tmp_path, "a", seed, [True, False, False, False])
        write(tmp_path, "b", seed, [True, True, True, False])
    rep = report.build_report("x", "math", ref="a", eval_dir=tmp_path)
    rows = {r["condition"]: r for r in rep["conditions"]}
    assert rows["a"]["pass_rate"]["mean"] == 0.25 and rows["b"]["pass_rate"]["mean"] == 0.75
    assert rows["b"]["tokens_in"]["mean"] == 100.0                      # per task
    comp = rep["comparisons"][0]["in_run"]
    assert comp["n"] == 8 and comp["diff"] == pytest.approx(0.5) and comp["b_better"] == 4
    assert "| b | 2 |" in report.render(rep)


def test_report_with_no_runs_is_graceful(tmp_path):
    assert report.render(report.build_report("nothing", "math", eval_dir=tmp_path)) == "no runs found"


def test_missing_metrics_serialise_as_null_not_nan(tmp_path):
    write(tmp_path, "a", 1, [True, False])
    rep = report.build_report("x", "math", eval_dir=tmp_path)
    assert rep["conditions"][0]["heldout_pass_rate"] == {"mean": None, "sd": None}
    json.dumps(rep, allow_nan=False)                      # raises if NaN slipped through
