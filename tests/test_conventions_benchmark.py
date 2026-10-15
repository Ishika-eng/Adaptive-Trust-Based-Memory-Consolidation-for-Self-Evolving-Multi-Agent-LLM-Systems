"""The conventions benchmark must stay honest: the saved file is exactly what the builder produces, every task's tests
accept its reference solution and reject the rule-ignoring one (the builder refuses to write otherwise), and the
question text never states the hidden rule."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pilot" / "benchmarks"))
import build_conventions as bc  # noqa: E402


def _saved():
    return [json.loads(l) for l in open(ROOT / "pilot" / "data" / "conventions.jsonl")]


def test_saved_file_matches_the_builder_and_every_task_is_validated():
    assert _saved() == json.loads(json.dumps(bc.build()))      # build() runs reference + naive solutions in the sandbox


def test_shape_of_the_benchmark():
    rows = _saved()
    assert len(rows) == 110 and len({r["id"] for r in rows}) == 110
    fams = Counter(r["family"] for r in rows)
    assert len(fams) == 11 and all(n == 10 for n in fams.values())
    roles = Counter(r["role"] for r in rows)
    assert roles == {"rule": 90, "control": 10, "known_practice": 10}


def test_controls_have_no_rule_and_rule_tasks_have_one():
    for r in _saved():
        assert bool(r["rule"]) == (r["family"] != "control")
        assert r["role"] == ("control" if r["family"] == "control" else "known_practice" if r["family"] == "no_mutation" else "rule")
        assert (r["naive"] is None) == (r["family"] == "control")


def test_the_question_does_not_give_away_the_rule():
    for r in _saved():
        q = r["question"].lower()
        for banned in ("paise", "zero-pad", "dd-mm-yyyy", "lower case", "lowercase", "must not modify", "ascending", "(true, ", "[level]"):
            assert banned not in q, (r["id"], banned)
