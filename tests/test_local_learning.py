"""Pieces added for the local-model learning experiment: evidence-style notes and the noisy checker."""
import experiment as ex
from config import Config, load_config
from domains import get_domain
from domains.base import Verdict


def _domain(note_style="evidence"):
    d = load_config().to_dict()
    d["coding"]["note_style"] = note_style
    return get_domain("code", Config(d))


TASK = {"question": "Write apply_discount(price, percent)."}


def test_failed_attempt_note_keeps_typed_evidence_and_drops_the_wrong_code():
    v = Verdict(False, "wrong_answer", "AssertionError: apply_discount(1999, 10): expected 1799, got 1799.1000000000001")
    note = _domain().reflection(TASK, "raw", "return price * 0.9", v)
    assert "expected 1799 (int) but got 1799.1000000000001 (float)" in note
    assert "price * 0.9" not in note and "INCORRECT" in note


def test_full_style_and_correct_attempts_are_unchanged():
    v = Verdict(False, "wrong_answer", "AssertionError: f(1): expected 1, got 2")
    assert "Solution:" in _domain("full").reflection(TASK, "raw", "code", v)
    ok = _domain().reflection(TASK, "raw", "def f(): pass", Verdict(True, "ok"))
    assert "Solution:" in ok and "CORRECT" in ok


class _Dom:
    name = "code"

    def check(self, task, answer):
        return Verdict(answer == "right", "ok" if answer == "right" else "wrong_answer", "")


def test_noisy_checker_flips_only_wrong_answers_and_remembers_the_truth():
    nc = ex.NoisyChecker(_Dom(), 1.0, seed=1)
    assert nc.check({}, "bad").correct is True and nc.last_true_correct is False
    assert nc.check({}, "right").correct is True and nc.last_true_correct is True
    quiet = ex.NoisyChecker(_Dom(), 0.0, seed=1)
    assert quiet.check({}, "bad").correct is False


def test_noisy_checker_rate_is_roughly_p_and_reproducible():
    flips = lambda seed: [ex.NoisyChecker(_Dom(), 0.25, seed).check({}, "bad").correct for _ in range(1)]
    nc = ex.NoisyChecker(_Dom(), 0.25, seed=3)
    rate = sum(nc.check({}, "bad").correct for _ in range(2000)) / 2000
    assert 0.2 < rate < 0.3
    a = ex.NoisyChecker(_Dom(), 0.25, 5); b = ex.NoisyChecker(_Dom(), 0.25, 5)
    assert [a.check({}, "bad").correct for _ in range(50)] == [b.check({}, "bad").correct for _ in range(50)]
