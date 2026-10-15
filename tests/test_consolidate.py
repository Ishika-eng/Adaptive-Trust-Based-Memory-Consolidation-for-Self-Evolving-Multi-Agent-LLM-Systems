import numpy as np

import consolidate as C
from stores import AdaptiveTrustStore


def note(task, call, want, wt, got, gt):
    return f"Task: {task}\nA solution to this task FAILED the project's check: {call} -> expected {want} ({wt}) but got {got} ({gt})\nOutcome: INCORRECT"


A = note("Write apply_discount(price, percent)", "apply_discount(1999, 10)", "1799", "int", "1799.1", "float")
B = note("Write add_gst(amount, rate)", "add_gst(1000, 5)", "1050", "int", "1050.0", "float")


def test_card_states_the_pattern_and_keeps_every_real_check():
    card = C.build_card([A, B])
    assert card.startswith(C.CARD_HEAD) and "apply_discount(1999, 10)" in card and "add_gst(1000, 5)" in card
    assert "2 of 2 failures the checker expected type int but the solution returned type float" in card


def test_card_grows_and_is_bounded_and_ignores_duplicates():
    card = C.build_card([A, B])
    card = C.build_card([card, B])                    # same check again adds nothing
    assert card.count("Check: ") == 2
    many = [note(f"t{i}", f"f{i}(1)", str(i), "int", f"{i}.0", "float") for i in range(10)]
    assert C.build_card(many, max_lines=4).count("Check: ") == 4


def _store(**kw):
    return AdaptiveTrustStore(enable_forgetting=False, enable_compression=False, evidence_merge=True, **kw)


def test_similar_failures_fold_into_one_card_and_unrelated_ones_do_not():
    s = _store(evidence_threshold=0.9)
    near, far = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    s.add(A, near, correct=False, claim_verified=True)
    s.add(B, near * 0.99 + np.array([0, 0.05, 0]), correct=False, claim_verified=True)
    assert s.size() == 1 and s.items[0].text.startswith(C.CARD_HEAD) and s.n_cards_built == 1
    s.add(note("Write format_date", "format_date(2024, 3, 5)", "'05-03-2024'", "str", "'2024-03-05'", "str"), far, correct=False, claim_verified=True)
    assert s.size() == 2                               # a different kind of problem keeps its own note


def test_only_verified_failure_evidence_is_merged():
    s = _store(evidence_threshold=0.5)
    v = np.array([1.0, 0.0])
    s.add(A, v, correct=False, claim_verified=True)
    s.add("Task: x\nSolution:\ncode\nOutcome: CORRECT", v, correct=True)       # a success is never folded into a card
    s.add(B, v, correct=False, claim_verified=False)                            # unverified claims are not either
    assert s.size() == 3


def test_off_by_default():
    s = AdaptiveTrustStore(enable_forgetting=False, enable_compression=False)
    v = np.array([1.0, 0.0])
    s.add(A, v, correct=False, claim_verified=True); s.add(B, v, correct=False, claim_verified=True)
    assert s.size() == 2
