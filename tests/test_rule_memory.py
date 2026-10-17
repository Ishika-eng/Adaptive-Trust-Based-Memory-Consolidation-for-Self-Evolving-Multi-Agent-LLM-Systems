"""Rule memory: one slot per rule, bounded, overwritten when the rule changes."""
import numpy as np

import consolidate as C
from stores import AdaptiveTrustStore


def ev(task, call, want, wt, got, gt):
    return (f"Task: {task}\nA solution to this task FAILED the project's check: {call} -> expected {want} ({wt}) but got {got} ({gt})"
            f"\nOutcome: INCORRECT")


def ex(task, code):
    return f"Task: {task}\nVerified solution (it passes the project's check; an earlier attempt failed with: x):\n{code}\nOutcome: CORRECT"


def solved(task, code):
    return f"Task: {task}\nSolution:\n{code}\nOutcome: CORRECT"


def store(**kw):
    return AdaptiveTrustStore(enable_forgetting=False, enable_compression=False, rule_memory=True, **kw)


NEAR, FAR = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
E1 = ev("Write apply_discount", "apply_discount(1999, 10)", "1799", "int", "1799.1", "float")
E2 = ev("Write add_gst", "add_gst(1000, 5)", "1050", "int", "1050.0", "float")


def test_extract_solution_and_render_and_kind():
    assert C.extract_solution(ex("T", "def f():\n    return 1")) == ("T", "def f():\n    return 1")
    assert C.extract_solution(solved("T", "code")) == ("T", "code")
    assert C.extract_solution(E1) is None
    text = C.render_slot([("Write f", "f(1) -> expected 5 (int) but got 5.0 (float)")], [("Write f", "def f(): pass")], 3)
    assert text.startswith("Rule memory") and C.note_kind(text) == "rule" and "Pattern:" in text and "def f(): pass" in text


def test_similar_evidence_and_solutions_share_one_slot_and_the_slot_is_bounded():
    s = store()
    s.add(E1, NEAR, correct=False, claim_verified=True)
    s.add(E2, NEAR * 0.99 + np.array([0, 0.05, 0]), correct=False, claim_verified=True)
    for i in range(4):
        s.add(ex(f"Write g{i}", f"def g{i}():\n    return {i}"), NEAR, correct=True)
    assert s.size() == 1
    sl = s.items[0].slot
    assert len(sl["evidence"]) == 2 and len(sl["solutions"]) == 2                  # solutions bounded: the latest two survive
    assert [c for _, c in sl["solutions"]] == ["def g2():\n    return 2", "def g3():\n    return 3"]
    assert "Rule memory (from 6 earlier tasks" in s.items[0].text
    assert s.solutions_of(s.items[0]) == [c for _, c in sl["solutions"]]


def test_unrelated_or_differently_failing_notes_get_their_own_slot():
    s = store()
    s.add(E1, NEAR, correct=False, claim_verified=True)
    s.add(E1.replace("apply_discount", "x"), FAR, correct=False, claim_verified=True)           # far away in meaning
    other_kind = ev("Write h", "h(1)", "'a'", "str", "'b'", "str")
    s.add(other_kind, NEAR, correct=False, claim_verified=True)                               # near, but a different failure type
    assert s.size() == 3


def test_a_stale_slot_is_emptied_its_older_neighbours_demoted_and_new_evidence_refounds_it():
    s = store()
    s.add(E1, NEAR, correct=False, claim_verified=True)
    s.add(ex("Write t", "def t(): pass"), NEAR, correct=True)
    old_neighbour = s._slot_new(NEAR * 0.98 + FAR * 0.1, 0.55)                                  # another older slot of the same family
    old_neighbour.created_at_step = 0
    slot = s.items[0]
    hit = s.mark_stale(slot)
    assert slot.trust == 0.0 and slot.slot["evidence"] == [] and slot.slot["solutions"] == [] and slot.text == ""
    assert hit == 1 and old_neighbour.trust < s.t_min
    assert s.retrieve(NEAR, k=3) == []                                                        # nothing obsolete is shown
    s.add(E2, NEAR, correct=False, claim_verified=True)                                       # new-rule evidence arrives
    got = s.retrieve(NEAR, k=3)
    assert len(got) == 1 and got[0] is slot and "add_gst" in got[0].text and "apply_discount" not in got[0].text


def test_off_by_default_and_other_notes_fall_through():
    plain = AdaptiveTrustStore(enable_forgetting=False, enable_compression=False)
    plain.add(E1, NEAR, correct=False, claim_verified=True)
    assert getattr(plain.items[0], "slot", None) is None
    s = store()
    s.add("Task: x\nsome free text note", NEAR, correct=True)                                  # not evidence, not a solution
    assert s.size() == 1 and getattr(s.items[0], "slot", None) is None


def test_short_view_shows_only_the_nearest_evidence_and_solution():
    s = store(rule_show_evidence=1, rule_show_solutions=1)
    q_near, q_far = np.array([1.0, 0.0, 0.0]), np.array([0.9, 0.3, 0.0])
    s.add(E1, q_far, correct=False, claim_verified=True)
    s.add(E2, q_near, correct=False, claim_verified=True)
    s.add(ex("Write far", "def far():\n    return 1"), q_far, correct=True)
    s.add(ex("Write near", "def near():\n    return 2"), q_near, correct=True)
    assert s.size() == 1 and len(s.items[0].slot["evidence"]) == 2 and len(s.items[0].slot["solutions"]) == 2   # storage unchanged
    shown = s.retrieve(q_near, k=3)[0].text
    assert "add_gst" in shown and "apply_discount" not in shown          # only the nearest evidence line
    assert "def near()" in shown and "def far()" not in shown            # only the nearest solution
    assert shown.startswith("Rule memory")


def test_evidence_only_mode_keeps_solutions_as_ordinary_notes():
    s = store(rule_mode="evidence_only")
    s.add(E1, NEAR, correct=False, claim_verified=True)
    s.add(ex("Write t", "def t():\n    return 1"), NEAR, correct=True)
    s.add(solved("Write u", "def u():\n    return 2"), NEAR, correct=True)
    slots = [m for m in s.items if getattr(m, "slot", None) is not None]
    assert len(slots) == 1 and slots[0].slot["solutions"] == [] and len(slots[0].slot["evidence"]) == 1
    assert s.size() == 3                                                   # one evidence slot plus two separate solution notes
    assert C.note_kind([m.text for m in s.items if getattr(m, "slot", None) is None][0]) in ("example", "solved")


def test_stale_slot_reset_also_clears_the_parallel_embedding_lists():
    s = store(rule_show_evidence=1, rule_show_solutions=1)
    s.add(E1, NEAR, correct=False, claim_verified=True)
    s.add(ex("Write t", "def t():\n    return 1"), NEAR, correct=True)
    slot = s.items[0]
    s.mark_stale(slot)
    assert slot.slot["ev_emb"] == [] and slot.slot["sol_emb"] == []
    s.add(E2, NEAR, correct=False, claim_verified=True)                    # refounded; the view must not crash on the rebuilt lists
    assert "add_gst" in s.retrieve(NEAR, k=3)[0].text
