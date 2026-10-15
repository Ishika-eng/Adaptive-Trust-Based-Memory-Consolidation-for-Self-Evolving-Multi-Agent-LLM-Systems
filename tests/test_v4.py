import numpy as np

import consolidate as C
from stores import AdaptiveTrustStore

EV = ("Task: Write apply_discount\nA solution to this task FAILED the project's check: "
      "apply_discount(1999, 10) -> expected 1799 (int) but got 1799.1 (float)\nOutcome: INCORRECT")
EX = ("Task: Write add_gst\nVerified solution (it passes the project's check; an earlier attempt failed with: "
      "add_gst(1000, 5) -> expected 1050 (int) but got 1050.0 (float)):\ndef add_gst(a, r):\n    return a + a * r // 100\nOutcome: CORRECT")


def test_signatures_and_kinds():
    assert C.note_signatures(EV) == {("int", "float")} == C.note_signatures(EX)
    assert C.detail_signature("AssertionError: f(1): expected None, got 3") == {("NoneType", "int")}
    assert C.detail_signature("ZeroDivisionError: division by zero") == {("exception", "ZeroDivisionError")}
    assert ("exception", "KeyError") in C.note_signatures("A solution FAILED the project's check: KeyError: 'a'")
    assert [C.note_kind(t) for t in (EV, EX, C.build_card([EV]), "Task: x\nSolution:\ncode\nOutcome: CORRECT")] == ["evidence", "example", "card", "solved"]


def _store():
    return AdaptiveTrustStore(enable_forgetting=False, enable_compression=False, utility_rank=True, hit_boost=False)


def test_the_store_learns_that_examples_help_more_than_bare_evidence():
    s = _store()
    v = np.array([1.0, 0.0])
    s.add(EV, v, correct=False, claim_verified=True)
    s.add(EX, v, correct=True)
    ev, ex = s.items
    for _ in range(6):                      # examples keep coinciding with passes, evidence with failures
        s.record_outcome([ex], True)
        s.record_outcome([ev], False)
    assert s.utility(ex) > 0.7 > 0.3 > s.utility(ev)
    assert s.retrieve(v, k=1)[0] is ex      # equal similarity: the helpful kind now ranks first


def test_new_notes_inherit_their_kinds_track_record():
    s = _store()
    v = np.array([1.0, 0.0])
    s.add(EV, v, correct=False, claim_verified=True)
    for _ in range(8):
        s.record_outcome([s.items[0]], False)
    fresh = AdaptiveTrustStore(enable_forgetting=False, enable_compression=False, utility_rank=True)
    fresh.kind_stats = dict(s.kind_stats)
    fresh.add(EV.replace("apply_discount", "tip_amount"), v, correct=False, claim_verified=True)
    assert fresh.utility(fresh.items[0]) < 0.3            # no history of its own yet, but its kind has failed every time


def test_retry_retrieval_prefers_notes_about_the_same_kind_of_failure():
    s = _store()
    near, other = np.array([1.0, 0.0]), np.array([0.9, 0.4])
    s.add("Task: a\nA solution to this task FAILED the project's check: f(1) -> expected 'x' (str) but got 'y' (str)\nOutcome: INCORRECT", near, correct=False, claim_verified=True)
    s.add(EX, other, correct=True)
    got = s.retrieve_for_repair(near, {("int", "float")}, k=1)
    assert got[0].text == EX                     # less similar to the question, but about the same kind of failure


def test_abstention_by_kind_in_both_stores():
    from stores import StaticMemoryStore
    v = np.array([1.0, 0.0])
    for store in (StaticMemoryStore(show_kinds=["example", "solved"]),
                  AdaptiveTrustStore(enable_forgetting=False, enable_compression=False, show_kinds=["example", "solved"])):
        store.add(EV, v, correct=False, claim_verified=True)          # bare evidence: abstained from
        assert store.retrieve(v, k=3) == []
        store.add(EX, v, correct=True)
        got = store.retrieve(v, k=3)
        assert [m.text for m in got] == [EX]
    plain = StaticMemoryStore()
    plain.add(EV, v, correct=False)
    assert len(plain.retrieve(v, k=3)) == 1
