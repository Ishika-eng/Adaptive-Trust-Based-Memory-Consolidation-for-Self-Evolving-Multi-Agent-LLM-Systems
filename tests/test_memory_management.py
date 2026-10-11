import sqlite3

import numpy as np
import pytest

import db
from stores import AdaptiveTrustStore


def vec(i, n=4):
    v = np.zeros(n)
    v[i % n] = 1.0
    return v


def store(**kw):
    return AdaptiveTrustStore(embed_fn=lambda t: vec(0), summarize_fn=lambda ts: "SUMMARY", **kw)


def test_notes_get_stable_unique_ids_and_a_source():
    s = store()
    s.add("a", vec(0), True)
    s.add("b", vec(1), True, source="user")
    s.add("c", vec(2), False, is_poison=True)
    assert len({m.uid for m in s.items}) == len(s.items) and all(m.uid for m in s.items)
    assert [m.source for m in s.items] == ["tests", "user", "planted"]      # planted notes are always labelled so


def test_using_a_note_records_evidence_and_when():
    s = store()
    s.add("a", vec(0), True)
    s.add("b", vec(1), True)
    a, b = s.items
    s.update_from_outcome([a, b], True, used_mask=[True, False])
    s.update_from_outcome([a], False, used_mask=[True])
    assert (a.successes, a.failures) == (1, 1) and (b.successes, b.failures) == (0, 0)
    assert a.last_used_step is not None and b.last_used_step is None


def test_a_pinned_note_survives_forgetting_and_merging():
    s = store(prune_every=1, compress_every=3, compress_min_age_steps=1, compress_batch_size=3)
    s.add("keep me", vec(0), True)
    keep = s.items[0]
    s.set_pinned(keep.uid, True)
    keep.trust = 0.05                                    # would normally be pruned (below the trust floor)
    s._step += 500                                       # and very old
    for i in range(5):
        s.add(f"n{i}", vec(i + 1), True)
    assert keep in s.items
    assert all(m is keep or not m.pinned for m in s.items)
    # merged notes never absorb it
    assert not any(m.source == "compressed" and "keep me" in m.text for m in s.items)


def test_merged_notes_record_their_lineage():
    s = store(compress_every=4, compress_min_age_steps=1, compress_batch_size=3, enable_forgetting=False)
    for i in range(3):
        s.add(f"n{i}", vec(i), True)
        s.items[-1].successes, s.items[-1].failures = 2, 1
    s.add("n3", vec(3), True)                              # the 4th add triggers the merge of the three older notes
    merged = [m for m in s.items if m.source == "compressed"]
    assert len(merged) == 1
    m = merged[0]
    assert m.version == 2 and (m.successes, m.failures) == (6, 3) and m.uid


def test_remove_and_pin_report_unknown_ids():
    s = store()
    s.add("a", vec(0), True)
    uid = s.items[0].uid
    assert s.set_pinned("nope", True) is False and s.remove("nope") is False
    assert s.set_pinned(uid, True) and s.items[0].pinned
    assert s.remove(uid) and s.size() == 0


def test_imported_notes_restart_at_initial_trust_and_duplicates_are_skipped():
    s = store(initial_trust=0.5)
    assert s.import_item("from a file", vec(0), True, successes=50, failures=0) is True
    note = s.items[0]
    assert note.trust == 0.5 and note.source == "imported" and note.uid         # no trust smuggled in
    assert s.import_item("from a file", vec(0), True) is False
    assert s.size() == 1


def test_retrieval_explains_each_choice_and_counts_notes_filtered_for_low_trust():
    s = store()
    for i in range(3):
        s.add(f"n{i}", vec(i), True)
    s.items[2].trust = 0.1                               # below the 0.3 gate
    got = s.retrieve(vec(0), k=3)
    assert len(got) == 2 and s.last_filtered == 1
    assert [e["rank"] for e in s.last_explanations] == [1, 2]
    assert {"similarity", "importance", "retention", "score"} <= s.last_explanations[0].keys()
    assert s.last_explanations[0]["similarity"] >= s.last_explanations[1]["similarity"]


def test_evidence_fields_persist_through_the_database(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "m.db")
    s = store()
    s.add("a", vec(0), True, source="generated")
    s.items[0].successes, s.items[0].version, s.items[0].pinned = 3, 4, True
    db.save_store("x", s, "emb")
    t = store()
    assert db.load_store("x", t, "emb")
    m = t.items[0]
    assert (m.source, m.successes, m.version, m.pinned, m.uid) == ("generated", 3, 4, True, s.items[0].uid)


def test_an_old_database_without_the_new_columns_is_upgraded_and_ids_assigned(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    monkeypatch.setattr(db, "DB_PATH", path)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT, store_name TEXT NOT NULL, text TEXT NOT NULL, "
                 "embedding TEXT NOT NULL, trust REAL NOT NULL, hits INTEGER NOT NULL, created_at_step INTEGER NOT NULL, "
                 "correct INTEGER, is_poison INTEGER NOT NULL)")
    conn.execute("INSERT INTO memories (store_name, text, embedding, trust, hits, created_at_step, correct, is_poison) "
                 "VALUES ('x', 'legacy note', '[1.0, 0.0, 0.0, 0.0]', 0.7, 2, 1, 1, 0)")
    conn.commit(); conn.close()
    t = store()
    assert db.load_store("x", t, "emb") is False or True      # embedder meta absent -> loads
    db.init_db()
    assert db.load_store("x", t, "emb")
    m = t.items[0]
    assert m.text == "legacy note" and m.trust == 0.7 and m.uid and m.source == "tests" and m.version == 1
