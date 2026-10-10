import numpy as np
import pytest

import db
from config import load_config
from memory import MemoryItem
from stores import StaticMemoryStore


@pytest.mark.parametrize("bad", [
    {"composite": {"w_trust": 0.5}},
    {"trust": {"penalty": 0.01}},
    {"trust": {"minimum": 1.5}},
    {"embedding": {"backend": "nope"}},
    {"retrieval": {"top_k": 0}},
    {"agent": {"domain": "poetry"}},
    {"agent": {"curriculum": "chaos"}},
    {"agent": {"curriculum_explore": 2.0}},
])
def test_invalid_config_is_rejected(bad):
    with pytest.raises(ValueError):
        load_config(overrides=bad)


def test_override_changes_only_what_was_asked():
    c = load_config(overrides={"retrieval": {"top_k": 5}})
    assert c.retrieval.top_k == 5 and c.trust.minimum == 0.30


def test_default_config_matches_documented_values(cfg):
    assert (cfg.trust.reward, cfg.trust.penalty, cfg.trust.minimum) == (0.05, 0.15, 0.30)
    assert (cfg.composite.w_trust, cfg.composite.w_importance, cfg.composite.w_decay) == (0.4, 0.4, 0.2)
    assert cfg.model.temperature == 0.0


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    db.init_db()


def _store():
    s = StaticMemoryStore()
    s.items = [MemoryItem(text="x", embedding=np.arange(4.0), trust=0.7, hits=3, created_at_step=5,
                          correct=False, is_poison=True)]
    return s


def test_roundtrip_preserves_everything(tmp_db):
    db.save_store("static", _store(), "emb:A")
    fresh = StaticMemoryStore()
    assert db.load_store("static", fresh, "emb:A")
    m = fresh.items[0]
    assert (m.text, m.trust, m.hits, m.created_at_step, m.correct, m.is_poison) == ("x", 0.7, 3, 5, False, True)
    assert np.allclose(m.embedding, np.arange(4.0))


def test_vectors_from_a_different_embedder_are_discarded(tmp_db):
    db.save_store("static", _store(), "emb:A")
    fresh = StaticMemoryStore()
    assert not db.load_store("static", fresh, "emb:B") and fresh.items == []


def test_clear_store(tmp_db):
    db.save_store("static", _store(), "emb:A")
    db.clear_store("static")
    assert not db.load_store("static", StaticMemoryStore(), "emb:A")
