import numpy as np
import pytest

from factory import build_atmc, build_store
from memory import update_trust


def vec(i, n=8):
    v = np.zeros(n); v[i % n] = 1.0
    return v


def mk(cfg, embedder, **kw):
    return build_atmc(cfg, embedder, summarize_fn=lambda texts: "SUMMARY: " + " | ".join(t[:10] for t in texts), **kw)


def test_trust_update_rule():
    assert update_trust(0.5, +1) == pytest.approx(0.55)
    assert update_trust(0.5, -1) == pytest.approx(0.35)       # penalty > reward, by design
    assert update_trust(0.99, +1) == 1.0 and update_trust(0.1, -1) == 0.0


def test_verified_note_enters_at_reward_level_and_wrong_note_just_above_the_gate(cfg, embedder):
    s = mk(cfg, embedder)
    s.add("good", vec(0), correct=True)
    s.add("bad", vec(1), correct=False)
    assert [round(m.trust, 2) for m in s.items] == [0.55, 0.35]


def test_note_below_the_minimum_is_rejected_at_the_gate(cfg, embedder):
    s = mk(cfg, embedder)
    s.initial_trust = 0.4               # 0.4 - 0.15 = 0.25 < t_min 0.30
    s.add("bad", vec(0), correct=False)
    assert s.size() == 0


def test_trust_only_moves_for_memories_the_model_cited(cfg, embedder):
    s = mk(cfg, embedder)
    s.add("a", vec(0), correct=True); s.add("b", vec(1), correct=True)
    a, b = s.items
    s.update_from_outcome([a, b], task_correct=True, used_mask=[True, False])
    assert a.trust == pytest.approx(0.60) and b.trust == pytest.approx(0.55)
    s.update_from_outcome([a, b], task_correct=False, used_mask=[False, True])
    assert a.trust == pytest.approx(0.60) and b.trust == pytest.approx(0.40)


def test_low_trust_memory_is_filtered_from_retrieval(cfg, embedder):
    s = mk(cfg, embedder)
    s.add("good", vec(0), correct=True)
    s.add("bad", vec(0), correct=False)
    s.items[1].trust = 0.2              # below t_min
    got = s.retrieve(vec(0), k=2)
    assert [m.text for m in got] == ["good"]


def test_trust_outweighs_similarity_for_ranking(cfg, embedder):
    s = mk(cfg, embedder)
    s.add("trusted", vec(0), correct=True)
    s.add("poison", vec(1), correct=False, is_poison=True)
    q = 0.8 * vec(1) + 0.6 * vec(0)     # closer to the poison note, yet trust should win
    assert s.retrieve(q, k=1)[0].text == "trusted"


def test_trust_pillar_off_ignores_outcomes(cfg, embedder):
    s = mk(cfg, embedder, enable_trust=False)
    s.add("bad", vec(0), correct=False)
    assert s.items[0].trust == 1.0 and s.size() == 1     # nothing is gated
    s.update_from_outcome(s.items, False)
    assert s.items[0].trust == 1.0


def test_forgetting_prunes_a_note_once_it_is_old_enough(cfg, embedder):
    s = mk(cfg, embedder, enable_compression=False)
    s.add("bad", vec(0), correct=False)
    s.items[0].trust = 0.31
    s._step += 200                      # age the note far past its decay horizon
    for i in range(1, 11):              # every 10th add triggers a prune pass
        s.add(f"filler{i}", vec(i), correct=True)
    assert "bad" not in [m.text for m in s.items]


def test_prune_does_not_fire_inside_a_short_run(cfg, embedder):
    """With the default decay a low-trust note needs ~70 steps to fall under the
    prune cutoff, so a 40-task run (54 steps incl. seeding) never deletes by
    forgetting -- at that horizon the forgetting pillar acts through the decay term
    in the ranking score, not by removal."""
    s = mk(cfg, embedder, enable_compression=False)
    s.add("bad", vec(0), correct=False)
    s.items[0].trust = 0.31
    for i in range(1, 54):
        s.add(f"filler{i}", vec(i), correct=True)
    assert "bad" in [m.text for m in s.items]


def test_forgetting_off_never_prunes(cfg, embedder):
    s = mk(cfg, embedder, enable_forgetting=False, enable_compression=False)
    for i in range(30):
        s.add(f"n{i}", vec(i), correct=True)
    assert s.size() == 30


def test_compression_merges_old_notes_and_flags_laundered_poison(cfg, embedder):
    s = mk(cfg, embedder, enable_forgetting=False)
    s.add("poison", vec(0), correct=False, is_poison=True)
    for i in range(1, 16):
        s.add(f"clean{i}", vec(i), correct=True)
    assert s.n_compressions >= 1 and s.size() < 16
    assert any(m.text.startswith("SUMMARY") for m in s.items)
    assert s.n_poison_laundered >= 1


def test_compression_without_callbacks_fails_loudly(cfg):
    from stores import AdaptiveTrustStore
    s = AdaptiveTrustStore(compress_every=2, compress_min_age_steps=0, compress_batch_size=2)
    s.add("a", vec(0), True)
    with pytest.raises(RuntimeError):
        for i in range(4):
            s.add("b", vec(i + 1), True)


def test_baselines(cfg, embedder):
    none = build_store("none", cfg, embedder, None)
    none.add("x", vec(0), True)
    assert none.retrieve(vec(0)) == [] and none.size() == 0

    ref = build_store("reflexion", cfg, embedder, None)
    ref.add("success", vec(0), True)                 # successes are not stored
    for i in range(5):
        ref.add(f"fail{i}", vec(i), False)
    assert ref.size() == cfg.retrieval.top_k
    assert [m.text for m in ref.retrieve(vec(0))] == ["fail4", "fail3", "fail2"]   # most recent first

    static = build_store("static", cfg, embedder, None)
    static.add("a", vec(0), True); static.add("b", vec(1), False)
    assert static.retrieve(vec(1), k=1)[0].text == "b"      # pure similarity


def test_unknown_store_kind_is_rejected(cfg, embedder):
    with pytest.raises(ValueError):
        build_store("nope", cfg, embedder, None)
