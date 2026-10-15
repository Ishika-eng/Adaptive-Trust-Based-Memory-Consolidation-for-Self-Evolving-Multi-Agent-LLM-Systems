# 11 · Testing

Folder: `tests/` — 12 files, 135 test functions that expand to **148 test cases** (some are parameterised). The full suite runs in about 38 seconds and needs **no network and no API key**.

```bash
pilot/.venv/bin/python -m pytest tests -q          # 148 passed
```

## Philosophy (what made the suite useful)

1. **Fake the model, not the world.** The only thing replaced is `llm.complete`. The embedder is a deterministic stand-in, but the **sandbox is real**: code actually runs in a subprocess, so tests exercise the true checker.
2. **Tests assert behaviour, not existence.** A planner test that passes even if the planner is never called is worthless (we wrote one, found it vacuous, and replaced it with tests that count planner calls and assert the plan text reaches the prompt).
3. **Test the failure modes** the system was designed for: quota exhausted mid-step leaves memory untouched; a crash in the sandbox is a verified failure; an old database upgrades; a hostile import cannot raise trust.
4. **Each bug becomes a test.** NaN in JSON, results directory bound at import, cache used from threads, two wrong solutions agreeing on a trivial test — all have regression tests.

## The fixtures (`tests/conftest.py`)

- `sys.path` setup so `pilot/` and `app/backend/` import by bare name.
- **`WordHashEmbedder`**: an `Embedder` that hashes each word (md5 → one of 64 buckets) into a count vector. Deterministic, offline, instant, and similar wording gives similar vectors — enough to test ranking logic without loading the ONNX model. (It is crude: hash collisions mean *near*-wording is unreliable, so tests use identical or heavily overlapping text.)
- **`FakeLLM` / `fake_llm`**: `fake_llm(script)` monkeypatches `llm.complete` with a callable that **records every prompt** and returns `script(prompt)`. It also bumps `llm.USAGE` so cost accounting still works. Tests can assert on what the agent actually sent.
- `cfg` and `embedder` fixtures.

A typical scripted model (`tests/test_agent.py::math_llm`) answers `"Problem: <q>"` with the gold or a wrong number, returns a plan for "Break the following task…", a lesson for "previous attempt…", and adds a `USED:` line when the prompt contained "Past experience".

## The API fixture (`tests/test_api.py`)

`api` builds a **fresh backend per test**:

1. Point `db.DB_PATH` at a temporary file.
2. Replace `embeddings.make_embedder` with one that returns `WordHashEmbedder` (no model download).
3. `sys.modules.pop("main")` then `importlib.import_module("main")` — the module builds all state at import, so re-importing gives a clean server.
4. Install a scripted fake model that handles math prompts, plans, lessons, and **benchmark code prompts** by looking up the task's reference solution (or a deliberately wrong function in `FAIL` mode).
5. Set `main.BOOK.explore = 0.0` so live questions always use the champion (otherwise a 10% random exploration would make tests flaky).
6. Return a `fastapi.testclient.TestClient` (needs `httpx`).

Helper scripts for the chat tests: `_chat_script(api, code)` makes the model return a given code block for "careful Python coding assistant" prompts; `_autocheck_script` also scripts test generation and alternative solutions; `_capturing` records prompts to assert that follow-up context really reaches the model.

## What each file covers

| File | Functions | Covers |
|---|---|---|
| `test_stores.py` | 18 | Trust rule; gate; failed note starts just above the gate; cited-only credit; low-trust filtering; **trust outweighs similarity**; pillars off; pruning (and that it does *not* fire in a short run); compression + poison-laundering flag; relevance floor; evidence weighting |
| `test_memory_management.py` | 9 | uids and sources; evidence counts; pinned notes survive prune/merge; merge lineage; remove/pin errors; import resets trust and skips duplicates; retrieval explanations; DB round-trip; **old-database migration** |
| `test_agent.py` | 12 | One step end to end; cited-only trust; poison counting; planner on/off and isolation; reflection only on failure; quota mid-step leaves memory untouched; curricula; a code-domain step |
| `test_coding_domain.py` | 9 | Dataset sizes; reference solutions pass; wrong-code stages; timeouts; extraction rules; poison really fails; prompt contents |
| `test_sandbox.py` | 8 | Isolation guarantees (network, writes, deletes, `.env`, environment, output size, infinite loop) |
| `test_config_and_persistence.py` | 6 | Config validation and overrides; documented defaults; DB round-trip; vectors from a different embedder are discarded; clear |
| `test_experiment.py` | 8 | A complete result file; finished runs skipped unless forced; held-out phase frozen; poison flag; identical sequence for fixed-order conditions; overrides do not leak; quota abort writes no partial file; every preset valid |
| `test_report.py` | 6 | Exact McNemar values; detects a real difference; identical systems are null; aggregation and pairing; empty report; NaN→null |
| `test_llm_cache.py` | 4 | Cache hit and counters; different prompt misses; disabled when not deterministic; **usable from several threads** |
| `test_autoverify.py` | 11 | Parsing; run-tests; verified; switched; mistaken test cannot condemn; inconclusive; identical alternatives; underspecified; follow-up context in the tests prompt |
| `test_strategy.py` | 12 | Wilson; classification; promotion; **rollback**; per-type evidence; practise vs serve; unresolved events; table roles; each strategy runs the promised steps |
| `test_api.py` | 32 | Side-by-side demo; benchmark lab; chat with tests; failing tests; unverified → feedback; rejection → lesson; validation; follow-ups and new chat; rejection carried into context; suggestions and practice; strategy recording/resolution; dispute; auto-check cases; memory listing/pin/delete/export/import; import safety; readable 502 on unexpected errors |

## How to write a new test (pattern)

```python
def test_a_rejected_import_does_not_change_memory(api):
    before = len(api.get("/api/memory/chat").json())
    bad = {"embedder": "other:model", "items": []}
    assert api.post("/api/memory/chat/import", json=bad).status_code == 400
    assert len(api.get("/api/memory/chat").json()) == before
```

For agent-level behaviour: create a fake model with `fake_llm(script)`, build a store with `build_store("atmc", cfg, embedder, lambda t: "SUMMARY")` and an agent with `build_agent(...)`, call `agent.step(task)`, assert on `store.items`, `result.verdict`, and `fake.prompts`.

## Pitfalls

- **Flaky randomness** (strategy exploration, curriculum exploration): pin it in the fixture (`explore = 0.0`, fixed seeds).
- **Fake model too strict.** When a new prompt shape appears (the "review your draft" prompt), the fake must handle it or tests fail for the wrong reason.
- **Hash-embedder limits.** Do not test semantic similarity with it; test ranking with identical/overlapping text or use the real embedder in a one-off script.

## Checkpoint

All 148 pass. Then deliberately break something (change the penalty to 0.05, or make the planner see memories) and confirm a test fails — a suite that cannot fail proves nothing.
