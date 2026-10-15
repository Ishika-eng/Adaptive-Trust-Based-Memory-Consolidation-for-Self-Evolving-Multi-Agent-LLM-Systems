# 08 · The backend and the chat assistant

Files: `app/backend/main.py` (776 lines), `app/backend/db.py` (261). Tests: `tests/test_api.py` (32 functions).

## What you are building and why

A thin web layer over the engine. It adds exactly three things the engine does not have: **HTTP**, **persistence**, and **per-user conversation state**. All real behaviour (memory, checks, learning) stays in `pilot/`. If you find yourself writing memory logic inside a route, move it into `pilot/`.

Run it from the repo root:

```bash
cd app/backend && ../../pilot/.venv/bin/python -m uvicorn main:app --port 8000
```

`/docs` (FastAPI's generated page) lists every route.

## 8.1 Start-up sequence (module level in `main.py`)

Everything is built **once at import** and held in module globals; the server is single-process.

1. Load `.env`; add `pilot/` and `app/backend/` to `sys.path`.
2. `CFG = load_config()`; `llm.configure(CFG)`; `LOCK = threading.RLock()` (every route that touches shared memory takes it — requests are serialised).
3. `DOMAINS = {math, code}`; `EMBEDDER = make_embedder(...)` (loads the ONNX model: this is the ~20 s start-up).
4. `db.init_db()` (creates/migrates tables).
5. `ISOLATION = run_python("pass").isolation` — checked once so status polls are cheap.
6. Three groups of state:
   - **Side-by-side demo** (`COMPARE`): a `static` and an `atmc` agent on a fixed 20-question math pool. (Backend still exposes it; the redesigned UI no longer shows it.)
   - **Benchmark-lab assistants** (`ASSISTANTS`): one ATMC agent per domain with planner + reflection + weakness curriculum, on a 60-task pool.
   - **The chat assistant** (`CHAT`): an agent (code domain) plus `turns` (history), `pending` (unverified attempts waiting for a user verdict), `learned` (auto-verified attempts that can be disputed), the current `suggestion`, and the conversation counter `conv`.
7. `VERIFIER = AutoVerifier(...)` and `BOOK = StrategyBook(...)` (chapter 09); `BOOK.events` loaded from the database.

## 8.2 Persistence (`db.py`)

SQLite file `app/backend/memory.db` (gitignored). **No ORM**, only `sqlite3`. Tables:

| Table | Holds |
|---|---|
| `memories` | One row per note: `store_name`, text, embedding (JSON list), trust, hits, created_at_step, correct, is_poison, `uid`, `source`, `successes`, `failures`, `last_used_step`, `version`, `pinned` |
| `store_state` | Per store: `step`, `n_added`, `n_compressions`, `n_poison_laundered` (bookkeeping counters a store needs to resume correctly) |
| `history` | Agent step logs (so the curriculum knows what was already attempted after a restart) |
| `chat` | One row per chat turn, as JSON, ordered by `idx` |
| `strategy_events` | Strategy attempts and outcomes (chapter 09) |
| `meta` | `embedder` = the name of the embedder that produced the saved vectors |

Key behaviours:

- `save_store` **deletes and rewrites** that store's rows (simple; O(notes) per save — fine at our scale, the first thing to change if it grows).
- `load_store` first checks `meta.embedder`; if it differs from the current embedder the stored vectors are **discarded** (`clear_store`), because vectors from different models are meaningless to each other.
- **In-place migration.** `_connect()` runs `PRAGMA table_info(memories)` and `ALTER TABLE … ADD COLUMN` for any evidence column that is missing, so a database from before those columns existed still loads. Rows with an empty `uid` get ids assigned on load. This was tested on the real database and has a unit test.
- Saves happen **after every mutation** inside the route (`_save_chat`, `_save_book`, …), so a crash loses at most the request in flight.

## 8.3 Error handling — a lesson worth copying

If a route raises an unexpected exception, FastAPI returns a 500 **without CORS headers**; the browser then reports a useless "Failed to fetch". `_quota_guard(fn)` wraps every model-touching call:

```python
try: return fn()
except llm.QuotaExhausted: raise HTTPException(429, "Gemini daily quota exhausted — try again later…")
except HTTPException: raise
except Exception as e:    raise HTTPException(502, f"{type(e).__name__}: {str(e)[:300]}")
```

`HTTPException` is handled inside the middleware that adds CORS headers, so the browser gets a readable JSON message instead. The frontend's `readError` (chapter 10) extracts `detail`.

## 8.4 The chat assistant flows

### Task creation and conversation context

`_conversation_context()` takes the last `CONTEXT_TURNS=3` non-practice turns of the **current conversation**, and renders each as: the user's prompt (≤600 chars), the assistant's code (≤1500 chars) and an outcome line — "the tests passed", "it failed (wrong answer: …)", "the user confirmed it worked", "the user said it did NOT work: <comment>", or "not verified". Passing the user's own rejection forward lets a follow-up like "fix it" be meaningful. `POST /api/chat/new` increments `CHAT["conv"]`; earlier turns stay on screen but stop being context.

### `POST /api/chat/ask {prompt, tests, autocheck}`

1. Validate (non-empty; ≤4000 characters each) → 400 otherwise.
2. Under `LOCK`: build `user_task(...)`; **choose a strategy** (`_pick_strategy`, mode `"serve"`).
3. `agent.attempt(task, strategy)` (inside `_quota_guard`).
4. If the verdict is `unverified` **and** `autocheck` **and** `autoverify.enabled`: run `VERIFIER.run(task, answer)`. On `verified` or `switched` the answer may be replaced and the verdict becomes `Verdict(True, "ok", evidence="generated")`. The verifier's token usage is added to the turn.
5. If the verdict is now verified (tests, or auto-check): `agent.learn(result)`; auto-verified results are kept in `CHAT["learned"]` so the user can dispute them; the strategy outcome is recorded. If not verified: store in `CHAT["pending"]`.
6. Build the **turn JSON** (`_turn`): `status` ∈ `passed | failed | unverified | auto_verified` (later `disputed`), explanation, code, plan, `retrieved` (text, trust, `used`, evidence counts, and the `why` explanation), `autocheck`, `strategy`, `filtered`, cost, memory size. Save it (`db.save_turn`) and return it.

### `POST /api/chat/feedback {id, correct, comment}`

Only for `unverified` turns. Pops the pending result (410 if the server restarted since — pending state is in memory only), calls `agent.learn(result, correct=…, feedback=…)`, records the strategy outcome, saves.

### `POST /api/chat/dispute {id, comment}`

For `auto_verified` turns: penalise the notes it relied on (`update_from_outcome(..., False)`), **retire its own note** (`correct=False`, `trust=0.0`), store a new failure note carrying the user's comment, mark the strategy attempt as a loss, set status `disputed`. This is the safety valve for generated-test evidence.

### Practice and suggestion

- `GET /api/chat/suggest[?refresh=true]` → `agent.next_task()` through the weakness curriculum plus a human-readable **reason** that is only "close to a task I got wrong…" when the curriculum really used the weakness path (`last_mode`), "trying something different…" for exploration, "I haven't made a mistake yet…" otherwise. It costs no model call. The suggestion is cached until used or refreshed.
- `POST /api/chat/practise {task_id}` → the assistant solves a *benchmark* task (which has real tests, so the result is verified), with strategy mode `"practise"` (explore), learns, records the outcome, and clears the suggestion.

### Other routes

| Route | Purpose |
|---|---|
| `GET /api/chat/history`, `/summary`, `/strategies` | Data for the UI (summary includes counts, `conversation`, sandbox isolation) |
| `POST /api/chat/reset` | Clear memory, turns, history, strategy events; rebuild state |
| `GET /api/memory/{chat|code|math|static|atmc}` | List notes with evidence |
| `POST /api/memory/chat/pin`, `DELETE /api/memory/chat/{uid}` | Pin/unpin, delete |
| `GET /api/memory/chat/export`, `POST /api/memory/chat/import` | Move memory between machines |
| `GET /api/eval[?experiment&domain&ref]` | Saved experiment results + paired statistics (calls `report.build_report`) |
| `GET /api/results`, `/api/ablation` | Older pilot summaries kept for the original dashboard |
| `/api/agent/*`, `/api/solve`, `/api/poison`, `/api/reset`, `/api/state`, `/api/tasks` | Benchmark lab and the side-by-side demo |

### Import safety (`/api/memory/chat/import`)

Rejects an embedder name that differs from the server's (vectors incompatible), more than 2000 notes, wrong vector dimension, empty or >6000-character text. Accepted notes go through `store.import_item`, which resets trust to the initial value.

## 8.5 Concurrency model

One process, one `RLock`: requests that change memory run one at a time. That is simple and safe; its limit is throughput (chapter 14). The model call dominates latency anyway (seconds), so the lock is not the bottleneck until you have many users.

## Pitfalls

- **Pending state is volatile.** Unverified and auto-verified attempts live in memory; after a restart the UI gets 410. A real product would persist the attempt (note text + embedding) too.
- **CORS is wide open** (`allow_origins=["*"]`) for development. Restrict it before exposing the server.
- **No authentication.** There is one shared memory. Do not expose this to the internet.
- **Import-time construction** makes tests awkward; the API tests reload the module per test with a temporary DB path and a fake embedder (chapter 11).
- **Quota inside routes.** Practice and auto-check multiply model calls; the free tier ends quickly under heavy use.

## Checkpoint

```bash
pilot/.venv/bin/python -m pytest tests/test_api.py -q                       # 32 functions
cd app/backend && ../../pilot/.venv/bin/python -m uvicorn main:app --port 8000 &
sleep 25 && curl -s localhost:8000/api/chat/summary                        # JSON with asked, solved, memory_size…
curl -s -X POST localhost:8000/api/chat/ask -H 'Content-Type: application/json' \
  -d '{"prompt":"write add(a, b)","tests":"assert add(1, 2) == 3"}' | head -c 400
```
