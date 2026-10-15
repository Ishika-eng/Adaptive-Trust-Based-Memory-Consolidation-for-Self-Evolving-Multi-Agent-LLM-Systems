# 04 · Plumbing: config, embeddings, the LLM wrapper, the factory

Files: `configs/default.yaml`, `pilot/config.py` (101), `pilot/embeddings.py` (89), `pilot/llm.py` (190), `pilot/factory.py` (59).

These are the connectors. None is clever, but each one removes a whole category of bug.

## 4.1 Configuration — one file for every tunable

### Why

Early in the project constants (trust reward, penalty, top-k, seeds) were scattered through scripts. Two scripts could silently use different values and produce incomparable results. The fix: **one YAML file** (`configs/default.yaml`) and a loader.

### The file's sections

| Section | Keys (defaults) | Used by |
|---|---|---|
| `model` | `name: gemini-3.5-flash-lite`, `temperature: 0.0`, `max_retries: 4`, `request_interval_s: 4.5`, `cache: true` | `llm.configure` |
| `embedding` | `backend: fastembed`, `model: BAAI/bge-small-en-v1.5`, `gemini_model` | `make_embedder` |
| `retrieval` | `top_k: 3`, `min_similarity: 0.0` | agent, stores |
| `trust` | `initial 0.5`, `minimum 0.30`, `reward 0.05`, `penalty 0.15` | store |
| `decay` | `lambda_0 0.1`, `alpha 1`, `beta 1`, `prune_every 10`, `prune_min_retention 0.05` | store |
| `composite` | `w_trust 0.4`, `w_importance 0.4`, `w_decay 0.2` (must sum to 1) | store |
| `compression` | `every 8`, `batch_size 3`, `min_age_steps 5` | store |
| `pillars` | `trust/forgetting/compression: true` | ablation switches |
| `evaluation` | `dataset`, `n_tasks 40`, `seeds [42,123,456]`, `poison_every 3`, `heldout 0` | experiments |
| `agent` | `domain`, `use_planner`, `use_reflection`, `curriculum`, `curriculum_explore 0.2` | agent |
| `strategy` | `enabled`, `min_trials 5`, `explore 0.1` | chat backend |
| `autoverify` | `enabled`, `candidates 3`, `tests 8`, `evidence_weight 0.5` | chat backend |
| `coding` | `datasets [humaneval, mbpp]`, `timeout_s 10` | coding domain |
| `app` | `n_sample_tasks`, `sample_seed`, `assistant_pool 60`, `assistant{planner, reflection, curriculum}` | backend |

### The loader (`config.py`)

```python
class Config:                      # attribute AND item access over a nested dict
    def __getattr__(self, name):   # cfg.trust.minimum
        v = self._data[name]
        return Config(v) if isinstance(v, dict) else v
    def to_dict(self): return copy.deepcopy(self._data)

def load_config(path=None, overrides=None):
    data = yaml.safe_load(open(path or DEFAULT_PATH))
    if overrides: _deep_update(data, overrides)    # nested merge, e.g. {"agent": {"use_planner": True}}
    _validate(data)
    return Config(data)
```

`_validate` rejects nonsense **at load time** with a clear message: composite weights not summing to 1, `penalty ≤ reward`, `trust.minimum` outside [0,1), unknown embedding backend / domain / curriculum, `top_k < 1`, bad `autoverify` values, `min_similarity` outside [0,1). Fail early is cheaper than a mystery result.

**Reproducibility trick:** every experiment run saves `cfg.to_dict()` inside its result JSON, so any result can be re-run from its own file.

**Pitfall:** `Config.__getattr__` raises `AttributeError` for missing keys, so optional sections must be read with `cfg.to_dict().get("section", default)` (the code does this for `autoverify`, `strategy`, `retrieval.min_similarity`).

## 4.2 Embeddings — text to vectors

### Why embeddings matter

Retrieval asks "which past notes concern a *similar* question?". The first version used TF-IDF (word counts), which cannot match paraphrases or code. The upgrade was a real sentence-embedding model. The change was committed as "Central config and real sentence embeddings", and the earlier TF-IDF-era results were archived in `pilot/results/tfidf_baseline/` because they are **not comparable** with later ones.

### The interface (`embeddings.py`)

```python
class Embedder:
    name = "base"
    def embed(self, text) -> np.ndarray:       # cached; returns an L2-normalised float32 vector
        ...
```

Three backends behind one interface:

- `FastEmbedder("BAAI/bge-small-en-v1.5")` — local ONNX model, 384 dimensions, no API quota. **Default.**
- `GeminiEmbedder` — uses the API (spends quota).
- `TfidfEmbedder(corpus)` — the legacy baseline; needs a corpus to fit.

Two design choices to copy:

1. **Always normalise to length 1** (`_normalise`). Then cosine similarity is just a dot product, whichever backend produced the vector.
2. **Cache by text.** The same question is embedded once per condition per task; caching makes repeats free.

`embedder.name` (e.g. `fastembed:BAAI/bge-small-en-v1.5`) is stored in the database; if the embedder changes, saved vectors are **discarded**, because vectors from different models are meaningless to each other (chapter 08).

## 4.3 The LLM wrapper (`llm.py`)

One function does the work: `complete(prompt) -> str`. Everything else (prompts) lives in the domains. Responsibilities, in the order they run:

1. **Cache lookup** (only when `temperature == 0.0` and `model.cache` is true). The key is `sha256(model | temperature | prompt)`; the store is SQLite at `pilot/results/llm_cache.sqlite`; the connection is opened with `check_same_thread=False` and guarded by a `threading.Lock` because the web server calls it from worker threads (**we hit a real crash when this was missing**). A hit still increments the *logical* `calls`/token counters so cost metrics stay comparable, and increments `cache_hits`.
2. **Pacing**: before each real call, sleep until `MIN_INTERVAL_S` has passed since the previous call (4.5 s keeps us under 15 requests/minute).
3. **The call**: `client.models.generate_content(model, contents, config=GenerateContentConfig(temperature=...))`.
4. **Accounting**: add `usage_metadata.prompt_token_count` / `candidates_token_count` to the global `USAGE` dict (`calls`, `prompt_tokens`, `output_tokens`, `api_calls`, `cache_hits`). Experiments take `usage_snapshot()` before and after to get per-run cost.
5. **Retries**: up to `MAX_RETRIES` with exponential backoff; if the error text says "retry in Ns" (a rate-limit hint) wait that long instead.
6. **Daily quota**: if the error contains `"PerDay"`, raise `QuotaExhausted` *immediately*. Retrying is pointless; the experiment runner catches it, saves progress and exits.

### Why temperature 0 and a cache

Determinism makes experiments comparable and lets us cache. The cache saved us in two ways: runs that crashed on a quota error replay their finished calls for free on resume, and conditions that send identical prompts (for example "reflection on" before the first failure) share answers. Measured honestly: it saved 56% of calls in the agent experiment but only about one call per run in the ablation, because prompts contain retrieved memory text and diverge as soon as memories differ.

### The parsers in `llm.py`

- `extract_final_number(text)`: for GSM8K; finds the number after the last `####`.
- `numbers_match(pred, gold)`: numeric comparison with tolerance `1e-4`, string fallback.
- **`extract_used_mask(text, n_memories)`**: the citation parser, crucial for credit assignment. The prompt asks the model to end with `USED: 1,3` (or `USED: none`). The parser returns a list of booleans over the retrieved notes:
  - no memories → `[]`;
  - no `USED:` line, or no parseable numbers → **all True** (degrade to the old behaviour rather than silently freezing all trust);
  - `none` → all False;
  - otherwise True exactly for the listed (1-based) indices, ignoring out-of-range numbers.
- `summarize(texts)`: the merge prompt used by compression (“Merge the following … into a single short consolidated note (3-4 sentences) …”).

## 4.4 The factory (`factory.py`)

`build_store(kind, cfg, embedder, summarize_fn, **pillar_overrides)` is the **only** place config keys are mapped to constructor arguments. Experiments, tests and the backend all call it, so they cannot drift apart. `kind` ∈ `none | static | reflexion | atmc`; `pillar_overrides` takes `enable_trust=False` etc. for ablations.

## Pitfalls

- **Pacing and parallel processes.** Pacing is per process. Two evaluation processes at once exceed 15 RPM and the second dies with a 429.
- **Cache + temperature.** Never cache when temperature > 0; the code disables it automatically.
- **Thread safety.** A single shared SQLite connection from a multi-threaded server raises `ProgrammingError` without `check_same_thread=False` + a lock. A unit test now calls the cache from several threads.
- **Error visibility.** An uncaught exception in a FastAPI route shows up in the browser as "Failed to fetch" (the error response lacks CORS headers). The backend wraps model calls so failures become readable JSON (chapter 08).

## Checkpoint

```bash
pilot/.venv/bin/python -m pytest tests/test_config_and_persistence.py tests/test_llm_cache.py -q
pilot/.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, "pilot")
from config import load_config
from embeddings import make_embedder
import numpy as np
cfg = load_config(); e = make_embedder(cfg)
a, b = e.embed("merge overlapping intervals"), e.embed("combine ranges that overlap")
print(e.name, a.shape, float(np.linalg.norm(a)), float(a @ b))   # 384-d, norm 1.0, high similarity
EOF
```
