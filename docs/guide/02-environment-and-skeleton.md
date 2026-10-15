# 02 · Environment and project skeleton

## What you are building and why

A repository where `python`, `node`, an API key and three datasets all work, before any ATMC logic exists. Most "it does not run on my machine" problems are decided here.

## Step 1 — Create the repo and a virtual environment

```bash
mkdir atmc && cd atmc && git init
python3 -m venv pilot/.venv          # the project keeps its venv inside pilot/ (gitignored)
source pilot/.venv/bin/activate
```

Why `pilot/.venv`? The engine lives in `pilot/` (it began as a "pilot study"), and keeping the interpreter next to it means every command in the project is `pilot/.venv/bin/python …` with no activation step. It is a convention, not a requirement.

## Step 2 — Dependencies

`requirements.txt` (pinned to what the project ran on):

```
fastapi==0.141.1        # web API
fastembed==0.9.0        # local sentence embeddings (ONNX; no PyTorch needed)
google-genai==2.20.0    # Gemini client
numpy==2.5.2
pydantic==2.13.5        # request validation in FastAPI
python-dotenv==1.2.3    # loads .env
PyYAML==6.0.3           # configs/default.yaml
scikit-learn==1.9.0     # cosine_similarity, TF-IDF baseline
uvicorn==0.53.0         # runs FastAPI
```

Two things are **not** in that file but are needed, so install them yourself:

```bash
pilot/.venv/bin/pip install -r requirements.txt
pilot/.venv/bin/pip install pytest httpx      # tests; FastAPI's TestClient needs httpx
# only if you will generate the paper:
pilot/.venv/bin/pip install matplotlib python-docx pypdf
```

For the paper PDF you also need LibreOffice (`soffice`) on your PATH.

## Step 3 — The API key (and its limits)

1. Create a Gemini API key at Google AI Studio.
2. Put it in a file named `.env` at the repo root:

   ```
   GEMINI_API_KEY=your_key_here
   ```

3. `.env` is in `.gitignore`. **Never commit it, print it, or paste it into chat.** The sandbox (chapter 05) also *denies reads of `.env`* so generated code cannot steal it.

**Free-tier reality, measured on this project:**

| Limit | Value we observed | Consequence in the design |
|---|---|---|
| Requests per minute | 15 | The wrapper spaces calls 4.5 s apart (`model.request_interval_s`) |
| Requests per day (per project, per model) | about 500 | Experiments are **resumable**; the runner stops cleanly on a daily-quota error |
| Reset | roughly a daily window (hours) | Re-run the same command later; finished runs are skipped |

Quota belongs to a *project*, not a key: a second key from the same Cloud project shares the allowance. A key from a new project has its own. Two processes running at once add up against the per-minute limit — we hit this by accident once and got a 429 from our own duplicate run, so run **one** evaluation process at a time.

## Step 4 — Get the datasets

Put them in `pilot/data/`. The code reads these exact field names:

| File | Lines | Fields used | Where it comes from |
|---|---|---|---|
| `gsm8k_test.jsonl` | 1319 | `question`, `answer` (the numeric answer follows `####`) | GSM8K test split (Cobbe et al.) |
| `HumanEval.jsonl` | 164 | `task_id`, `prompt`, `entry_point`, `canonical_solution`, `test` | OpenAI `human-eval` |
| `mbpp.jsonl` | 974 | `task_id`, `text`, `code`, `test_list`, `test_setup_code` | Google MBPP (sanitized splits not used) |

Each is JSON-lines: one JSON object per line. Check with `wc -l pilot/data/*`.

## Step 5 — Folder skeleton

```bash
mkdir -p configs pilot/{data,results/eval,domains} app/backend tests docs/guide paper
touch pilot/domains/__init__.py
```

## Step 6 — The import convention (important)

The project is **not an installed package**. Modules import each other by bare name (`import llm`, `from stores import AdaptiveTrustStore`). This works because every entry point adds the right folders to `sys.path` first:

- scripts in `pilot/` insert their own directory (`sys.path.insert(0, str(Path(__file__).resolve().parent))`);
- the backend inserts `pilot/` and `app/backend/` (top of `main.py`);
- `tests/conftest.py` inserts `pilot/` and `app/backend/`.

Trade-off: very little ceremony, but you must run things from places that set the path (the provided entry points do). If you later package it, switch to relative imports.

## Step 7 — `.gitignore`

```
.env
pilot/.venv/
__pycache__/
*.pyc
node_modules/
dist/
*.log
app/backend/memory.db             # the app's SQLite database (user data)
pilot/results/llm_cache.sqlite    # cached model replies (large, regenerable)
```

Result files under `pilot/results/eval/` **are** committed: they are the evidence behind every table in the paper.

## Step 8 — Frontend skeleton

```bash
cd app && npm create vite@latest frontend -- --template react
cd frontend && npm install
npm run dev -- --port 5173
```

The stack is React 19 + Vite 8 (see `app/frontend/package.json`); there are no UI libraries. The API address is a constant in `src/api.js` (`http://127.0.0.1:8000`) and the backend enables permissive CORS for development.

## Pitfalls

- **Forgetting `pytest`/`httpx`.** `pip install -r requirements.txt` alone does not run the tests.
- **Wrong interpreter.** Using system `python3` instead of `pilot/.venv/bin/python` gives "No module named fastembed".
- **First run is slow.** `fastembed` downloads the embedding model (about 130 MB) the first time. The backend then takes ~20 s to start because it loads that model.
- **Port already in use.** `lsof -ti:8000 | xargs kill` frees 8000; same for 5173.

## Checkpoint

```bash
pilot/.venv/bin/python -c "import fastembed, google.genai, sklearn, yaml, fastapi; print('deps ok')"
wc -l pilot/data/*.jsonl                       # 164, 1319, 974
test -f .env && echo "key file present"        # never cat it
cd app/frontend && npm run build               # builds without errors
```
