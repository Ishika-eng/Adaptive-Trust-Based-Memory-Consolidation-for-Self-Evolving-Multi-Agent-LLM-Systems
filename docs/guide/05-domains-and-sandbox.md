# 05 · Tasks, checkers and the sandbox

Files: `pilot/domains/base.py` (79), `gsm8k.py` (42), `coding.py` (227), `pilot/tasks.py` (49), `pilot/sandbox.py` (104).

## What you are building and why

The agent loop (chapter 06) must work for *any* kind of task. Everything that differs between task types goes behind one interface called a **Domain**: how to load tasks, build the prompt, pull the answer out of the reply, **check it against ground truth**, write the memory note, and build a deliberately wrong note for poison experiments. Two domains exist: `math` (GSM8K) and `code` (HumanEval + MBPP, plus user-typed tasks).

The rule that makes the whole project honest: **verification is deterministic**. The answer is compared to a gold number or executed against real tests. An LLM is never the judge of correctness.

## 5.1 The `Verdict` and the `Domain` interface (`base.py`)

```python
@dataclass
class Verdict:
    correct: bool
    stage: str          # ok | wrong_answer | syntax_error | runtime_error | timeout | no_answer | unverified | user_rejected
    detail: str = ""    # short human-readable failure detail (fed to the reflector)
    evidence: str = "tests"   # tests | user | generated   -> how much to trust this verdict
    verified: bool = True     # False = no ground truth yet (user task with no tests)
```

`verified=False` is the crucial idea added for the chat assistant: an answer with no tests is *not yet knowledge*, so memory must not learn from it until the user (or auto-check) supplies a verdict.

`Domain` methods to implement: `load_tasks(cfg, n, seed)`, `build_prompt(task, memories, plan)`, `extract_answer(raw, task)`, `check(task, answer)`, `reflection(task, raw, answer, verdict)` (the note text), `poison_text(task)`, and (code only) `revise_prompt`. `poison_specs(tasks, embedder, every)` is shared: for every `every`-th task it returns `(poison_text, embedding_of_the_question)`.

Two helpers are shared by every prompt:

- `memory_block(memories)` → a prefix `"Here are some past problem-solving experiences that may or may not be relevant: Past experience 1: … Past experience 2: …"` and an instruction suffix asking the model to end with `USED: <numbers>` or `USED: none`. **This suffix is what makes citation-conditioned trust possible.**
- `plan_block(plan)` → `"Plan to follow: …"`.

## 5.2 The math domain (`gsm8k.py`, `tasks.py`)

- `load_tasks`: reads `gsm8k_test.jsonl`, gives each row an id `gsm8k/{line_number}` (**ids are required**: the curriculum tracks which tasks were attempted — we once forgot and hit a `KeyError`), shuffles with `random.Random(seed)`, takes `n`, and sets `gold` = the text after `####`.
- Prompt: "Solve this grade-school math problem step by step. End your response with a line of the exact form '#### <final numeric answer>'." (+ the `USED` instruction when memories exist).
- `check`: `numbers_match(pred, gold)` → `Verdict(True,"ok")`, else `no_answer` or `wrong_answer` with "answered X, expected Y".
- Note text: `Q: …\nA: <first 400 chars of the reply>\nOutcome: CORRECT|INCORRECT (gold=…)`.
- Poison: `make_wrong_answer(gold)` adds 7 (or subtracts 7 for negatives) so the wrong answer is plausible, and `build_poison_text` formats a fake note ending `Outcome: INCORRECT (…)`.

## 5.3 The coding domain (`coding.py`)

### Loading

- **HumanEval**: each task has `prompt` (a function stub with docstring), `entry_point`, `test` (defines `check(candidate)`), and a reference `canonical_solution`.
- **MBPP**: `text` + the first test shown in the prompt (`"Your code should pass this test:\n<test_list[0]>"`), full `test_list`, optional `test_setup_code`, reference `code`.
- `KNOWN_BROKEN_MBPP = {180, 367, 493, 927}`: four tasks whose *own reference solution fails its own tests* (found by running all 974). Two assert wrong values; two use a `Node` class the dataset never defines. They are skipped.
- `load_tasks` concatenates the datasets named in `coding.datasets`, shuffles with the seed, returns `n`.

### Building the executable program for a check (`_program`)

- HumanEval: `<question stub>\n\n<model code>\n\n<test>\n\ncheck(<entry_point>)`. The stub supplies imports/helpers; a model's full definition simply overrides it.
- MBPP: `<setup>\n<model code>\n\n<tests joined by newlines>`.

### `check` — the verification pipeline

1. No code → `no_answer`.
2. `ast.parse` for syntax (HumanEval-style bodies are parsed with the question prepended) → `syntax_error` with the line.
3. `run_python(program, timeout_s)` in the sandbox (next section).
4. If this is a **user task with no tests** → `_check_without_tests`: timeout/crash are verified *failures*; a clean run is `Verdict(False, "unverified", stdout, verified=False)`.
5. Otherwise: timed out → `timeout`; exit code 0 → `ok`; non-zero → `wrong_answer` if `AssertionError` is in stderr else `runtime_error`, with the last stderr line as detail.

### Extracting the answer

`extract_answer` takes the first fenced code block that defines the needed function (`def <entry_point>`); for user tasks it takes the first block containing a `def`/`class`; falls back to the last block, then to the raw text minus the `USED:` line. `explanation(raw)` returns the prose outside code fences (what the chat shows above the code).

### User tasks

`CodingDomain.user_task(question, tests, idx, context, previous)` builds a task dict with `dataset="user"`, `tests` as a single block of assertions (or empty), and — for follow-ups — `context` (earlier exchanges, injected into the prompt) and a `question` that *also names the earlier request* (`"<previous>\nFollow-up: <new>"`) so retrieval and the saved note still make sense later ("make it faster" alone is meaningless).

### Poison construction (`poison_text`)

Goal: a **plausible wrong solution that really fails the tests**, so we are not testing the system against nonsense.

1. Take the reference solution (HumanEval: stub + canonical body).
2. List single-site edits from a table (`==`↔`!=`, `<`↔`<=`, `+`↔`-`, `and`↔`or`, `True`↔`False`, `[0]`↔`[1]`, `min(`↔`max(`, …), restricted to the **solution region** (never the docstring: edits there change nothing). Shuffle with `random.Random(task_id)` for determinism.
3. Append a fallback: replace each `return <expr>` value with `None` (needed because 27 of 60 tasks had no flippable operator in an earlier version).
4. Try up to 40 edits; **run each through `check`** and keep the first that fails with `wrong_answer` or `runtime_error`.
5. Format: `Task: …\nSolution:\n<wrong code>\nOutcome: INCORRECT (wrong_answer: assertion failed)`. If no edit fails the tests, return `None` and skip that task.

> **Honest caveat.** The note says `Outcome: INCORRECT`, and ATMC also stores planted notes as *failed attempts* (trust 0.35). A stronger test inserts wrong notes recorded as **correct**. We did not run it; it is the most important future experiment (chapter 14).

## 5.4 The sandbox (`sandbox.py`)

### Why it exists

You are about to execute code written by a language model. It could loop forever, crash, read your files or open network connections. The sandbox limits the damage. It is **not** a hard security boundary (see end).

### The call

`run_python(code, timeout_s=10.0) -> RunResult(returncode, stdout, stderr, timed_out, isolation)`

### The five layers (outermost first)

1. **macOS `sandbox-exec` profile** (when available). A generated Seatbelt profile:

   ```
   (version 1)
   (allow default)
   (deny network*)
   (deny file-write* (require-not (require-any (subpath "<workdir>") (literal "/dev/null") (literal "/dev/dtracehelper"))))
   (deny file-read* (subpath "<repo>/.env"))  (deny file-read* (subpath "~/.ssh")) ... .aws .gnupg .config/gcloud
   ```

   Meaning: everything allowed by default, **no network**, **writes only inside the private temp dir**, and **no reads of secrets**. `isolation` is reported as `sandbox-exec`; elsewhere it falls back to `process-limits-only`.
2. **A fresh temp directory** (`tempfile.mkdtemp`, resolved with `os.path.realpath` because on macOS `/var` is a symlink to `/private/var` and the profile matches real paths only), deleted afterwards. The code is written to `main.py` there.
3. **A scrubbed environment**: `PATH=/usr/bin:/bin`, `HOME` and `TMPDIR` pointed at the temp dir. No API keys are inherited.
4. **`python -I`** (isolated mode: ignores user site-packages and env vars), run with `start_new_session=True` so a timeout can `os.killpg` the **whole process group**, plus `preexec_fn` resource limits: CPU seconds = timeout+2, max file size 10 MB, max 64 open files (and 1 GB address space on Linux; macOS does not enforce `RLIMIT_AS`).
5. **Output truncation** to 4000 characters each for stdout/stderr.

### Not a hard boundary

Reads are allowed (except the listed secrets) and a determined attacker could do harm. For a hostile threat model, run everything in a container with no network and a read-only filesystem.

## Pitfalls

- **Prompt stubs vs. model code in HumanEval**: some models return only a function body. The assembled program handles both because the stub is valid Python and the model's definition overrides it.
- **Tests that are wrong.** Always run the dataset's own reference solutions through your checker once. That is how the four broken MBPP tasks were found.
- **Poison that is not wrong.** Mutating a docstring or a no-op changes nothing; verify each poison by execution.
- **`stdout` leaks into verdicts.** Code that prints is fine, but the auto-verifier's harness prints a marker line to carry results; it must not be confusable with user output (chapter 09).

## Tests (`tests/test_sandbox.py`, `tests/test_coding_domain.py`)

Sandbox: plain code runs; an infinite loop is killed; output is truncated; the environment is scrubbed; code cannot write or delete outside its work directory; the network is blocked; `.env` is unreadable.

Coding domain: dataset sizes are right; **the reference solutions pass their own tests** (this is the check that exposed the four broken MBPP tasks); wrong code is rejected *with the right stage*; an infinite loop times out; a body-only reply is accepted for HumanEval; extraction prefers the block that defines the entry point and falls back to unfenced text; **poison really fails its tests**; the prompt contains the task, plan, memories and the citation instruction.

## Checkpoint

```bash
pilot/.venv/bin/python -m pytest tests/test_sandbox.py tests/test_coding_domain.py -q
pilot/.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, "pilot")
from sandbox import run_python
print(run_python("print(sum(range(10)))"))                                    # stdout '45\n', returncode 0, isolation 'sandbox-exec' on macOS
print(run_python("while True: pass", timeout_s=1).timed_out)                  # True
print(run_python("import urllib.request as u; u.urlopen('http://example.com')").returncode)   # non-zero (no network)
EOF
```
