# ATMC: Project Status (in simple words)

## 1. What the project is
We are building a **coding assistant that learns from its own mistakes**. When it gets a Python question wrong, a program runs the code and says why. The assistant saves a short note about it in a **memory**. When a similar question comes later, it reads the note and does better.

The memory system is called **ATMC**. It also gives each note a **trust score** and can forget old or bad notes, so the assistant does not learn from wrong information.

The model's own brain (its weights) is **not changed**. Only the memory grows.

## 2. What is built and working
| Part | Status |
|---|---|
| Chat assistant (web app): ask Python questions, code is run in a safe sandbox, answer is checked | Working |
| Memory with trust scores, forgetting and merging of notes | Working |
| Automatic checking of answers when the user gives no tests | Working |
| "Memory" page: search, pin, delete, export and import notes | Working |
| Strategy page: the assistant tests which way of solving works best, and goes back if a change is worse | Working |
| Runs on **Gemini (cloud)** or on a **local model, Qwen 7B, on this laptop** (free, private) | Working (switch is in a config file, not yet in the web page) |
| Our own test: **110 coding questions** with hidden company rules | Built and checked |
| Experiment tools, tables and charts made from saved results | Working |
| Automatic tests of the code | 177 passing |
| Paper draft and a step-by-step guide to how it was built | Written (paper needs the latest results added) |

## 3. What we tested and what we found

**A. With Gemini, when we planted wrong notes in memory**
- Trust scoring kept wrong notes away from the model: about **10 times fewer** wrong notes shown.
- Accuracy did **not** improve, because Gemini was already right about 90% of the time.

**B. Our own test (110 questions) with the local model**
- Each question hides a company rule, for example "money is always whole paise" or "dates are written DD-MM-YYYY". The question never says the rule.
- Without memory, Qwen scored **24%**. On ordinary questions with no hidden rule, it scored **100%**. So it is a good coder that simply does not know our rules.
- With a **simple memory** that stores what the checker said after a failure, it scored **40%**. That is **+16 points**, and the statistics are strong (3 runs, 330 question comparisons, p < 0.0001).

**C. Did ATMC's extra parts (trust, forgetting, merging) help on top of simple memory?**
- **No.** Trust alone made no difference. Forgetting made it **worse** (about 8 points lower). Merging made no difference.
- We found the reasons: the trust score was marking "I failed" notes as unreliable (they are actually the useful ones), and forgetting threw away lessons too early. The small model also almost never says which note it used, which our trust system relied on.
- We also tested the case where the checker is sometimes wrong. Trust did **not** protect the memory there.

## 4. What we can honestly claim and what we cannot
**Can claim**
- A small local model **learns company rules from its own verified failures**, with a measured +16 points.
- Trust scoring keeps known-wrong notes out of the model's view (shown with Gemini).
- We tested our own design carefully and know why parts of it did not help.

**Cannot claim yet**
- That ATMC beats a simple memory.
- That trust makes learning safe when the checker is wrong.
- That the model weights improve (we have not trained the model).

**Limits:** the 110-question test was written by us; one local model; three runs per setting.

## 5. What we plan next
1. Improve ATMC using what we learned (score the note's verified facts, not the attempt; test directly whether a note helps; forget by usefulness) and test it on **new rule families it has never seen**.
2. Optional: train the local model on its own verified answers (fine-tuning) to show the model itself improving.
3. Add a model switch and a live "model learning" view to the web app for the demo.
4. Update the paper, README and slides with these results.

## 6. How to run it
Backend: `pilot/.venv/bin/uvicorn app.backend.main:app --port 8000`
Frontend: `cd app/frontend && npm run dev`, then open http://localhost:5173

All result files are in `pilot/results/`; tables and charts are in `docs/results/`.
