# 10 · The frontend

Folder: `app/frontend/` (React 19 + Vite 8, plain JavaScript, no UI or state libraries). About 1,800 lines of JSX/CSS.

## What you are building and why

A product-style web app over the API: a **chat assistant** as the main page, and supporting pages that make the memory visible. Visibility is the point: a reviewer should *see* which notes were used, how much each is trusted, and what the assistant learned.

## 10.1 File map

```
src/
  main.jsx                 mounts <App/> in StrictMode
  api.js                   API address + getJSON/postJSON with readable errors
  index.css                design tokens (colours, radii, shadows), light + dark
  App.css                  every component's style
  App.jsx                  app shell: header, nav, status, theme, pages
  lib/
    Logo.jsx               the SVG mark
    highlight.jsx          tiny Python syntax highlighter (React elements)
    markdown.jsx           safe Markdown subset for explanations
  pages/
    Assistant.jsx          chat (main page)          425 lines
    MemoryExplorer.jsx     manage notes              184
    Evolution.jsx          strategy scoreboard       143
    Results.jsx            saved experiment tables   222
    BenchmarkLab.jsx       the older step-through    190
    Architecture.jsx       parts and honest status    89
public/favicon.svg         the logo as favicon
index.html                 title, description, theme-color
```

## 10.2 The API client (`api.js`)

```js
export const API = 'http://127.0.0.1:8000'
async function readError(r) { const t = await r.text(); try { return JSON.parse(t).detail ?? t } catch { return t } }
export const getJSON  = async (p) => { const r = await fetch(API + p); if (!r.ok) throw new Error(await readError(r)); return r.json() }
export const postJSON = async (p, body) => { /* POST with JSON body; same error handling */ }
```

Two decisions: (1) **throw on non-2xx with the server's `detail` text**, so the UI can show "Gemini daily quota exhausted…" instead of a blank failure; (2) one hard-coded address — change it (or read `import.meta.env.VITE_API`) when you deploy.

## 10.3 State management: just `useState` + one wrapper

No Redux. Pages keep their data in `useState` and reload from the server after each action. The one pattern worth copying is a `run` wrapper that handles busy and error state uniformly:

```js
const run = async (fn) => {
  setBusy(true); setError(null)
  try { await fn() } catch (e) { setError(e.message) } finally { setBusy(false) }
}
```

Every button is `onClick={() => run(async () => { await postJSON(...); await refresh() })}`. `busy` disables controls so two requests cannot overlap.

## 10.4 The app shell (`App.jsx`)

- **Pages table**: `assistant`, `memory`, `evolution`, `results`, `lab`, `architecture`; the nav renders buttons from it (`aria-current` on the active one).
- **Health check**: `GET /api/chat/summary` on mount and every 20 s → header shows "Connecting…", "Online · sandboxed" or "Backend offline"; when offline a banner shows the exact command to start the backend, with a Retry link.
- **Theme**: `system → light → dark`, stored in `localStorage` (`atmc-theme`, always inside `try/catch` because storage can be unavailable) and applied as `data-theme` on `<html>`. `system` removes the attribute so the OS preference applies.

## 10.5 The design system (`index.css`, `App.css`)

Colours, radii and shadows are **CSS custom properties** defined once on `:root` (light) and overridden in two places for dark: `:root[data-theme='dark']` and `@media (prefers-color-scheme: dark){ :root:not([data-theme='light']) … }`. Components only reference tokens (`var(--surface)`, `var(--accent)`, `var(--good-soft)`), so adding a theme means redefining tokens, not touching components.

Tokens: `--bg --surface --surface-2 --border --text --text-dim --accent --accent-soft --good/bad/warn/info` (+ `-soft` backgrounds), `--code-bg`, `--radius`, `--shadow`. Typography uses the system font stack (no web fonts, so it works offline). `prefers-reduced-motion` turns animations off. Every interactive element has a visible `:focus-visible` ring.

Responsive rules: the chat page is a two-column grid (chat + 320 px sidebar) that collapses to one column below 960 px; the sidebar becomes a full-screen panel opened by a "Memory" button. Tables sit in scroll containers.

## 10.6 The chat page (`Assistant.jsx`)

State: `turns`, `summary`, `notes`, `suggestion`, composer text (`prompt`, `tests`, `showTests`, `autocheck`), `busy`, `pending`, `error`, `panel` (mobile drawer). `refresh()` loads `/api/chat/history`, `/summary`, `/api/memory/chat`, `/api/chat/suggest` **in parallel** (`Promise.all`).

Components inside the file:

| Component | Job |
|---|---|
| `Turn` | One exchange: the user bubble (or a "practice task I chose" card) and the assistant block |
| `CodeBlock` | Highlighted code with a Copy button (`navigator.clipboard`, failure ignored) |
| `Markdown` / `Highlight` (from `lib/`) | Render the explanation and the code — as React elements, **never `dangerouslySetInnerHTML`**, so model output cannot inject markup |
| `AutoCheckNote` | Explains what auto-check did (how many agreed tests; why inconclusive/underspecified) |
| `Feedback` | "It worked / It didn't (+comment)" for unverified answers |
| `Dispute` | "This is wrong" for auto-verified answers |
| `Details` | "How I solved this": strategy, plan, retrieved notes with trust + the *why* line, lesson, cost |
| `NextTask` | The assistant's own practice suggestion with Practise / Another buttons |
| `Sidebar` | Counts, most-trusted notes, the suggestion, Clear everything |
| `Working` | Animated "thinking" indicator that cycles through stage names |

Behaviours that make it feel like a product:

- **Composer**: Enter sends, Shift+Enter inserts a newline, ⌘/Ctrl+Enter also sends; auto-grows to 200 px; it **clears immediately** on send and is restored if the request fails.
- **Optimistic pending bubble** while the request runs.
- **Auto-scroll** to the newest message (`scrollIntoView`).
- **Follow-up awareness**: placeholder changes, "↪ follow-up" tag, "+ New chat" starts a fresh context; a divider appears between conversations.
- **Honest status pills**: *Tests passed*, *Tests failed*, *Not verified*, *Auto-verified*, *Disputed by you*.

### The two library helpers

- `highlight.jsx`: one regular expression with alternatives for comments, triple-quoted strings, strings, decorators, numbers and words; keywords/built-ins from sets; the word after `def`/`class` is coloured as function/class; output is an array of strings and `<span class="tok-…">`.
- `markdown.jsx`: splits on blank lines; handles bullet and numbered lists, `` `code` ``, `**bold**`, `*italic*`, and `$math$` (shown as code — there is no LaTeX renderer).

## 10.7 The other pages

- **Memory (`MemoryExplorer.jsx`)** — search, filter (worked / failed / pinned / blocked / planted), sort, expandable note text, trust chip, evidence ✓/✗ counts, source badge, version, last used, Pin/Delete, Export (Blob download) and Import (file picker → POST; shows how many were added/skipped). Read-only views for the benchmark-lab stores.
- **Evolution (`Evolution.jsx`)** — the three strategies explained, a card per problem type (success rate bars, evidence needed, roles *Default / Challenger / Champion*), the recent attempts, and "Let it practise 5 tasks" (loops `suggest` → `practise`, with a Stop button).
- **Results (`Results.jsx`)** — lists saved experiments from `/api/eval`, tabs per experiment/domain, the mean ± sd table, paired comparisons with CIs and p-values, a "preliminary" warning if any condition has fewer than 3 seeds, plus the older pilot dashboards.
- **Architecture (`Architecture.jsx`)** — the eight parts of a self-evolving agent with an honest status for each ("Evaluated — no significant gain" for planner/reflection/curriculum).
- **Benchmark lab (`BenchmarkLab.jsx`)** — the original autonomous step-through on benchmark tasks (pick a task or let the agent choose, plant wrong notes, learning curve).

## 10.8 Build and run

```bash
cd app/frontend
npm install
npm run dev -- --port 5173      # http://localhost:5173 (needs the backend on :8000)
npm run build                   # production bundle in dist/ (must finish without errors)
npm run lint                    # oxlint: currently only warnings (state-in-effect, components created in render)
```

## Pitfalls

- **Inline `code` styles leaking into code blocks** gave every line its own background; block code needs its own reset.
- **Clearing the composer only after success** felt broken; clear immediately and restore on error.
- **Opaque network errors**: solved on both sides (backend `HTTPException`, frontend `readError`).
- **Browser storage can throw**; wrap every read/write.
- **Verify in a real browser.** Several bugs (clipped diagram, overlapping legend, a click that did not navigate, a table overflowing its column) were invisible in code and obvious in a screenshot.

## Checkpoint

With the backend running: open `http://localhost:5173`, ask "Write a function merge(intervals)…" with the test `assert merge([[1,3],[2,6]]) == [[1,6]]`. You should get a *Tests passed* answer with highlighted code, a note in the sidebar, and the same note listed on the Memory page with trust 0.55.
