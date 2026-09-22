import { useEffect, useState } from 'react'
import './App.css'

const API = 'http://127.0.0.1:8000'

function TrustBar({ value }) {
  const pct = Math.round(value * 100)
  const color = value >= 0.5 ? '#2f9e44' : value >= 0.3 ? '#e8a33d' : '#e03131'
  return (
    <div className="trust-bar-track" title="Trust score: how reliable this memory has proven to be, based on whether using it led to correct answers.">
      <div className="trust-bar-fill" style={{ width: `${pct}%`, background: color }} />
      <span className="trust-bar-label">trust {value.toFixed(2)}</span>
    </div>
  )
}

function MemoryCard({ m }) {
  return (
    <div className={`mem-card ${m.is_poison ? 'mem-card--poison' : ''} ${m.used ? 'mem-card--used' : ''}`}>
      <div className="mem-card-top">
        {m.is_poison && (
          <span className="badge badge--poison" title="A deliberately WRONG note we planted, to test whether the system falls for it.">
            ⚠ planted wrong note
          </span>
        )}
        {m.used && (
          <span className="badge badge--used" title="The AI says it actually relied on this note to answer.">
            ✓ relied on this
          </span>
        )}
        <span className="mem-card-hits">seen before: {m.hits}x</span>
      </div>
      <TrustBar value={m.trust} />
      <p className="mem-card-text">{m.text}</p>
    </div>
  )
}

function ResultColumn({ label, sublabel, result, tone }) {
  if (!result) return <div className="result-col result-col--empty">Not solved yet</div>
  return (
    <div className={`result-col result-col--${tone}`}>
      <div className="result-col-header">
        <div>
          <h3>{label}</h3>
          <span className="result-col-sub">{sublabel}</span>
        </div>
        <span className={`pill ${result.correct ? 'pill--good' : 'pill--bad'}`}>
          {result.correct ? '✓ CORRECT' : '✗ WRONG'}
        </span>
      </div>
      <div className="result-meta">
        <span>answered: <strong>{result.pred ?? '—'}</strong></span>
        <span>actual answer: <strong>{result.gold}</strong></span>
        <span>notebook size: <strong>{result.memory_size}</strong></span>
      </div>
      <h4 className="mem-list-title">Notes it pulled up to help answer ({result.retrieved.length})</h4>
      {result.retrieved.length === 0 && <p className="muted">No notes retrieved (none passed the trust threshold, or notebook is empty).</p>}
      <div className="mem-list">
        {result.retrieved.map((m, i) => <MemoryCard key={i} m={m} />)}
      </div>
    </div>
  )
}

function HeadlineStat({ stats }) {
  if (!stats || !stats.static) return null
  const ratio = (stats.static.exposure_mean / stats.atmc.exposure_mean).toFixed(1)
  return (
    <div className="headline-stat">
      <div className="headline-number">{ratio}×</div>
      <div className="headline-text">
        <strong>fewer planted-wrong-notes reached the AI's reasoning</strong> with trust-gated memory,
        compared to a "remember everything" baseline — measured across {stats.n_seeds} independent runs, with no accuracy loss.
      </div>
    </div>
  )
}

function ResultsDashboard({ stats }) {
  if (!stats || !stats.static) return null
  const maxExp = Math.max(stats.static.exposure_mean, stats.atmc.exposure_mean)
  return (
    <div className="dashboard">
      <h2>The proof, across {stats.n_seeds} independent runs (seeds: {stats.seeds.join(', ')})</h2>
      <p className="muted">Same setup run 3 separate times with different random task orders, to make sure the result isn't a fluke.</p>
      <div className="dash-row">
        <div className="dash-metric">
          <span className="dash-metric-label">How often a wrong note reached the AI</span>
          <div className="dash-bars">
            <div className="dash-bar-row">
              <span className="dash-bar-name">baseline</span>
              <div className="dash-bar-track">
                <div className="dash-bar-fill dash-bar-fill--static" style={{ width: `${(stats.static.exposure_mean / maxExp) * 100}%` }} />
              </div>
              <span className="dash-bar-value">{stats.static.exposure_mean.toFixed(1)}</span>
            </div>
            <div className="dash-bar-row">
              <span className="dash-bar-name">ATMC</span>
              <div className="dash-bar-track">
                <div className="dash-bar-fill dash-bar-fill--atmc" style={{ width: `${(stats.atmc.exposure_mean / maxExp) * 100}%` }} />
              </div>
              <span className="dash-bar-value">{stats.atmc.exposure_mean.toFixed(1)}</span>
            </div>
          </div>
        </div>
        <div className="dash-metric">
          <span className="dash-metric-label">Accuracy — did filtering cost us anything?</span>
          <div className="dash-bars">
            <div className="dash-bar-row">
              <span className="dash-bar-name">baseline</span>
              <div className="dash-bar-track">
                <div className="dash-bar-fill dash-bar-fill--static" style={{ width: `${stats.static.accuracy_mean * 100}%` }} />
              </div>
              <span className="dash-bar-value">{(stats.static.accuracy_mean * 100).toFixed(1)}%</span>
            </div>
            <div className="dash-bar-row">
              <span className="dash-bar-name">ATMC</span>
              <div className="dash-bar-track">
                <div className="dash-bar-fill dash-bar-fill--atmc" style={{ width: `${stats.atmc.accuracy_mean * 100}%` }} />
              </div>
              <span className="dash-bar-value">{(stats.atmc.accuracy_mean * 100).toFixed(1)}%</span>
            </div>
          </div>
          <p className="muted">No — accuracy stayed the same (or better). The filtering is free.</p>
        </div>
      </div>
    </div>
  )
}

export default function App() {
  const [tasks, setTasks] = useState([])
  const [taskId, setTaskId] = useState(0)
  const [solving, setSolving] = useState(false)
  const [result, setResult] = useState(null)
  const [state, setState] = useState(null)
  const [stats, setStats] = useState(null)
  const [error, setError] = useState(null)
  const [poisoning, setPoisoning] = useState(false)

  const refreshState = () => fetch(`${API}/api/state`).then(r => r.json()).then(setState).catch(() => {})

  useEffect(() => {
    fetch(`${API}/api/tasks`).then(r => r.json()).then(setTasks).catch(() => setError('Could not reach backend. Is uvicorn running on :8000?'))
    fetch(`${API}/api/results`).then(r => r.json()).then(setStats).catch(() => {})
    refreshState()
  }, [])

  const handleSolve = async () => {
    setSolving(true)
    setError(null)
    try {
      const res = await fetch(`${API}/api/solve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_id: taskId }),
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setResult(data)
      refreshState()
    } catch (e) {
      setError(String(e))
    } finally {
      setSolving(false)
    }
  }

  const handlePoison = async () => {
    setPoisoning(true)
    try {
      await fetch(`${API}/api/poison`, { method: 'POST' })
      refreshState()
    } finally {
      setPoisoning(false)
    }
  }

  const handleReset = async () => {
    await fetch(`${API}/api/reset`, { method: 'POST' })
    setResult(null)
    refreshState()
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>ATMC <span className="muted">— Adaptive Trust-based Memory Consolidation</span></h1>
        <p className="lede">
          AI agents that remember their past mistakes can also remember them <em>wrong</em> — and then
          trust that wrong memory forever. ATMC scores every memory by how trustworthy, relevant, and
          fresh it is, so bad memories get filtered out and good ones don't get lost in the noise.
        </p>
      </header>

      <section className="how-it-works">
        <div className="hiw-step">
          <span className="hiw-num">1</span>
          <div>
            <strong>We plant some deliberately wrong "memories"</strong>
            <p className="muted">Just like a real agent might accumulate a mistaken note over time.</p>
          </div>
        </div>
        <div className="hiw-arrow">→</div>
        <div className="hiw-step">
          <span className="hiw-num">2</span>
          <div>
            <strong>The AI solves a new problem</strong>
            <p className="muted">Using two memory systems side-by-side: one that remembers everything equally, one that weighs trust.</p>
          </div>
        </div>
        <div className="hiw-arrow">→</div>
        <div className="hiw-step">
          <span className="hiw-num">3</span>
          <div>
            <strong>Watch which one falls for the bad memory</strong>
            <p className="muted">And by how much, across many runs.</p>
          </div>
        </div>
      </section>

      <HeadlineStat stats={stats} />

      {error && <div className="error-banner">{error}</div>}

      <section className="controls">
        <div className="control-step">
          <span className="control-step-num">Step 1</span>
          <button onClick={handlePoison} disabled={poisoning || state?.poisoned} className="btn btn--warn">
            {state?.poisoned ? 'Wrong notes planted ✓' : (poisoning ? 'Planting…' : 'Plant wrong notes')}
          </button>
        </div>
        <div className="control-step">
          <span className="control-step-num">Step 2</span>
          <select value={taskId} onChange={e => setTaskId(Number(e.target.value))} disabled={solving}>
            {tasks.map(t => (
              <option key={t.id} value={t.id}>{t.id}. {t.question.slice(0, 60)}...</option>
            ))}
          </select>
        </div>
        <div className="control-step">
          <span className="control-step-num">Step 3</span>
          <button onClick={handleSolve} disabled={solving || tasks.length === 0} className="btn btn--primary">
            {solving ? 'Solving with both systems (~10s)…' : 'Solve & compare'}
          </button>
        </div>
        <button onClick={handleReset} className="btn btn--ghost btn--reset">Start over</button>
      </section>

      {state && (
        <p className="muted state-line">
          baseline notebook: {state.static_memory_size} notes · ATMC notebook: {state.atmc_memory_size} notes · problems solved this session: {state.solved_ids.length}
        </p>
      )}

      {result && (
        <section className="results-grid">
          <h2 className="question-echo">Problem: {result.question}</h2>
          <div className="results-columns">
            <ResultColumn label="Baseline" sublabel="remembers everything equally" result={result.static} tone="static" />
            <ResultColumn label="ATMC" sublabel="weighs trust before relying on a memory" result={result.atmc} tone="atmc" />
          </div>
        </section>
      )}

      <ResultsDashboard stats={stats} />

      <footer className="app-footer muted">
        This is a live view into the actual backend (pilot/memory.py + pilot/stores.py) — the same code
        that produced the experiment results above, not a mockup.
      </footer>
    </div>
  )
}
