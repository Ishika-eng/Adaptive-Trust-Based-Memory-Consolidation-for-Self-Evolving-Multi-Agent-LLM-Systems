import { useEffect, useState } from 'react'
import { getJSON, postJSON } from '../api'

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

export default function Demo() {
  const [tasks, setTasks] = useState([])
  const [taskId, setTaskId] = useState(0)
  const [solving, setSolving] = useState(false)
  const [result, setResult] = useState(null)
  const [state, setState] = useState(null)
  const [error, setError] = useState(null)
  const [poisoning, setPoisoning] = useState(false)

  const refreshState = () => getJSON('/api/state').then(setState).catch(() => {})

  useEffect(() => {
    getJSON('/api/tasks').then(setTasks).catch(() => setError('Could not reach backend. Is uvicorn running on :8000?'))
    refreshState()
  }, [])

  const handleSolve = async () => {
    setSolving(true)
    setError(null)
    try {
      const data = await postJSON('/api/solve', { task_id: taskId })
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
      await postJSON('/api/poison')
      refreshState()
    } finally {
      setPoisoning(false)
    }
  }

  const handleReset = async () => {
    await postJSON('/api/reset')
    setResult(null)
    refreshState()
  }

  return (
    <div>
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
          baseline notebook: {state.static_memory_size} notes · ATMC notebook: {state.atmc_memory_size} notes
          {state.atmc_compressions > 0 && <> · compressions run: {state.atmc_compressions}</>}
          {' '}· problems solved this session: {state.solved_ids.length}
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
    </div>
  )
}
