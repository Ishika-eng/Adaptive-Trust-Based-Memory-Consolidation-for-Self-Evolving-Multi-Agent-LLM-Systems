import { useEffect, useState } from 'react'
import { getJSON, postJSON } from '../api'

const STAGE_LABEL = {
  ok: 'all tests passed', wrong_answer: 'a test failed', syntax_error: 'syntax error',
  runtime_error: 'crashed while running', timeout: 'timed out', no_answer: 'no code returned',
}

function LearningCurve({ curve }) {
  if (curve.length < 2) return <p className="muted">The learning curve appears after two steps.</p>
  const W = 420, H = 110, pad = 8
  const pts = curve.map((v, i) => [pad + (i / (curve.length - 1)) * (W - 2 * pad), H - pad - v * (H - 2 * pad)])
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="curve" role="img" aria-label="running pass rate">
      <line x1={pad} x2={W - pad} y1={pad} y2={pad} className="curve-grid" />
      <line x1={pad} x2={W - pad} y1={H - pad} y2={H - pad} className="curve-grid" />
      <polyline points={pts.map(p => p.join(',')).join(' ')} className="curve-line" />
      {pts.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r="2.5" className="curve-dot" />)}
      <text x={W - pad} y={pad - 1} textAnchor="end" className="curve-label">100%</text>
      <text x={W - pad} y={H - 1} textAnchor="end" className="curve-label">0%</text>
    </svg>
  )
}

function StepView({ step }) {
  const v = step.verdict
  return (
    <div className="step">
      <div className="step-head">
        <span className={`pill ${v.correct ? 'pill--good' : 'pill--bad'}`}>
          {v.correct ? '✓ PASSED' : '✗ FAILED'} — {STAGE_LABEL[v.stage] || v.stage}
        </span>
        <span className="muted">
          picked by {step.chosen_by === 'you' ? 'you' : `the agent (${step.chosen_by} curriculum)`} ·
          {' '}{step.cost.tokens_in + step.cost.tokens_out} tokens · retrieval {step.cost.retrieval_ms} ms
        </span>
      </div>

      <h4>Task</h4>
      <pre className="code code--task">{step.task.question}</pre>

      {step.plan && (<><h4>1 · Plan (made without seeing memory)</h4><pre className="code">{step.plan}</pre></>)}

      <h4>2 · Memories it pulled up ({step.retrieved.length})</h4>
      {step.retrieved.length === 0 && <p className="muted">Memory is empty or nothing passed the trust threshold.</p>}
      <div className="mem-list">
        {step.retrieved.map((m, i) => (
          <div key={i} className={`mem-card ${m.is_poison ? 'mem-card--poison' : ''} ${m.used ? 'mem-card--used' : ''}`}>
            <div className="mem-card-top">
              {m.is_poison && <span className="badge badge--poison">⚠ planted wrong note</span>}
              {m.used && <span className="badge badge--used">✓ relied on this</span>}
              <span className="mem-card-hits">trust {m.trust.toFixed(2)} · seen {m.hits}×</span>
            </div>
            <p className="mem-card-text">{m.text}</p>
          </div>
        ))}
      </div>
      {step.poison.in_context > 0 && (
        <p className="muted">Planted wrong notes in context: {step.poison.in_context} · actually used: {step.poison.used}</p>
      )}

      <h4>3 · Code it wrote</h4>
      <pre className="code">{step.answer || '(none)'}</pre>

      <h4>4 · Checked by running the real tests</h4>
      <p>{v.correct ? 'Every test passed.' : <>Failed at <strong>{v.stage}</strong>: <code>{v.detail}</code></>}</p>

      {step.lesson && (<><h4>5 · Lesson it wrote for next time</h4><pre className="code code--lesson">{step.lesson}</pre></>)}
    </div>
  )
}

export default function Assistant() {
  const [domain, setDomain] = useState('code')
  const [state, setState] = useState(null)
  const [tasks, setTasks] = useState([])
  const [pick, setPick] = useState('')
  const [step, setStep] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const refresh = async (d = domain) => {
    try {
      const [s, t] = await Promise.all([getJSON(`/api/agent/state?domain=${d}`), getJSON(`/api/agent/tasks?domain=${d}`)])
      setState(s); setTasks(t); setError(null)
    } catch (e) { setError(`Could not reach the backend: ${e.message}`) }
  }
  useEffect(() => { setStep(null); setPick(''); refresh(domain) }, [domain])

  const act = async (fn) => {
    setBusy(true); setError(null)
    try { await fn() } catch (e) { setError(e.message) } finally { setBusy(false) }
  }
  const runStep = (task_id) => act(async () => {
    const r = await postJSON('/api/agent/step', { domain, task_id: task_id || null })
    setStep(r); setState(r.summary); setPick('')
    setTasks(await getJSON(`/api/agent/tasks?domain=${domain}`))
  })
  const plant = () => act(async () => { await postJSON('/api/agent/poison', { domain }); await refresh() })
  const reset = () => act(async () => { await postJSON('/api/agent/reset', { domain }); setStep(null); await refresh() })

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>Self-evolving assistant</h2>
          <p className="muted">
            One agent that plans, remembers, writes code, runs it against real tests, learns from failures, and
            chooses what to practise next. Everything it learns is held in ATMC memory.
          </p>
        </div>
        <div className="seg">
          {['code', 'math'].map(d => (
            <button key={d} className={`seg-btn ${domain === d ? 'seg-btn--on' : ''}`} onClick={() => setDomain(d)}>
              {d === 'code' ? 'Coding' : 'Math'}
            </button>
          ))}
        </div>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {state && (
        <section className="stats">
          <div className="stat"><span>{state.steps}</span>tasks attempted</div>
          <div className="stat"><span>{state.steps ? Math.round(100 * state.passed / state.steps) : '–'}{state.steps ? '%' : ''}</span>pass rate</div>
          <div className="stat"><span>{state.memory_size}</span>memories{state.n_compressions > 0 && ` (${state.n_compressions} merges)`}</div>
          <div className="stat"><span>{state.remaining}</span>tasks left in pool</div>
          {state.isolation && (
            <div className="stat stat--wide" title="Generated code runs in a separate process; with sandbox-exec it cannot write outside a temp folder or use the network.">
              <span>{state.isolation === 'sandbox-exec' ? '🔒 sandboxed' : '⚠ limits only'}</span>code execution
            </div>
          )}
        </section>
      )}

      <section className="controls">
        <div className="control-step">
          <span className="control-step-num">Autonomous</span>
          <button className="btn btn--primary" disabled={busy} onClick={() => runStep(null)}>
            {busy ? 'Working…' : 'Let the agent choose & solve the next task'}
          </button>
        </div>
        <div className="control-step">
          <span className="control-step-num">Or pick one yourself</span>
          <select value={pick} onChange={e => setPick(e.target.value)} disabled={busy}>
            <option value="">— choose a task —</option>
            {tasks.filter(t => !t.done).map(t => <option key={t.id} value={t.id}>{t.id}: {t.question.replace(/\s+/g, ' ').slice(0, 60)}…</option>)}
          </select>
        </div>
        <button className="btn btn--ghost" disabled={busy || !pick} onClick={() => runStep(pick)}>Solve it</button>
        <button className="btn btn--warn" disabled={busy || state?.poisoned} onClick={plant}>
          {state?.poisoned ? 'Wrong notes planted ✓' : 'Plant wrong past solutions'}
        </button>
        <button className="btn btn--ghost btn--reset" disabled={busy} onClick={reset}>Start over</button>
      </section>

      {state && state.steps > 0 && (
        <section className="curve-wrap">
          <h4>Learning curve — running pass rate over {state.steps} tasks</h4>
          <LearningCurve curve={state.learning_curve} />
        </section>
      )}

      {step ? <StepView step={step} /> : (
        <p className="muted">Run a step to watch the agent work. The first few tasks are the cold start — memory is empty.</p>
      )}

      {state && state.history.length > 0 && (
        <section>
          <h4>Recent tasks</h4>
          <table className="explorer-table">
            <thead><tr><th>task</th><th>result</th><th>failure stage</th><th>memories used</th><th>planted notes in context</th></tr></thead>
            <tbody>
              {[...state.history].reverse().map((h, i) => (
                <tr key={i}>
                  <td>{h.task_id}</td>
                  <td>{h.correct ? '✓' : '✗'}</td>
                  <td>{h.correct ? '—' : STAGE_LABEL[h.stage] || h.stage}</td>
                  <td>{h.n_retrieved}</td>
                  <td>{h.n_poison_retrieved}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  )
}
