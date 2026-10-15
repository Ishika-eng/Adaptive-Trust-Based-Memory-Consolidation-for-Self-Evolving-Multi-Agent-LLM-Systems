import { useCallback, useEffect, useRef, useState } from 'react'
import { getJSON, postJSON } from '../api'
import { Highlight } from '../lib/highlight'

const pct = (x) => (x == null ? '–' : `${Math.round(x * 100)}%`)
const SPEEDS = { fast: 0, normal: 700, slow: 1800 }
const RULE_FAMILIES_NOTE = 'Each question hides a company rule (units, formats, error handling) that the question never states.'

function rolling(rows, key, w = 15) {
  return rows.map((_, i) => {
    const lo = Math.max(0, i - w + 1)
    const slice = rows.slice(lo, i + 1)
    return slice.filter(r => r[key].correct).length / slice.length
  })
}

function Chart({ rows }) {
  const W = 640, H = 220, P = { l: 38, r: 12, t: 12, b: 28 }
  const n = Math.max(rows.length, 2)
  const x = (i) => P.l + (i / (n - 1)) * (W - P.l - P.r)
  const y = (v) => P.t + (1 - v) * (H - P.t - P.b)
  const line = (vals) => vals.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ')
  const a = rolling(rows, 'no_memory'), b = rolling(rows, 'memory')
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="lab-chart" role="img" aria-label="Rolling pass rate, with and without memory">
      {[0, 0.25, 0.5, 0.75, 1].map(g => (
        <g key={g}>
          <line x1={P.l} x2={W - P.r} y1={y(g)} y2={y(g)} className="lab-grid" />
          <text x={P.l - 6} y={y(g) + 4} textAnchor="end" className="lab-axis">{Math.round(g * 100)}%</text>
        </g>
      ))}
      <text x={P.l} y={H - 8} className="lab-axis">question 1</text>
      <text x={W - P.r} y={H - 8} textAnchor="end" className="lab-axis">question {rows.length || '…'}</text>
      {rows.length > 1 && <path d={line(a)} className="lab-line lab-line--base" />}
      {rows.length > 1 && <path d={line(b)} className="lab-line lab-line--mem" />}
      {rows.length === 1 && <><circle cx={x(0)} cy={y(a[0])} r="4" className="lab-dot--base" /><circle cx={x(0)} cy={y(b[0])} r="4" className="lab-dot--mem" /></>}
    </svg>
  )
}

function Verdict({ view }) {
  return (
    <div className="lab-verdict">
      <span className={`pill ${view.correct ? 'pill--good' : 'pill--bad'}`}>{view.correct ? 'Passed' : 'Failed'}</span>
      {!view.correct && <span className="muted lab-detail">{view.detail || view.stage}</span>}
    </div>
  )
}

function Pane({ title, sub, view, memory }) {
  const [open, setOpen] = useState(false)
  return (
    <div className={`dashboard lab-pane ${memory ? 'lab-pane--mem' : ''}`}>
      <h3>{title} <span className="muted" style={{ fontWeight: 400 }}>{sub}</span></h3>
      {!view ? <p className="muted">Waiting for the first question…</p> : (
        <>
          <Verdict view={view} />
          <div className="codeblock lab-code">
            <div className="codeblock-bar"><span>answer</span></div>
            <pre><code>{view.answer ? <Highlight code={view.answer} /> : '(no code returned)'}</code></pre>
          </div>
          {memory && (
            <div className="lab-notes">
              <button className="link-btn" onClick={() => setOpen(!open)}>
                {view.notes_used.length ? `Recalled ${view.notes_used.length} earlier note${view.notes_used.length > 1 ? 's' : ''} ${open ? '▾' : '▸'}` : 'Nothing relevant in memory yet'}
              </button>
              {open && view.notes_used.map((n, i) => (
                <div key={i} className="lab-note">
                  <span className="trust-chip">trust {n.trust}</span>
                  <pre>{n.text}</pre>
                </div>
              ))}
              {view.note_saved && !view.correct && <p className="muted lab-saved">Saved for next time: the checker’s verified result.</p>}
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default function LearningLab() {
  const [model, setModel] = useState(null)
  const [summary, setSummary] = useState(null)
  const [rows, setRows] = useState([])
  const [running, setRunning] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [cfg, setCfg] = useState({ n: 110, seed: 42, speed: 'normal' })
  const stop = useRef(true)
  const cfgRef = useRef(cfg)
  useEffect(() => { cfgRef.current = cfg }, [cfg])

  const loadModel = useCallback(() => getJSON('/api/model').then(setModel).catch(e => setError(`Could not reach the backend: ${e.message}`)), [])
  useEffect(() => {
    loadModel()
    getJSON('/api/learn/state').then(async s => {
      if (s && !s.idle) { setSummary(s); setRows(await getJSON('/api/learn/rows')) }
    }).catch(() => {})
  }, [loadModel])

  const switchModel = async (mode) => {
    stop.current = true; setRunning(false); setError(null)
    try { setModel(await postJSON('/api/model', { mode })); setSummary(null); setRows([]) } catch (e) { setError(e.message) }
  }

  const doStep = async () => {
    const r = await postJSON('/api/learn/step')
    setRows(prev => [...prev, r.row]); setSummary(r.summary)
    return r.summary
  }

  const play = async () => {
    stop.current = false; setRunning(true); setError(null)
    try {
      for (;;) {
        if (stop.current) break
        const s = await doStep()
        if (s.done) break
        const wait = SPEEDS[cfgRef.current.speed]
        if (wait) await new Promise(res => setTimeout(res, wait))
      }
    } catch (e) { setError(e.message) } finally { setRunning(false); stop.current = true }
  }

  const start = async () => {
    setBusy(true); setError(null); setRows([])
    try { setSummary(await postJSON('/api/learn/start', { n: cfg.n, seed: cfg.seed })); setBusy(false); await play() }
    catch (e) { setError(e.message); setBusy(false) }
  }
  const stepOnce = async () => {
    setBusy(true); setError(null)
    try { if (!summary || summary.done) { setSummary(await postJSON('/api/learn/start', { n: cfg.n, seed: cfg.seed })); setRows([]) } await doStep() }
    catch (e) { setError(e.message) } finally { setBusy(false) }
  }
  const pause = () => { stop.current = true }
  const reset = () => { stop.current = true; setRunning(false); setSummary(null); setRows([]) }

  const last = rows[rows.length - 1]
  const a = summary?.no_memory, b = summary?.memory
  const delta = a && b && a.steps ? (b.passed - a.passed) / a.steps : null
  const families = a ? Object.keys(b.by_family).sort() : []
  const rescued = rows.filter(r => r.memory.correct && !r.no_memory.correct).slice(-8).reverse()
  const hurt = summary?.hurt?.length || 0
  const live = last && last.from_cache < last.model_calls

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>Learning Lab</h2>
          <p className="muted">
            The same questions go to two assistants running the <em>same model</em>: one with no memory, one that remembers what the
            checker told it after each failure. {RULE_FAMILIES_NOTE} Watch whether the one with memory gets better as it goes.
          </p>
        </div>
        <div className="seg" role="group" aria-label="Model">
          <button className={`seg-btn ${model?.mode === 'local' ? 'seg-btn--on' : ''}`} onClick={() => switchModel('local')} disabled={running}>
            Local · Qwen 7B
          </button>
          <button className={`seg-btn ${model?.mode === 'cloud' ? 'seg-btn--on' : ''}`} onClick={() => switchModel('cloud')} disabled={running}>
            Cloud · Gemini
          </button>
        </div>
      </div>

      {error && <div className="error-banner" role="alert">{error}</div>}
      {model && model.mode === 'cloud' && <p className="muted">Gemini already answers most of these correctly, so the difference is small. The local model is where learning shows.</p>}
      {model && model.mode === 'local' && !model.local_available && <div className="error-banner">The local model isn’t running. Start Ollama (<code>ollama serve</code>).</div>}

      <div className="dashboard lab-controls">
        <label>Questions
          <select value={cfg.n} onChange={e => setCfg({ ...cfg, n: +e.target.value })} disabled={running}>
            <option value={30}>30</option><option value={60}>60</option><option value={110}>110 (full)</option>
          </select>
        </label>
        <label>Order
          <select value={cfg.seed} onChange={e => setCfg({ ...cfg, seed: +e.target.value })} disabled={running}>
            <option value={42}>Question order A</option><option value={123}>Question order B</option><option value={456}>Question order C</option>
          </select>
        </label>
        <label>Speed
          <select value={cfg.speed} onChange={e => setCfg({ ...cfg, speed: e.target.value })}>
            <option value="fast">Fast</option><option value="normal">Normal</option><option value="slow">Slow</option>
          </select>
        </label>
        <div className="lab-buttons">
          {running
            ? <button className="btn btn--ghost" onClick={pause}>Pause</button>
            : summary && !summary.done && rows.length
              ? <button className="btn btn--primary" onClick={play} disabled={busy}>Resume</button>
              : <button className="btn btn--primary" onClick={start} disabled={busy || (model && !model.local_available && model.mode === 'local')}>{summary?.done ? 'Run again' : 'Start'}</button>}
          <button className="btn btn--ghost" onClick={stepOnce} disabled={running || busy}>One question</button>
          <button className="link-btn" onClick={reset} disabled={busy}>Reset</button>
        </div>
      </div>

      {summary && !summary.idle && (
        <>
          <div className="lab-scores">
            <div className="stat lab-score">
              <div className="dash-metric-label">Without memory</div>
              <div className="headline-number">{pct(a.steps ? a.passed / a.steps : null)}</div>
              <div className="muted">{a.passed} of {a.steps} correct</div>
            </div>
            <div className="stat lab-score lab-score--mem">
              <div className="dash-metric-label">With memory</div>
              <div className="headline-number">{pct(b.steps ? b.passed / b.steps : null)}</div>
              <div className="muted">{b.passed} of {b.steps} correct · {summary.memory_size} notes stored</div>
            </div>
            <div className="stat lab-score">
              <div className="dash-metric-label">Difference</div>
              <div className="headline-number" style={{ color: delta > 0 ? 'var(--good)' : delta < 0 ? 'var(--bad)' : undefined }}>
                {delta == null ? '–' : `${delta > 0 ? '+' : ''}${Math.round(delta * 100)} pts`}
              </div>
              <div className="muted">question {summary.step} of {summary.n}</div>
            </div>
          </div>

          <div className="dashboard">
            <h2>Pass rate over the last 15 questions</h2>
            <Chart rows={rows} />
            <div className="strat-legend lab-legend">
              <span><i className="lab-key lab-key--base" /> Without memory</span>
              <span><i className="lab-key lab-key--mem" /> With memory</span>
              {last && <span className="muted">{live ? 'live model call' : 'model answers replayed from the recorded run'} · checks and memory ran for real</span>}
            </div>
          </div>

          {last && (
            <>
              <div className="dashboard">
                <h2>Question {last.i}{last.family && last.role === 'rule' ? <span className="pill pill--pending" style={{ marginLeft: 8 }}>hidden rule: {last.family}</span> : last.role ? <span className="pill pill--pending" style={{ marginLeft: 8 }}>{last.role === 'control' ? 'ordinary question' : 'already-known practice'}</span> : null}</h2>
                <p className="lab-question">{last.question}</p>
              </div>
              <div className="lab-panes">
                <Pane title="Without memory" sub="same model, starts fresh every time" view={last.no_memory} />
                <Pane title="With memory" sub="remembers earlier failures" view={last.memory} memory />
              </div>
            </>
          )}

          <div className="dashboard">
            <h2>Moments where memory made the difference</h2>
            {rescued.length === 0 ? <p className="muted">None yet. Memory needs a few failures in a family before it can help.</p> : (
              <ul className="how-list">
                {rescued.map(r => (
                  <li key={r.i}>Question {r.i} <span className="pill pill--pending">{r.family}</span> — passed with memory, failed without.</li>
                ))}
              </ul>
            )}
            <p className="muted">{summary.rescued.length} questions rescued by memory so far; {hurt} where memory did worse.</p>
          </div>

          {families.length > 0 && (
            <div className="dashboard">
              <h2>By kind of rule</h2>
              <table className="explorer-table">
                <thead><tr><th>Hidden rule</th><th>Without memory</th><th>With memory</th><th /></tr></thead>
                <tbody>
                  {families.map(f => {
                    const [pa, na] = a.by_family[f], [pb, nb] = b.by_family[f]
                    const d = nb ? pb / nb - pa / na : 0
                    return (
                      <tr key={f}>
                        <td>{f}</td><td>{pa}/{na}</td><td>{pb}/{nb}</td>
                        <td className={d > 0 ? 'ev-ok' : 'muted'}>{d > 0 ? `+${Math.round(d * 100)} pts` : d < 0 ? `${Math.round(d * 100)} pts` : ''}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}

      {!summary && (
        <div className="dashboard"><p className="muted">
          Press <strong>Start</strong>. The first questions will look the same for both assistants; once the one with memory has failed a few
          questions of a kind, it begins to get that kind right while the other keeps repeating the mistake.
        </p></div>
      )}
    </div>
  )
}
