import { useCallback, useEffect, useRef, useState } from 'react'
import { getJSON, postJSON } from '../api'

const NAMES = { plan_first: 'Plan first', direct: 'Direct', revise: 'Write, then review' }
const ROLE = {
  champion: { label: 'Champion', cls: 'pill--good' },
  baseline: { label: 'Default', cls: 'pill--pending' },
  challenger: { label: 'Challenger', cls: 'pill--impl' },
}
const pct = (x) => (x == null ? '–' : `${Math.round(x * 100)}%`)

function RateBar({ row }) {
  if (!row.trials) return <span className="muted">not tried yet</span>
  return (
    <div className="rate">
      <div className="rate-track"><div className="rate-fill" style={{ width: `${row.rate * 100}%` }} /></div>
      <span className="num">{pct(row.rate)} <span className="muted">({row.wins}/{row.trials})</span></span>
    </div>
  )
}

function CategoryCard({ cat, minTrials }) {
  const tried = cat.strategies.some(s => s.trials > 0)
  return (
    <div className="strat-card">
      <div className="strat-card-head">
        <h3>{cat.label}</h3>
        <span className={`pill ${cat.role === 'promoted' ? 'pill--good' : 'pill--pending'}`}>
          {cat.role === 'promoted' ? `${NAMES[cat.champion]} promoted` : 'Using the default'}
        </span>
      </div>
      {!tried ? <p className="muted">No attempts on this kind of problem yet.</p> : (
        <table className="explorer-table">
          <thead><tr><th>Approach</th><th>Success rate</th><th>Evidence</th><th /></tr></thead>
          <tbody>
            {cat.strategies.map(s => (
              <tr key={s.strategy} className={s.role === 'champion' ? 'row--highlight' : ''}>
                <td>{NAMES[s.strategy]}</td>
                <td><RateBar row={s} /></td>
                <td className="muted">{s.enough_evidence ? 'enough' : `needs ${minTrials - s.trials} more`}</td>
                <td><span className={`pill ${ROLE[s.role].cls}`}>{ROLE[s.role].label}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

export default function Evolution() {
  const [board, setBoard] = useState(null)
  const [error, setError] = useState(null)
  const [running, setRunning] = useState(null)         // { done, total } while practising
  const stop = useRef(false)

  const load = useCallback(async () => {
    try { setBoard(await getJSON('/api/chat/strategies')); setError(null) }
    catch (e) { setError(`Could not reach the backend: ${e.message}`) }
  }, [])
  useEffect(() => { load() }, [load])

  const practise = async (n) => {
    stop.current = false
    setError(null)
    try {
      for (let i = 0; i < n && !stop.current; i++) {
        setRunning({ done: i, total: n })
        const s = await getJSON('/api/chat/suggest')
        if (!s.id) break
        await postJSON('/api/chat/practise', { task_id: s.id })
        await load()
      }
    } catch (e) { setError(e.message) } finally { setRunning(null); await load() }
  }

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>How the assistant evolves</h2>
          <p className="muted">
            Memory isn’t the only thing that learns. The assistant also tracks <em>which way of solving</em> works best
            for each kind of problem, and changes how it works when the evidence says so.
          </p>
        </div>
        <div className="evo-actions">
          {running ? (
            <button className="btn btn--ghost" onClick={() => { stop.current = true }}>Stop after this one ({running.done + 1}/{running.total})</button>
          ) : (
            <button className="btn btn--primary" onClick={() => practise(5)} disabled={!board?.enabled}>Let it practise 5 tasks</button>
          )}
        </div>
      </div>

      {error && <div className="error-banner" role="alert">{error}</div>}
      {board && !board.enabled && <p className="muted">Strategy evolution is turned off in the config.</p>}

      {board && (
        <>
          <div className="dashboard">
            <h2>The three approaches</h2>
            <div className="strat-legend">
              {Object.entries(board.descriptions).map(([k, d]) => (
                <div key={k}><strong>{NAMES[k]}</strong>{k === board.baseline && <span className="pill pill--pending">default</span>}<p className="muted">{d}</p></div>
              ))}
            </div>
            <ol className="how-list muted">
              <li>While <strong>practising</strong> (tasks with real tests) it deliberately tries the approach it has tested least.</li>
              <li>An approach is only <strong>promoted</strong> once it has {board.min_trials}+ attempts and its success rate, with a safety margin for small samples, beats the default’s.</li>
              <li>Promotion is recomputed from the evidence every time, so a promoted approach that starts failing is <strong>rolled back</strong> automatically.</li>
            </ol>
          </div>

          <div className="strat-grid">
            {board.categories.map(c => <CategoryCard key={c.category} cat={c} minTrials={board.min_trials} />)}
          </div>

          <div className="dashboard">
            <h2>Recent attempts <span className="muted" style={{ fontWeight: 400 }}>({board.total} recorded)</span></h2>
            {board.events.length === 0 ? (
              <p className="muted">Nothing yet. Press “Let it practise 5 tasks”, or ask questions with tests in the Assistant.</p>
            ) : (
              <table className="explorer-table">
                <thead><tr><th /><th>Type</th><th>Approach</th><th>Mode</th><th>Task</th></tr></thead>
                <tbody>
                  {board.events.map((e, i) => (
                    <tr key={i}>
                      <td>{e.success === null ? '…' : e.success ? '✓' : '✗'}</td>
                      <td>{e.category}</td><td>{NAMES[e.strategy]}</td>
                      <td className="muted">{e.mode === 'practise' ? 'practice' : 'your question'}</td>
                      <td className="muted">{e.prompt.replace(/\s+/g, ' ')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </>
      )}
    </div>
  )
}
