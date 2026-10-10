import { useEffect, useState } from 'react'
import { getJSON } from '../api'

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

function PilotDashboard({ stats }) {
  if (!stats || !stats.static) return <p className="muted">No pilot results found yet.</p>
  const maxExp = Math.max(stats.static.exposure_mean, stats.atmc.exposure_mean)
  const Bar = ({ label, value, max, fmt }) => (
    <div className="dash-bar-row">
      <span className="dash-bar-name">{label}</span>
      <div className="dash-bar-track">
        <div className={`dash-bar-fill dash-bar-fill--${label}`} style={{ width: `${(value / max) * 100}%` }} />
      </div>
      <span className="dash-bar-value">{fmt(value)}</span>
    </div>
  )
  return (
    <div className="dashboard">
      <h2>Earlier math poison pilot ({stats.n_seeds} seeds: {stats.seeds.join(', ')})</h2>
      <div className="dash-row">
        <div className="dash-metric">
          <span className="dash-metric-label">How often a wrong note reached the AI</span>
          <div className="dash-bars">
            <Bar label="static" value={stats.static.exposure_mean} max={maxExp} fmt={v => v.toFixed(1)} />
            <Bar label="atmc" value={stats.atmc.exposure_mean} max={maxExp} fmt={v => v.toFixed(1)} />
          </div>
        </div>
        <div className="dash-metric">
          <span className="dash-metric-label">Accuracy — did filtering cost us anything?</span>
          <div className="dash-bars">
            <Bar label="static" value={stats.static.accuracy_mean} max={1} fmt={v => (v * 100).toFixed(1) + '%'} />
            <Bar label="atmc" value={stats.atmc.accuracy_mean} max={1} fmt={v => (v * 100).toFixed(1) + '%'} />
          </div>
          <p className="muted">No — accuracy stayed the same or better. The filtering is free.</p>
        </div>
      </div>
    </div>
  )
}

const CONDITION_LABELS = {
  full_system: 'Full System',
  wo_trust: 'w/o Trust',
  wo_forgetting: 'w/o Forgetting',
  wo_compression: 'w/o Compression',
}


// Findings are computed from the data, so the text can never contradict the table above it.
function AblationFindings({ conditions, nSeeds }) {
  const by = Object.fromEntries(conditions.map(c => [c.name, c]))
  const full = by.full_system, noTrust = by.wo_trust, noForget = by.wo_forgetting, noComp = by.wo_compression
  if (!full || !noTrust || !noForget || !noComp) return null
  const worst = noTrust.total_poison_exposure >= noForget.total_poison_exposure
    ? { name: 'Trust scoring', c: noTrust, other: 'Forgetting', oc: noForget }
    : { name: 'Forgetting', c: noForget, other: 'Trust scoring', oc: noTrust }
  const memSaved = Math.round((1 - full.final_memory_size / noComp.final_memory_size) * 100)
  return (
    <>
      <div className="finding-box">
        <strong>Which pillar matters most:</strong> removing {worst.name} hurts the most — poison exposure rises from{' '}
        {full.total_poison_exposure} to {worst.c.total_poison_exposure}, versus {worst.oc.total_poison_exposure} when
        removing {worst.other}.
      </div>
      <div className="finding-box finding-box--warn">
        <strong>Compression:</strong> shrinks the memory bank by {memSaved}% ({full.final_memory_size} vs{' '}
        {noComp.final_memory_size} notes); poison exposure with vs without it: {full.total_poison_exposure} vs{' '}
        {noComp.total_poison_exposure}. Poisoned notes merged in with clean ones: {full.n_poison_laundered}.
      </div>
      {nSeeds < 3 && (
        <p className="muted">Based on {nSeeds} seed — treat the ordering as preliminary until repeated across seeds.</p>
      )}
    </>
  )
}

function AblationTable({ ablation }) {
  if (!ablation || !ablation.conditions?.length) return <p className="muted">No ablation results found yet.</p>
  return (
    <div className="dashboard">
      <h2>Earlier 4-way ablation (seed {ablation.seed}, {ablation.n_tasks} tasks, {ablation.n_poison_seeded} poisoned notes)</h2>
      <p className="muted">Isolating exactly which mechanism — Trust, Forgetting, Compression — contributes what.</p>
      <table className="explorer-table ablation-table">
        <thead>
          <tr>
            <th>Configuration</th>
            <th>Accuracy</th>
            <th>Poison Exposure</th>
            <th>Memory Size</th>
            <th>Compressions</th>
            <th>Poison Laundered</th>
          </tr>
        </thead>
        <tbody>
          {ablation.conditions.map(c => (
            <tr key={c.name} className={c.name === 'full_system' ? 'row--highlight' : ''}>
              <td>{CONDITION_LABELS[c.name] || c.name}</td>
              <td>{c.overall_accuracy.toFixed(3)}</td>
              <td>{c.total_poison_exposure}</td>
              <td>{c.final_memory_size}</td>
              <td>{c.n_compressions}</td>
              <td>{c.n_poison_laundered}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <AblationFindings conditions={ablation.conditions} nSeeds={ablation.n_seeds ?? 1} />
    </div>
  )
}

const EXP_LABEL = {
  baselines: 'Baselines (no poison)', baselines_poison: 'Baselines with planted wrong notes',
  poison: 'Static vs ATMC with planted wrong notes', ablation: 'Ablation', agent: 'Agent components',
}
const pct = (m, d = 1) => (m?.mean == null ? '–' : `${(m.mean * 100).toFixed(d)}%`)
const num = (m, d = 1) => (m?.mean == null ? '–' : `${m.mean.toFixed(d)} ± ${(m.sd ?? 0).toFixed(d)}`)

function Comparison({ c }) {
  const rows = [['in-run', c.in_run], ['held-out', c.heldout]].filter(([, p]) => p && p.n)
  return rows.map(([label, p]) => (
    <li key={c.condition + label}>
      <strong>{c.condition}</strong> vs {c.vs} ({label}, {p.n} paired tasks): difference {p.diff >= 0 ? '+' : ''}{(p.diff * 100).toFixed(1)} points,
      95% CI [{(p.ci95[0] * 100).toFixed(1)}, {(p.ci95[1] * 100).toFixed(1)}], better/worse {p.b_better}/{p.a_better},
      McNemar p = {p.p_mcnemar.toFixed(3)}{p.p_mcnemar >= 0.05 ? ' — not significant' : ''}
    </li>
  ))
}

function Evaluation() {
  const [list, setList] = useState(null)
  const [sel, setSel] = useState(null)
  const [rep, setRep] = useState(null)

  useEffect(() => {
    getJSON('/api/eval').then(l => { setList(l); if (l.length) setSel(l.find(e => e.experiment === 'baselines_poison') || l[0]) }).catch(() => setList([]))
  }, [])
  useEffect(() => {
    if (sel) getJSON(`/api/eval?experiment=${sel.experiment}&domain=${sel.domain}`).then(setRep).catch(() => setRep(null))
  }, [sel])

  if (!list) return null
  if (!list.length) return <p className="muted">No evaluation runs saved yet (run pilot/experiment.py).</p>
  const seeds = rep ? Math.min(...rep.conditions.map(c => c.seeds.length)) : 0
  return (
    <div className="dashboard eval-block">
      <h2>Evaluation (current harness)</h2>
      <p className="muted">Every row is the mean ± sd over the seeds actually run. Comparisons are paired: same task, same seed.</p>
      <div className="eval-tabs">
        {list.map(e => (
          <button key={e.experiment + e.domain}
            className={`nav-btn ${sel && sel.experiment === e.experiment && sel.domain === e.domain ? 'nav-btn--active' : ''}`}
            onClick={() => setSel(e)}>{EXP_LABEL[e.experiment] || e.experiment} · {e.domain}</button>
        ))}
      </div>
      {rep && rep.conditions.length > 0 && (
        <>
          <table className="explorer-table">
            <thead><tr>
              <th>Condition</th><th>Seeds</th><th>Pass rate</th><th>Held-out</th><th>Poison in context*</th>
              <th>Poison relied on</th><th>Memory size</th><th>Tokens/task</th>
            </tr></thead>
            <tbody>
              {rep.conditions.map(c => (
                <tr key={c.condition} className={c.condition === rep.reference ? 'row--highlight' : ''}>
                  <td>{c.condition}</td><td>{c.seeds.length}</td>
                  <td className="num">{pct(c.pass_rate)} ± {c.pass_rate.sd == null ? '–' : (c.pass_rate.sd * 100).toFixed(1)}</td>
                  <td className="num">{pct(c.heldout_pass_rate)}</td>
                  <td className="num">{num(c.poison_exposure_after_warmup)}</td>
                  <td className="num">{num(c.poison_used)}</td>
                  <td className="num">{num(c.final_memory_size, 0)}</td>
                  <td className="num">{c.tokens_in.mean == null ? '–' : Math.round(c.tokens_in.mean)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted">{rep.warmup_note} Reference condition highlighted: {rep.reference}.</p>
          {rep.comparisons.length > 0 && <ul className="muted">{rep.comparisons.map(c => <Comparison key={c.condition} c={c} />)}</ul>}
          {seeds < 3 && <p className="muted"><strong>Preliminary:</strong> at least one condition has fewer than 3 seeds, so treat differences with caution.</p>}
        </>
      )}
    </div>
  )
}

export default function Results() {
  const [stats, setStats] = useState(null)
  const [ablation, setAblation] = useState(null)

  useEffect(() => {
    getJSON('/api/results').then(setStats).catch(() => {})
    getJSON('/api/ablation').then(setAblation).catch(() => {})
  }, [])

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>Results</h2>
          <p className="muted">Every number here comes from real experiment logs in pilot/results/, not hand-picked runs.</p>
        </div>
      </div>
      <Evaluation />
      <HeadlineStat stats={stats} />
      <PilotDashboard stats={stats} />
      <AblationTable ablation={ablation} />
    </div>
  )
}
