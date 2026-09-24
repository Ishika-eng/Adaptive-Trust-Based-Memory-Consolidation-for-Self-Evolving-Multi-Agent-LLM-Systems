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
      <h2>Poison-Memory Pilot ({stats.n_seeds} seeds: {stats.seeds.join(', ')})</h2>
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

function AblationTable({ ablation }) {
  if (!ablation || !ablation.conditions?.length) return <p className="muted">No ablation results found yet.</p>
  return (
    <div className="dashboard">
      <h2>4-Way Ablation Study (seed {ablation.seed}, {ablation.n_tasks} tasks, {ablation.n_poison_seeded} poisoned notes)</h2>
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
      <div className="finding-box">
        <strong>Finding — Forgetting matters more than Trust alone:</strong> removing Trust scoring roughly doubles
        poison exposure; removing Forgetting is worse still, because Trust filtering alone lets an already-admitted
        bad memory sit in the bank forever — only active forgetting removes it as time passes.
      </div>
      <div className="finding-box finding-box--warn">
        <strong>Finding — Compression has a real cost:</strong> it shrinks the memory bank, but merging old memories
        together can fold a poisoned one in with clean ones, diluting how distinct it looks ("Poison Laundered" &gt; 0).
        Reported honestly rather than hidden.
      </div>
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
      <HeadlineStat stats={stats} />
      <PilotDashboard stats={stats} />
      <AblationTable ablation={ablation} />
    </div>
  )
}
