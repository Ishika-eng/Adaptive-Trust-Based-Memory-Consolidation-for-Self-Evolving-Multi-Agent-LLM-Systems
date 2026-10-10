const PARTS = [
  { n: 1, name: 'Goal', role: 'Receive or generate a task to work on', status: 'roadmap' },
  { n: 2, name: 'Planner', role: 'Break the goal into concrete steps (never sees memory)', status: 'implemented' },
  { n: 3, name: 'Actor', role: 'Write code / solve the task, using retrieved memory', status: 'implemented' },
  { n: 4, name: 'Checker', role: 'Run the real tests in a sandbox (code) or compare to gold (math)', status: 'implemented' },
  { n: 5, name: 'Memory (ATMC)', role: 'Decide what to keep, trust, and forget', status: 'built' },
  { n: 6, name: 'Reflection', role: 'Turn outcomes into reusable insight', status: 'implemented' },
  { n: 7, name: 'Curriculum', role: 'Choose what to attempt next (fixed / random / weakness-targeted)', status: 'implemented' },
  { n: 8, name: 'Persistence', role: 'Memory and history saved in SQLite across restarts', status: 'implemented' },
]

const STATUS_LABEL = {
  built: '✓ Built & validated',
  implemented: 'Implemented — evaluation in progress',
  roadmap: 'Roadmap',
}

const ROADMAP = [
  { month: 'Done', focus: 'Working loop', items: 'Planner, sandboxed code Actor + test Checker, Reflection, Curriculum, persistence, real embeddings, baselines, paired statistics' },
  { month: 'In progress', focus: 'Full evaluation', items: 'More seeds (free-tier API quota limits runs per day), ablation on both domains, held-out scoring for the curriculum' },
  { month: 'Remaining', focus: 'Deliverables', items: 'Paper draft, final documentation, demo video' },
]

export default function Architecture() {
  return (
    <div>
      <div className="page-header">
        <div>
          <h2>Architecture</h2>
          <p className="muted">A self-evolving agent needs eight parts, working as a repeating loop. Here's where each part stands — implemented does not mean validated.</p>
        </div>
      </div>

      <div className="loop-strip">
        <span>Goal</span><span className="loop-arrow">→</span>
        <span>Plan</span><span className="loop-arrow">→</span>
        <span>Act</span><span className="loop-arrow">→</span>
        <span>Check</span><span className="loop-arrow">→</span>
        <span className="loop-highlight">Update Memory</span><span className="loop-arrow">→</span>
        <span>Reflect</span><span className="loop-arrow">→</span>
        <span>pick next goal</span><span className="loop-arrow">→</span>
        <span>repeat</span>
      </div>

      <table className="explorer-table arch-table">
        <thead>
          <tr><th>#</th><th>Part</th><th>Role</th><th>Status</th></tr>
        </thead>
        <tbody>
          {PARTS.map(p => (
            <tr key={p.n} className={p.status === 'built' ? 'row--highlight' : ''}>
              <td>{p.n}</td>
              <td>{p.name}</td>
              <td>{p.role}</td>
              <td>
                <span className={`pill ${p.status === 'built' ? 'pill--good' : p.status === 'implemented' ? 'pill--impl' : 'pill--pending'}`}>
                  {STATUS_LABEL[p.status]}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <p className="arch-note">
        The memory mechanism was validated first, in isolation. The rest of the loop is now built around it; its
        individual contribution (planner, reflection, curriculum) is still being measured and is not yet claimed.
      </p>

      <h3 className="roadmap-title">Project status</h3>
      <table className="explorer-table roadmap-table">
        <thead>
          <tr><th>Phase</th><th>Focus</th><th>Key additions</th></tr>
        </thead>
        <tbody>
          {ROADMAP.map(r => (
            <tr key={r.month}>
              <td><strong>{r.month}</strong></td>
              <td>{r.focus}</td>
              <td>{r.items}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
