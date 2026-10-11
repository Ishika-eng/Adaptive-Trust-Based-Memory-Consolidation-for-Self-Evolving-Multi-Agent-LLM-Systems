const PARTS = [
  { n: 1, name: 'Goal', role: 'Receive or generate a task to work on', status: 'basic' },
  { n: 2, name: 'Planner', role: 'Break the goal into concrete steps (never sees memory). Held-out +6.7 pts, not significant', status: 'implemented' },
  { n: 3, name: 'Actor', role: 'Write code / solve the task, using retrieved memory', status: 'working' },
  { n: 4, name: 'Checker', role: 'Run the real tests in a sandbox (code) or compare to gold (math)', status: 'working' },
  { n: 5, name: 'Memory (ATMC)', role: 'Decide what to keep, trust, and forget', status: 'built' },
  { n: 6, name: 'Reflection', role: 'Turn outcomes into reusable insight', status: 'implemented' },
  { n: 7, name: 'Curriculum', role: 'Choose what to attempt next (fixed / random / weakness-targeted)', status: 'implemented' },
  { n: 8, name: 'Persistence', role: 'Memory and history saved in SQLite across restarts', status: 'working' },
]

const STATUS_LABEL = {
  working: '✓ Built & tested',
  basic: 'Basic — the task is the prompt',
  built: '✓ Built & validated',
  implemented: 'Evaluated — no significant gain',
  roadmap: 'Roadmap',
}

const ROADMAP = [
  { month: 'Done', focus: 'Working loop and evaluation', items: 'Planner, sandboxed code actor and test checker, reflection, curriculum, persistence, chat assistant with follow-ups and self-chosen practice; poison, ablation and agent-component experiments with paired statistics' },
  { month: 'Findings', focus: 'What the evidence shows', items: 'Trust-gated memory keeps planted wrong notes out (about 10× fewer in math, none after warm-up in code) at no accuracy cost; trust is the pillar that matters most; the extra components show no significant accuracy gain at this benchmark’s ceiling' },
  { month: 'Remaining', focus: 'Deliverables', items: 'Paper draft, final slides, demo video; optionally a larger held-out run to test the planner’s hint' },
]

export default function Architecture() {
  return (
    <div>
      <div className="page-header">
        <div>
          <h2>Architecture</h2>
          <p className="muted">A self-evolving agent needs eight parts, working as a repeating loop. Here's where each part stands, including what the evaluation did and did not show.</p>
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
                <span className={`pill ${p.status === 'built' || p.status === 'working' ? 'pill--good' : p.status === 'implemented' ? 'pill--impl' : 'pill--pending'}`}>
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
