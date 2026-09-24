const PARTS = [
  { n: 1, name: 'Goal', role: 'Receive or generate a task to work on', status: 'roadmap' },
  { n: 2, name: 'Planner', role: 'Break the goal into concrete steps', status: 'roadmap' },
  { n: 3, name: 'Actor', role: 'Actually do something — run code, use a tool', status: 'roadmap' },
  { n: 4, name: 'Checker', role: 'Verify whether the attempt succeeded', status: 'roadmap' },
  { n: 5, name: 'Memory (ATMC)', role: 'Decide what to keep, trust, and forget', status: 'built' },
  { n: 6, name: 'Reflection', role: 'Turn outcomes into reusable insight', status: 'roadmap' },
  { n: 7, name: 'Curriculum', role: 'Choose what to attempt next, autonomously', status: 'roadmap' },
  { n: 8, name: 'Persistence', role: 'Keep accumulating state across sessions', status: 'roadmap' },
]

const ROADMAP = [
  { month: 'Month 1', focus: 'Harden the core', items: 'Full ablation matrix at 5+ seeds, a real Planner agent, self-generated reflections' },
  { month: 'Month 2', focus: 'Give the agent something to act in', items: 'Coding-assistant domain (HumanEval/MBPP), code-execution checker, sandboxed tool execution, persistent storage' },
  { month: 'Month 3', focus: 'Autonomy', items: 'Self-directed curriculum, a reusable skill library, final full evaluation, paper write-up' },
]

export default function Architecture() {
  return (
    <div>
      <div className="page-header">
        <div>
          <h2>Architecture</h2>
          <p className="muted">A self-evolving agent needs eight parts, working as a repeating loop. Here's where this project actually stands.</p>
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
                <span className={`pill ${p.status === 'built' ? 'pill--good' : 'pill--pending'}`}>
                  {p.status === 'built' ? '✓ Built & validated' : 'Roadmap'}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <p className="arch-note">
        We deliberately built and rigorously validated the hardest, most overlooked piece — reliable memory —
        before building the rest of the loop around it. A self-evolving agent with an unreliable memory isn't
        worth building at all, so proving the memory mechanism works came first.
      </p>

      <h3 className="roadmap-title">Roadmap to the Full Agent</h3>
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
