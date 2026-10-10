import { useEffect, useState } from 'react'
import { getJSON } from '../api'

const COLUMNS = [
  { key: 'trust', label: 'Trust' },
  { key: 'hits', label: 'Hits' },
  { key: 'created_at_step', label: 'Age (step)' },
  { key: 'is_poison', label: 'Poisoned' },
]

function MemoryTable({ items, storeLabel }) {
  const [sortKey, setSortKey] = useState('trust')
  const [sortDir, setSortDir] = useState('desc')

  const sorted = [...items].sort((a, b) => {
    const av = a[sortKey], bv = b[sortKey]
    const cmp = av === bv ? 0 : av > bv ? 1 : -1
    return sortDir === 'asc' ? cmp : -cmp
  })

  const toggleSort = (key) => {
    if (key === sortKey) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortKey(key); setSortDir('desc') }
  }

  return (
    <div className="explorer-table-wrap">
      <h3>{storeLabel} <span className="muted">({items.length} notes)</span></h3>
      {items.length === 0 ? (
        <p className="muted">Empty — run some tasks on the Assistant (or Demo) page first.</p>
      ) : (
        <table className="explorer-table">
          <thead>
            <tr>
              {COLUMNS.map(c => (
                <th key={c.key} onClick={() => toggleSort(c.key)} className="sortable-th">
                  {c.label}{sortKey === c.key ? (sortDir === 'asc' ? ' ▲' : ' ▼') : ''}
                </th>
              ))}
              <th>Text</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((m, i) => (
              <tr key={i} className={m.is_poison ? 'row--poison' : ''}>
                <td><span className={`trust-chip ${m.trust >= 0.5 ? 'trust-chip--good' : m.trust >= 0.3 ? 'trust-chip--mid' : 'trust-chip--bad'}`}>{m.trust.toFixed(2)}</span></td>
                <td>{m.hits}</td>
                <td>{m.created_at_step}</td>
                <td>{m.is_poison ? '⚠ yes' : '—'}</td>
                <td className="explorer-text">{m.text}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

const VIEWS = {
  compare: { label: 'Side-by-side demo', stores: [['static', 'Baseline (remember everything)'], ['atmc', 'ATMC']] },
  code: { label: 'Coding assistant', stores: [['code', 'Coding assistant memory (ATMC)']] },
  math: { label: 'Math assistant', stores: [['math', 'Math assistant memory (ATMC)']] },
}

export default function MemoryExplorer() {
  const [view, setView] = useState('code')
  const [tables, setTables] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const load = () => {
    setLoading(true); setError(null)
    Promise.all(VIEWS[view].stores.map(([k]) => getJSON(`/api/memory/${k}`)))
      .then(setTables)
      .catch(e => { setTables([]); setError(`Could not load memory: ${e.message}`) })
      .finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [view])

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>Memory Bank Explorer</h2>
          <p className="muted">Full, live contents of the memory — sort by trust to see which notes ATMC relies on least.</p>
        </div>
        <div className="seg">
          {Object.entries(VIEWS).map(([k, v]) => (
            <button key={k} className={`seg-btn ${view === k ? 'seg-btn--on' : ''}`} onClick={() => setView(k)}>{v.label}</button>
          ))}
        </div>
        <button className="btn btn--ghost" onClick={load} disabled={loading}>{loading ? 'Refreshing…' : 'Refresh'}</button>
      </div>
      {error && <div className="error-banner">{error}</div>}
      <div className="explorer-grid">
        {VIEWS[view].stores.map(([k, label], i) => <MemoryTable key={k} items={tables[i] || []} storeLabel={label} />)}
      </div>
    </div>
  )
}
