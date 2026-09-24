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
        <p className="muted">Empty — go run the demo first to populate this notebook.</p>
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

export default function MemoryExplorer() {
  const [staticItems, setStaticItems] = useState([])
  const [atmcItems, setAtmcItems] = useState([])
  const [loading, setLoading] = useState(false)

  const load = () => {
    setLoading(true)
    Promise.all([getJSON('/api/memory/static'), getJSON('/api/memory/atmc')])
      .then(([s, a]) => { setStaticItems(s); setAtmcItems(a) })
      .finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [])

  return (
    <div>
      <div className="page-header">
        <div>
          <h2>Memory Bank Explorer</h2>
          <p className="muted">Full, live contents of both notebooks — sort by trust to see which notes ATMC has learned to rely on least.</p>
        </div>
        <button className="btn btn--ghost" onClick={load} disabled={loading}>{loading ? 'Refreshing…' : 'Refresh'}</button>
      </div>
      <div className="explorer-grid">
        <MemoryTable items={staticItems} storeLabel="Baseline" />
        <MemoryTable items={atmcItems} storeLabel="ATMC" />
      </div>
    </div>
  )
}
