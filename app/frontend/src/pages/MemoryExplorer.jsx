import { useEffect, useMemo, useRef, useState } from 'react'
import { API, getJSON, postJSON } from '../api'

const SOURCE = {
  tests: { label: 'tests', hint: 'Verified by real tests' },
  user: { label: 'you', hint: 'Verified by your verdict' },
  generated: { label: 'auto-check', hint: 'Verified by generated tests (weaker evidence)' },
  planted: { label: 'planted', hint: 'Deliberately wrong note from the poison demo' },
  compressed: { label: 'merged', hint: 'Several notes merged into one summary' },
  imported: { label: 'imported', hint: 'Came from an import file; trust restarts at the initial level' },
}
const FILTERS = {
  all: ['All notes', () => true],
  worked: ['Worked', m => m.correct === true],
  failed: ['Failed', m => m.correct === false && !m.is_poison],
  pinned: ['Pinned', m => m.pinned],
  gated: ['Blocked (low trust)', m => m.gated],
  planted: ['Planted', m => m.is_poison],
}
const SORTS = {
  trust: ['Trust', (a, b) => b.trust - a.trust],
  recent: ['Newest', (a, b) => b.created_at_step - a.created_at_step],
  used: ['Most used', (a, b) => b.hits - a.hits],
}
const trustClass = (t) => (t >= 0.5 ? 'trust-chip--good' : t >= 0.3 ? 'trust-chip--mid' : 'trust-chip--bad')
const title = (t) => (t.split('\n')[0] || '').replace(/^Task:\s*/, '')

function ManagedMemory() {
  const [notes, setNotes] = useState([])
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState('all')
  const [sort, setSort] = useState('trust')
  const [open, setOpen] = useState(null)
  const [msg, setMsg] = useState(null)
  const [error, setError] = useState(null)
  const fileRef = useRef(null)

  const load = async () => {
    try { setNotes(await getJSON('/api/memory/chat')); setError(null) }
    catch (e) { setError(`Could not load memory: ${e.message}`) }
  }
  useEffect(() => { load() }, [])

  const act = async (fn, okMsg) => {
    setError(null); setMsg(null)
    try { await fn(); if (okMsg) setMsg(okMsg); await load() } catch (e) { setError(e.message) }
  }
  const pin = (m) => act(() => postJSON('/api/memory/chat/pin', { uid: m.uid, pinned: !m.pinned }))
  const del = (m) => {
    if (!window.confirm(`Delete this note?\n\n${title(m.text).slice(0, 120)}`)) return
    act(async () => {
      const r = await fetch(`${API}/api/memory/chat/${m.uid}`, { method: 'DELETE' })
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || 'Delete failed')
    }, 'Note deleted.')
  }
  const exportAll = async () => {
    try {
      const data = await getJSON('/api/memory/chat/export')
      const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 1)], { type: 'application/json' }))
      const a = Object.assign(document.createElement('a'), { href: url, download: 'atmc-memory.json' })
      a.click(); URL.revokeObjectURL(url)
    } catch (e) { setError(e.message) }
  }
  const importFile = (e) => {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (!file) return
    act(async () => {
      const data = JSON.parse(await file.text())
      const r = await postJSON('/api/memory/chat/import', { embedder: data.embedder, items: data.items })
      setMsg(`Imported ${r.added} note(s)${r.skipped_duplicates ? `, skipped ${r.skipped_duplicates} duplicate(s)` : ''}. Imported notes start at the initial trust level and must earn more.`)
    })
  }

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase()
    return notes.filter(FILTERS[filter][1]).filter(m => !q || m.text.toLowerCase().includes(q)).sort(SORTS[sort][1])
  }, [notes, query, filter, sort])
  const count = (k) => notes.filter(FILTERS[k][1]).length

  return (
    <div className="dashboard">
      <div className="mem-toolbar">
        <input type="text" value={query} onChange={e => setQuery(e.target.value)} placeholder="Search notes…" aria-label="Search notes" />
        <select value={filter} onChange={e => setFilter(e.target.value)} aria-label="Filter">
          {Object.entries(FILTERS).map(([k, [l]]) => <option key={k} value={k}>{l} ({count(k)})</option>)}
        </select>
        <select value={sort} onChange={e => setSort(e.target.value)} aria-label="Sort">
          {Object.entries(SORTS).map(([k, [l]]) => <option key={k} value={k}>Sort: {l}</option>)}
        </select>
        <span className="mem-spacer" />
        <button className="btn btn--ghost" onClick={exportAll}>Export</button>
        <button className="btn btn--ghost" onClick={() => fileRef.current?.click()}>Import</button>
        <input ref={fileRef} type="file" accept="application/json,.json" hidden onChange={importFile} />
      </div>
      {error && <div className="error-banner" role="alert" style={{ margin: '0 0 12px' }}>{error}</div>}
      {msg && <p className="mem-msg" role="status">{msg}</p>}

      {notes.length === 0 ? (
        <p className="muted">No notes yet. Ask the assistant something (ideally with tests) and what it learns will appear here.</p>
      ) : shown.length === 0 ? <p className="muted">No notes match.</p> : (
        <div className="table-scroll">
          <table className="explorer-table">
            <thead><tr><th>Note</th><th>Trust</th><th>Evidence</th><th>Source</th><th>Use</th><th /></tr></thead>
            <tbody>
              {shown.map(m => (
                <tr key={m.uid} className={m.is_poison ? 'row--poison' : m.pinned ? 'row--highlight' : ''}>
                  <td className="note-cell">
                    <button className="note-title" onClick={() => setOpen(open === m.uid ? null : m.uid)} aria-expanded={open === m.uid}>
                      {m.pinned && <span title="Pinned: protected from forgetting and merging">📌 </span>}
                      <span className={`dot ${m.correct ? 'dot--ok' : 'dot--bad'}`} title={m.correct ? 'worked' : m.correct === false ? 'did not work' : 'merged summary'} />
                      {title(m.text).slice(0, 90) || '(note)'}
                    </button>
                    {m.gated && <span className="badge badge--poison" title="Below the trust threshold: never retrieved">blocked</span>}
                    {open === m.uid && <pre className="how-pre">{m.text}</pre>}
                  </td>
                  <td><span className={`trust-chip ${trustClass(m.trust)}`}>{m.trust.toFixed(2)}</span></td>
                  <td className="num" title="Times relied on in a verified-correct / wrong answer">
                    <span className="ev-ok">✓ {m.successes}</span> <span className="ev-bad">✗ {m.failures}</span>
                  </td>
                  <td><span className="src" title={SOURCE[m.source]?.hint}>{SOURCE[m.source]?.label || m.source}</span>{m.version > 1 && <span className="muted"> v{m.version}</span>}</td>
                  <td className="muted num">{m.hits}× · {m.last_used_step == null ? 'never' : `step ${m.last_used_step}`}</td>
                  <td className="row-actions">
                    <button className="link-btn" onClick={() => pin(m)}>{m.pinned ? 'Unpin' : 'Pin'}</button>
                    <button className="link-btn link-btn--danger" onClick={() => del(m)}>Delete</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function ReadOnlyTable({ store }) {
  const [items, setItems] = useState([])
  useEffect(() => { getJSON(`/api/memory/${store}`).then(setItems).catch(() => setItems([])) }, [store])
  return (
    <div className="explorer-table-wrap">
      <h3>Benchmark lab memory <span className="muted">({items.length} notes)</span></h3>
      {items.length === 0 ? <p className="muted">Empty. Run some tasks in the Benchmark lab first.</p> : (
        <table className="explorer-table">
          <thead><tr><th>Trust</th><th>Hits</th><th>Planted</th><th>Text</th></tr></thead>
          <tbody>
            {[...items].sort((a, b) => b.trust - a.trust).map((m, i) => (
              <tr key={i} className={m.is_poison ? 'row--poison' : ''}>
                <td><span className={`trust-chip ${trustClass(m.trust)}`}>{m.trust.toFixed(2)}</span></td>
                <td>{m.hits}</td><td>{m.is_poison ? '⚠ yes' : '—'}</td>
                <td className="explorer-text">{m.text}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

const VIEWS = { chat: 'Assistant', code: 'Lab · coding', math: 'Lab · math' }

export default function MemoryExplorer() {
  const [view, setView] = useState('chat')
  return (
    <div>
      <div className="page-header">
        <div>
          <h2>Memory</h2>
          <p className="muted">
            Everything the assistant remembers. Each note carries its evidence: how often it helped or misled, who
            verified it, and how much it is trusted. Pin a note to protect it from forgetting and merging, or delete it.
          </p>
        </div>
        <div className="seg">
          {Object.entries(VIEWS).map(([k, l]) => (
            <button key={k} className={`seg-btn ${view === k ? 'seg-btn--on' : ''}`} onClick={() => setView(k)}>{l}</button>
          ))}
        </div>
      </div>
      {view === 'chat' ? <ManagedMemory /> : <ReadOnlyTable store={view} />}
    </div>
  )
}
