import { useState } from 'react'
import './App.css'
import Demo from './pages/Demo'
import MemoryExplorer from './pages/MemoryExplorer'
import Results from './pages/Results'
import Architecture from './pages/Architecture'

const PAGES = {
  demo: { label: 'Demo', component: Demo },
  memory: { label: 'Memory Explorer', component: MemoryExplorer },
  results: { label: 'Results', component: Results },
  architecture: { label: 'Architecture', component: Architecture },
}

export default function App() {
  const [page, setPage] = useState('demo')
  const Page = PAGES[page].component

  return (
    <div className="app">
      <header className="app-header">
        <h1>ATMC <span className="muted">— Adaptive Trust-based Memory Consolidation</span></h1>
        <p className="lede">
          AI agents that remember their past mistakes can also remember them <em>wrong</em> — and then
          trust that wrong memory forever. ATMC scores every memory by how trustworthy, relevant, and
          fresh it is, so bad memories get filtered out and good ones don't get lost in the noise.
        </p>
      </header>

      <nav className="app-nav">
        {Object.entries(PAGES).map(([key, { label }]) => (
          <button
            key={key}
            className={`nav-btn ${page === key ? 'nav-btn--active' : ''}`}
            onClick={() => setPage(key)}
          >
            {label}
          </button>
        ))}
      </nav>

      <main className="app-main">
        <Page />
      </main>

      <footer className="app-footer muted">
        This is a live view into the actual backend (pilot/memory.py + pilot/stores.py) — the same code
        that produced the experiment results above, not a mockup.
      </footer>
    </div>
  )
}
