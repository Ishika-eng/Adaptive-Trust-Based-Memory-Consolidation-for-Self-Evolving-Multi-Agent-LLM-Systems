import { useState } from 'react'
import './App.css'
import Assistant from './pages/Assistant'
import BenchmarkLab from './pages/BenchmarkLab'
import MemoryExplorer from './pages/MemoryExplorer'
import Results from './pages/Results'
import Architecture from './pages/Architecture'

const PAGES = {
  assistant: { label: 'Assistant', component: Assistant },
  lab: { label: 'Benchmark lab', component: BenchmarkLab },
  memory: { label: 'Memory', component: MemoryExplorer },
  results: { label: 'Results', component: Results },
  architecture: { label: 'Architecture', component: Architecture },
}

export default function App() {
  const [page, setPage] = useState('assistant')
  const Page = PAGES[page].component

  return (
    <div className="app">
      <header className="app-header">
        <h1>ATMC <span className="muted">— a coding assistant that remembers what works</span></h1>
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
        The memory behind it is ATMC (Adaptive Trust-based Memory Consolidation): notes the assistant can't trust are filtered out.
      </footer>
    </div>
  )
}
