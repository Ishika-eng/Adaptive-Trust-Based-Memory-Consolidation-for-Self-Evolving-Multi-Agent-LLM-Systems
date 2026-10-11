import { useCallback, useEffect, useState } from 'react'
import './App.css'
import { getJSON } from './api'
import Logo from './lib/Logo'
import Assistant from './pages/Assistant'
import BenchmarkLab from './pages/BenchmarkLab'
import MemoryExplorer from './pages/MemoryExplorer'
import Results from './pages/Results'
import Architecture from './pages/Architecture'

const PAGES = {
  assistant: { label: 'Assistant', component: Assistant },
  memory: { label: 'Memory', component: MemoryExplorer },
  results: { label: 'Results', component: Results },
  lab: { label: 'Benchmark lab', component: BenchmarkLab },
  architecture: { label: 'Architecture', component: Architecture },
}

const THEMES = ['system', 'light', 'dark']
const THEME_ICON = {
  system: <path d="M4 5h16v11H4zM9 20h6M12 16v4" />,
  light: <><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></>,
  dark: <path d="M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5z" />,
}

function readTheme() {
  try { const t = localStorage.getItem('atmc-theme'); return THEMES.includes(t) ? t : 'system' } catch { return 'system' }
}

export default function App() {
  const [page, setPage] = useState('assistant')
  const [theme, setTheme] = useState(readTheme)
  const [health, setHealth] = useState(null)         // null = checking, false = offline, object = online
  const Page = PAGES[page].component

  useEffect(() => {
    const root = document.documentElement
    if (theme === 'system') root.removeAttribute('data-theme'); else root.setAttribute('data-theme', theme)
    try { localStorage.setItem('atmc-theme', theme) } catch { /* storage unavailable */ }
  }, [theme])

  const check = useCallback(() => {
    getJSON('/api/chat/summary').then(setHealth).catch(() => setHealth(false))
  }, [])
  useEffect(() => { check(); const id = setInterval(check, 20000); return () => clearInterval(id) }, [check])

  const cycleTheme = () => setTheme(THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length])
  const online = !!health

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand"><Logo /> ATMC <small>Coding Assistant</small></div>
        <nav className="app-nav" aria-label="Main">
          {Object.entries(PAGES).map(([key, { label }]) => (
            <button key={key} className={`nav-btn ${page === key ? 'nav-btn--active' : ''}`}
              aria-current={page === key ? 'page' : undefined} onClick={() => setPage(key)}>{label}</button>
          ))}
        </nav>
        <div className="header-right">
          <div className={`status ${health === null ? '' : online ? 'status--ok' : 'status--down'}`} role="status">
            <i className="status-dot" />
            <span>{health === null ? 'Connecting…' : online ? (health.isolation === 'sandbox-exec' ? 'Online · sandboxed' : 'Online') : 'Backend offline'}</span>
          </div>
          <button className="icon-btn" onClick={cycleTheme} aria-label={`Theme: ${theme}. Click to change`} title={`Theme: ${theme}`}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">{THEME_ICON[theme]}</svg>
          </button>
        </div>
      </header>

      {health === false && (
        <div className="offline" role="alert">
          The backend isn't reachable. Start it with <code>cd app/backend &amp;&amp; ../../pilot/.venv/bin/python -m uvicorn main:app --port 8000</code>
          <button className="link-btn" onClick={check}>Retry</button>
        </div>
      )}

      <main className="app-main"><Page /></main>

      <footer className="app-footer">
        Memory by ATMC — Adaptive Trust-based Memory Consolidation. Notes the assistant can't trust are filtered out.
      </footer>
    </div>
  )
}
