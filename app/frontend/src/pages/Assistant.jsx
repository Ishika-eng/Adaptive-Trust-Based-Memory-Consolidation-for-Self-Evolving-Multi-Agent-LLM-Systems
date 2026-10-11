import { useEffect, useRef, useState } from 'react'
import { getJSON, postJSON } from '../api'

const EXAMPLES = [
  'Write a function that merges overlapping intervals',
  'Write a function that checks whether a string of brackets is balanced',
  'Write a function that returns the n-th Fibonacci number efficiently',
]
const TEST_PLACEHOLDER = 'assert merge([[1, 3], [2, 6], [8, 10]]) == [[1, 6], [8, 10]]'

const STATUS = {
  passed: { label: 'Tests passed', cls: 'pill--good' },
  failed: { label: 'Tests failed', cls: 'pill--bad' },
  unverified: { label: 'Not verified — no tests given', cls: 'pill--impl' },
}

function CodeBlock({ code }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    try { await navigator.clipboard.writeText(code); setCopied(true); setTimeout(() => setCopied(false), 1500) } catch { /* clipboard unavailable */ }
  }
  return (
    <div className="codeblock">
      <div className="codeblock-bar"><span>python</span><button onClick={copy}>{copied ? 'Copied ✓' : 'Copy'}</button></div>
      <pre>{code || '(no code returned)'}</pre>
    </div>
  )
}

function Details({ turn }) {
  return (
    <details className="how">
      <summary>How I solved this</summary>
      {turn.plan && (<><h5>Plan (made without looking at memory)</h5><pre className="how-pre">{turn.plan}</pre></>)}
      <h5>What I remembered ({turn.retrieved.length})</h5>
      {turn.retrieved.length === 0 && <p className="muted">Nothing relevant yet — memory is still empty.</p>}
      {turn.retrieved.map((m, i) => (
        <div key={i} className={`mem-card ${m.used ? 'mem-card--used' : ''}`}>
          <div className="mem-card-top">
            {m.used && <span className="badge badge--used">✓ relied on this</span>}
            <span className="mem-card-hits">trust {m.trust.toFixed(2)}</span>
          </div>
          <p className="mem-card-text">{m.text.slice(0, 280)}{m.text.length > 280 ? '…' : ''}</p>
        </div>
      ))}
      {turn.lesson && (<><h5>Lesson I saved for next time</h5><pre className="how-pre">{turn.lesson}</pre></>)}
      <p className="muted">{turn.cost.tokens_in + turn.cost.tokens_out} tokens · memory lookup {turn.cost.retrieval_ms} ms</p>
    </details>
  )
}

function Feedback({ turn, onSend, busy }) {
  const [comment, setComment] = useState('')
  const [open, setOpen] = useState(false)
  if (turn.feedback) {
    return <p className="muted">{turn.feedback.correct ? '✓ You confirmed this worked — I saved it.' : '✗ You said it did not work — I saved what went wrong.'}</p>
  }
  if (turn.status !== 'unverified') return null
  return (
    <div className="feedback">
      <span>Did this work for you? I only learn from it once you tell me.</span>
      <button className="btn btn--good" disabled={busy} onClick={() => onSend(turn.id, true, '')}>👍 It worked</button>
      <button className="btn btn--ghost" disabled={busy} onClick={() => setOpen(o => !o)}>👎 It didn't</button>
      {open && (
        <div className="feedback-form">
          <input value={comment} onChange={e => setComment(e.target.value)} placeholder="What went wrong? (optional)" />
          <button className="btn btn--warn" disabled={busy} onClick={() => onSend(turn.id, false, comment)}>Send</button>
        </div>
      )}
    </div>
  )
}

function Turn({ turn, onFeedback, busy }) {
  const st = STATUS[turn.status]
  return (
    <>
      <div className="msg msg--user">
        {turn.practice ? (
          <div className="bubble bubble--practice">
            <strong>🎯 Practice task I chose myself</strong>
            <p>{turn.prompt.length > 320 ? turn.prompt.slice(0, 320) + '…' : turn.prompt}</p>
          </div>
        ) : (
          <div className="bubble bubble--user">
            {turn.followup && <span className="followup-tag">↪ follow-up</span>}
            <p>{turn.prompt}</p>
            {turn.tests && <pre className="bubble-tests">{turn.tests}</pre>}
          </div>
        )}
      </div>
      <div className="msg">
        <div className="bubble bubble--bot">
          <div className="bubble-head">
            <span className={`pill ${st.cls}`}>{st.label}</span>
            {turn.status === 'failed' && <span className="muted">{turn.stage.replace('_', ' ')}: {turn.detail}</span>}
          </div>
          {turn.explanation && <p className="explain">{turn.explanation}</p>}
          <CodeBlock code={turn.code} />
          {turn.status === 'unverified' && !turn.feedback && turn.output && (
            <p className="muted">It ran without errors and printed: <code>{turn.output}</code></p>
          )}
          {turn.status === 'unverified' && !turn.feedback && !turn.output && (
            <p className="muted">It ran without errors, but nothing checks that it's correct. Add tests to verify it.</p>
          )}
          <Feedback turn={turn} onSend={onFeedback} busy={busy} />
          <Details turn={turn} />
        </div>
      </div>
    </>
  )
}

function NextTask({ suggestion, onPractise, onAnother, busy }) {
  if (!suggestion) return null
  if (!suggestion.id) return <p className="muted">I've worked through every practice task I know.</p>
  return (
    <div className="next-task">
      <h4>My next practice task</h4>
      <p className="muted">{suggestion.reason}</p>
      <details>
        <summary>See the task</summary>
        <pre className="how-pre">{suggestion.question}</pre>
      </details>
      <div className="next-task-actions">
        <button className="btn btn--primary" disabled={busy} onClick={() => onPractise(suggestion.id)}>Let me practise it</button>
        <button className="btn btn--ghost" disabled={busy} onClick={onAnother}>Another</button>
      </div>
      <p className="muted">I solve it against its real tests, so what I learn is verified.</p>
    </div>
  )
}

function Sidebar({ summary, notes, onReset, busy, suggestion, onPractise, onAnother }) {
  const top = [...notes].sort((a, b) => b.trust - a.trust).slice(0, 8)
  return (
    <aside className="side">
      <h3>What I've learned</h3>
      {summary && (
        <div className="side-stats">
          <div><span>{summary.asked}</span>questions</div>
          <div><span>{summary.solved}</span>solved</div>
          <div><span>{summary.memory_size}</span>memories</div>
        </div>
      )}
      {summary?.practised > 0 && <p className="muted">Practised {summary.practised} task(s) on my own, passed {summary.practised_passed}.</p>}
      <NextTask suggestion={suggestion} onPractise={onPractise} onAnother={onAnother} busy={busy} />
      {summary?.awaiting_feedback > 0 && <p className="muted">{summary.awaiting_feedback} answer(s) waiting for your verdict.</p>}
      {top.length === 0 ? (
        <p className="muted">Nothing yet. Every solved or failed question becomes a memory the assistant can reuse.</p>
      ) : (
        <ul className="side-notes">
          {top.map((m, i) => {
            const title = (m.text.split('\n')[0] || '').replace(/^Task:\s*/, '').slice(0, 70)
            const ok = m.correct === true
            return (
              <li key={i}>
                <span className={`dot ${ok ? 'dot--ok' : 'dot--bad'}`} title={ok ? 'worked' : 'did not work'} />
                <span className="side-note-text">{title || '(note)'}</span>
                <span className="muted">{m.trust.toFixed(2)}</span>
              </li>
            )
          })}
        </ul>
      )}
      <p className="muted side-foot">
        Trust (0–1) rises when a note helps and drops when it misleads; low-trust notes are never used.
        {summary?.isolation && <> Code runs {summary.isolation === 'sandbox-exec' ? 'in a sandbox' : 'with resource limits only'}.</>}
      </p>
      <button className="btn btn--ghost" disabled={busy} onClick={() => { if (window.confirm('Clear the conversation and everything the assistant has learned?')) onReset() }}>
        Clear everything
      </button>
    </aside>
  )
}

export default function Assistant() {
  const [turns, setTurns] = useState([])
  const [summary, setSummary] = useState(null)
  const [notes, setNotes] = useState([])
  const [prompt, setPrompt] = useState('')
  const [tests, setTests] = useState('')
  const [showTests, setShowTests] = useState(false)
  const [busy, setBusy] = useState(false)
  const [pending, setPending] = useState(null)
  const [suggestion, setSuggestion] = useState(null)
  const [error, setError] = useState(null)
  const endRef = useRef(null)

  const refresh = async () => {
    try {
      const [h, s, n, g] = await Promise.all([getJSON('/api/chat/history'), getJSON('/api/chat/summary'),
        getJSON('/api/memory/chat'), getJSON('/api/chat/suggest')])
      setTurns(h); setSummary(s); setNotes(n); setSuggestion(g); setError(null)
    } catch (e) { setError(`Could not reach the backend: ${e.message}`) }
  }
  useEffect(() => { refresh() }, [])
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [turns.length, pending])

  const run = async (fn) => {
    setBusy(true); setError(null)
    try { await fn() } catch (e) { setError(e.message) } finally { setBusy(false) }
  }
  const ask = () => {
    const text = prompt.trim()
    if (!text || busy) return
    run(async () => {
      setPending({ prompt: text, tests: showTests ? tests.trim() : '' })
      try {
        await postJSON('/api/chat/ask', { prompt: text, tests: showTests ? tests : '' })
        setPrompt(''); setTests('')
      } finally { setPending(null) }
      await refresh()
    })
  }
  const feedback = (id, correct, comment) => run(async () => { await postJSON('/api/chat/feedback', { id, correct, comment }); await refresh() })
  const practise = (id) => run(async () => {
    setPending({ practice: true })
    try { await postJSON('/api/chat/practise', { task_id: id }) } finally { setPending(null) }
    await refresh()
  })
  const another = () => run(async () => setSuggestion(await getJSON('/api/chat/suggest?refresh=true')))
  const newChat = () => run(async () => { await postJSON('/api/chat/new'); await refresh() })
  const reset = () => run(async () => { await postJSON('/api/chat/reset'); await refresh() })
  const inConversation = !!summary && turns.some(t => t.conversation === summary.conversation && !t.practice)
  const onKey = (e) => { if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') ask() }

  return (
    <div className="chat-layout">
      <section className="chat">
        <div className="chat-head">
          <span className="muted">{inConversation ? 'I can see this conversation, so follow-ups work.' : 'Start with a question.'}</span>
          <button className="link-btn" disabled={busy || !inConversation} onClick={newChat}>+ New chat</button>
        </div>
        <div className="thread">
          {turns.length === 0 && !pending && (
            <div className="empty">
              <h2>What do you want to build?</h2>
              <p className="muted">
                Ask a Python coding question. If you add tests, I run them to check my answer. Either way I remember
                what worked and what didn't, so I get better as we go.
              </p>
              <div className="chips">
                {EXAMPLES.map(x => <button key={x} className="chip" onClick={() => setPrompt(x)}>{x}</button>)}
              </div>
            </div>
          )}
          {turns.map((t, i) => (
            <div key={t.id} className="turn-group">
              {i > 0 && t.conversation !== turns[i - 1].conversation && <div className="divider"><span>New conversation</span></div>}
              <Turn turn={t} onFeedback={feedback} busy={busy} />
            </div>
          ))}
          {pending && pending.practice && (
            <div className="msg"><div className="bubble bubble--bot thinking">Practising a task I chose: planning, recalling, writing code, running its tests…</div></div>
          )}
          {pending && !pending.practice && (
            <>
              <div className="msg msg--user"><div className="bubble bubble--user"><p>{pending.prompt}</p></div></div>
              <div className="msg"><div className="bubble bubble--bot thinking">Planning, recalling what I know, writing code{pending.tests ? ', running your tests' : ''}…</div></div>
            </>
          )}
          <div ref={endRef} />
        </div>

        {error && <div className="error-banner">{error}</div>}

        <div className="composer">
          <textarea
            value={prompt} onChange={e => setPrompt(e.target.value)} onKeyDown={onKey} rows={3} disabled={busy}
            placeholder={inConversation ? 'Ask a follow-up, e.g. “now make it handle empty input”' : 'Describe what you want, e.g. “Write a function that merges overlapping intervals”'}
          />
          {showTests && (
            <textarea className="composer-tests" value={tests} onChange={e => setTests(e.target.value)} onKeyDown={onKey}
              rows={3} disabled={busy} placeholder={TEST_PLACEHOLDER} />
          )}
          <div className="composer-bar">
            <button className="link-btn" onClick={() => setShowTests(s => !s)}>
              {showTests ? '− Remove tests' : '+ Add tests (lets me check my own answer)'}
            </button>
            <span className="muted">Python only · ⌘/Ctrl + Enter to send</span>
            <button className="btn btn--primary" disabled={busy || !prompt.trim()} onClick={ask}>{busy ? 'Working…' : 'Send'}</button>
          </div>
        </div>
      </section>
      <Sidebar summary={summary} notes={notes} onReset={reset} busy={busy} suggestion={suggestion} onPractise={practise} onAnother={another} />
    </div>
  )
}
