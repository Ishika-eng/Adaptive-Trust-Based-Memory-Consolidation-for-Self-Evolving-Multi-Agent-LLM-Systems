import { useEffect, useRef, useState } from 'react'
import { getJSON, postJSON } from '../api'
import { Highlight } from '../lib/highlight'
import Logo from '../lib/Logo'
import { Markdown } from '../lib/markdown'

const EXAMPLES = [
  { title: 'Merge intervals', sub: 'Combine overlapping ranges into the fewest intervals', prompt: 'Write a function merge(intervals) that merges overlapping intervals and returns them sorted' },
  { title: 'Balanced brackets', sub: 'Check that ()[]{} in a string are properly nested', prompt: 'Write a function is_balanced(s) that checks whether the brackets in a string are balanced' },
  { title: 'Fast Fibonacci', sub: 'Return the n-th Fibonacci number without recursion blowup', prompt: 'Write a function fib(n) that returns the n-th Fibonacci number efficiently' },
]
const TEST_PLACEHOLDER = 'assert merge([[1, 3], [2, 6], [8, 10]]) == [[1, 6], [8, 10]]'
const WORKING = ['Planning the approach…', 'Recalling what worked before…', 'Writing the code…', 'Running checks…']

const STATUS = {
  passed: { label: 'Tests passed', cls: 'pill--good' },
  failed: { label: 'Tests failed', cls: 'pill--bad' },
  unverified: { label: 'Not verified', cls: 'pill--impl' },
}

const Icon = ({ d, size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{d}</svg>
)
const ICONS = {
  send: <path d="M5 12h14M13 6l6 6-6 6" />,
  shield: <path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z" />,
  brain: <path d="M9 4a3 3 0 0 0-3 3v1a3 3 0 0 0-2 3 3 3 0 0 0 2 3v1a3 3 0 0 0 3 3h1V4zM15 4a3 3 0 0 1 3 3v1a3 3 0 0 1 2 3 3 3 0 0 1-2 3v1a3 3 0 0 1-3 3h-1V4z" />,
  play: <path d="M6 4l14 8-14 8z" />,
  panel: <path d="M4 5h16v14H4zM15 5v14" />,
}

function Working({ text }) {
  const [i, setI] = useState(0)
  useEffect(() => { const id = setInterval(() => setI(n => (n + 1) % WORKING.length), 2200); return () => clearInterval(id) }, [])
  return (
    <div className="msg">
      <div className="avatar"><Logo size={30} /></div>
      <div className="bot thinking" role="status"><span className="dots"><i /><i /><i /></span>{text || WORKING[i]}</div>
    </div>
  )
}

function CodeBlock({ code }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    try { await navigator.clipboard.writeText(code); setCopied(true); setTimeout(() => setCopied(false), 1500) } catch { /* clipboard unavailable */ }
  }
  return (
    <div className="codeblock">
      <div className="codeblock-bar"><span>python</span><button onClick={copy} aria-label="Copy code">{copied ? 'Copied ✓' : 'Copy'}</button></div>
      <pre><code>{code ? <Highlight code={code} /> : '(no code returned)'}</code></pre>
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
            {m.used && <span className="badge badge--used">relied on this</span>}
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
    return <p className="run-note">{turn.feedback.correct ? '✓ You confirmed this worked — saved to memory.' : '✗ You said it didn’t work — I saved what went wrong.'}</p>
  }
  if (turn.status !== 'unverified') return null
  return (
    <div className="feedback">
      <span>Did this work for you? I only learn from it once you tell me.</span>
      <button className="btn btn--good" disabled={busy} onClick={() => onSend(turn.id, true, '')}>It worked</button>
      <button className="btn btn--ghost" disabled={busy} onClick={() => setOpen(o => !o)}>It didn’t</button>
      {open && (
        <div className="feedback-form">
          <input type="text" value={comment} onChange={e => setComment(e.target.value)} placeholder="What went wrong? (optional)"
            aria-label="What went wrong" onKeyDown={e => e.key === 'Enter' && onSend(turn.id, false, comment)} />
          <button className="btn btn--primary" disabled={busy} onClick={() => onSend(turn.id, false, comment)}>Send</button>
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
            <strong>Practice task I chose myself</strong>
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
        <div className="avatar"><Logo size={30} /></div>
        <div className="bot">
          <div className="bubble-head">
            <span className={`pill ${st.cls}`}>{st.label}</span>
            {turn.status === 'failed' && <span className="muted">{turn.stage.replace('_', ' ')}: {turn.detail}</span>}
          </div>
          {turn.explanation && <div className="explain"><Markdown text={turn.explanation} /></div>}
          <CodeBlock code={turn.code} />
          {turn.status === 'unverified' && !turn.feedback && (
            <p className="run-note">
              {turn.output ? <>It ran without errors and printed <code>{turn.output}</code>. </> : 'It ran without errors. '}
              Nothing checks that it’s correct — add tests next time to verify it.
            </p>
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
  if (!suggestion.id) return <p className="muted">I’ve worked through every practice task I know.</p>
  return (
    <div className="next-task">
      <h4>My next practice task</h4>
      <p className="muted">{suggestion.reason}</p>
      <details>
        <summary>See the task</summary>
        <pre className="how-pre">{suggestion.question}</pre>
      </details>
      <div className="next-task-actions">
        <button className="btn btn--primary" disabled={busy} onClick={() => onPractise(suggestion.id)}>Practise it</button>
        <button className="btn btn--ghost" disabled={busy} onClick={onAnother}>Another</button>
      </div>
      <p className="muted side-foot">It runs against the task’s real tests, so what I learn is verified.</p>
    </div>
  )
}

function Sidebar({ summary, notes, onReset, busy, suggestion, onPractise, onAnother, open, onClose }) {
  const top = [...notes].sort((a, b) => b.trust - a.trust).slice(0, 8)
  return (
    <aside className={`side ${open ? 'side--open' : ''}`} aria-label="What the assistant has learned">
      <div className="chat-head-actions" style={{ justifyContent: 'space-between' }}>
        <h3>What I’ve learned</h3>
        {open && <button className="link-btn" onClick={onClose}>Close</button>}
      </div>
      {summary && (
        <div className="side-stats">
          <div><span>{summary.asked}</span>questions</div>
          <div><span>{summary.solved}</span>solved</div>
          <div><span>{summary.memory_size}</span>memories</div>
        </div>
      )}
      {summary?.practised > 0 && <p className="muted side-foot">Practised {summary.practised} task(s) on my own; passed {summary.practised_passed}.</p>}
      <NextTask suggestion={suggestion} onPractise={onPractise} onAnother={onAnother} busy={busy} />
      {summary?.awaiting_feedback > 0 && <p className="side-foot muted">{summary.awaiting_feedback} answer(s) waiting for your verdict.</p>}
      <div>
        <div className="side-section-title" style={{ marginBottom: 8 }}>Most trusted notes</div>
        {top.length === 0 ? (
          <p className="muted side-foot">Nothing yet. Every solved or failed question becomes a note I can reuse.</p>
        ) : (
          <ul className="side-notes">
            {top.map((m, i) => {
              const title = (m.text.split('\n')[0] || '').replace(/^Task:\s*/, '').slice(0, 70)
              const ok = m.correct === true
              return (
                <li key={i}>
                  <span className={`dot ${ok ? 'dot--ok' : 'dot--bad'}`} title={ok ? 'worked' : 'did not work'} />
                  <span className="side-note-text" title={title}>{title || '(note)'}</span>
                  <span className="muted num">{m.trust.toFixed(2)}</span>
                </li>
              )
            })}
          </ul>
        )}
      </div>
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
  const [suggestion, setSuggestion] = useState(null)
  const [prompt, setPrompt] = useState('')
  const [tests, setTests] = useState('')
  const [showTests, setShowTests] = useState(false)
  const [busy, setBusy] = useState(false)
  const [pending, setPending] = useState(null)
  const [error, setError] = useState(null)
  const [panel, setPanel] = useState(false)
  const endRef = useRef(null)
  const boxRef = useRef(null)

  const refresh = async () => {
    try {
      const [h, s, n, g] = await Promise.all([getJSON('/api/chat/history'), getJSON('/api/chat/summary'),
        getJSON('/api/memory/chat'), getJSON('/api/chat/suggest')])
      setTurns(h); setSummary(s); setNotes(n); setSuggestion(g); setError(null)
    } catch (e) { setError(`Could not reach the backend: ${e.message}`) }
  }
  useEffect(() => { refresh() }, [])
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }) }, [turns.length, pending])
  useEffect(() => {                               // auto-grow the composer
    const el = boxRef.current
    if (el) { el.style.height = 'auto'; el.style.height = `${Math.min(el.scrollHeight, 200)}px` }
  }, [prompt])

  const run = async (fn) => {
    setBusy(true); setError(null)
    try { await fn() } catch (e) { setError(e.message) } finally { setBusy(false) }
  }
  const ask = () => {
    const text = prompt.trim()
    if (!text || busy) return
    const sentTests = showTests ? tests : ''
    run(async () => {
      setPending({ prompt: text, tests: sentTests.trim() })
      setPrompt(''); setTests('')                 // clear at once; restored below if the request fails
      try {
        await postJSON('/api/chat/ask', { prompt: text, tests: sentTests })
      } catch (e) {
        setPrompt(text); setTests(sentTests)
        throw e
      } finally { setPending(null) }
      await refresh()
    })
  }
  const feedback = (id, correct, comment) => run(async () => { await postJSON('/api/chat/feedback', { id, correct, comment }); await refresh() })
  const practise = (id) => run(async () => {
    setPanel(false); setPending({ practice: true })
    try { await postJSON('/api/chat/practise', { task_id: id }) } finally { setPending(null) }
    await refresh()
  })
  const another = () => run(async () => setSuggestion(await getJSON('/api/chat/suggest?refresh=true')))
  const newChat = () => run(async () => { await postJSON('/api/chat/new'); await refresh() })
  const reset = () => run(async () => { setPanel(false); await postJSON('/api/chat/reset'); await refresh() })

  const inConversation = !!summary && turns.some(t => t.conversation === summary.conversation && !t.practice)
  const onKey = (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); ask() }
    else if (e.key === 'Enter' && !e.shiftKey && e.target === boxRef.current) { e.preventDefault(); ask() }
  }

  return (
    <div className="chat-layout">
      <section className="chat" aria-label="Chat">
        <div className="chat-head">
          <div className="chat-head-title">
            {inConversation ? 'Conversation' : 'New conversation'}
            <span className="muted" style={{ fontWeight: 400 }}>{inConversation ? '· I can see earlier messages' : ''}</span>
          </div>
          <div className="chat-head-actions">
            <button className="link-btn panel-toggle" onClick={() => setPanel(true)}><Icon d={ICONS.panel} /> Memory</button>
            <button className="link-btn" disabled={busy || !inConversation} onClick={newChat}>+ New chat</button>
          </div>
        </div>

        <div className="thread" aria-live="polite">
          {turns.length === 0 && !pending && (
            <div className="empty">
              <span className="empty-mark"><Logo size={44} /></span>
              <h2>What are we building today?</h2>
              <p>Ask a Python question. Add tests and I’ll run them to check my own answer; either way I remember what worked and what didn’t.</p>
              <div className="features">
                <span className="feature"><Icon d={ICONS.shield} size={14} /> Runs code in a sandbox</span>
                <span className="feature"><Icon d={ICONS.brain} size={14} /> Learns from outcomes</span>
                <span className="feature"><Icon d={ICONS.play} size={14} /> Practises its own weak spots</span>
              </div>
              <div className="chips">
                {EXAMPLES.map(x => (
                  <button key={x.title} className="chip" onClick={() => { setPrompt(x.prompt); boxRef.current?.focus() }}>
                    <strong>{x.title}</strong><span>{x.sub}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
          {turns.map((t, i) => (
            <div key={t.id} className="turn-group">
              {i > 0 && t.conversation !== turns[i - 1].conversation && <div className="divider"><span>New conversation</span></div>}
              <Turn turn={t} onFeedback={feedback} busy={busy} />
            </div>
          ))}
          {pending && !pending.practice && (
            <>
              <div className="msg msg--user"><div className="bubble bubble--user"><p>{pending.prompt}</p></div></div>
              <Working />
            </>
          )}
          {pending && pending.practice && <Working text="Practising a task I chose: planning, recalling, writing code, running its tests…" />}
          <div ref={endRef} />
        </div>

        {error && <div className="error-banner" role="alert">{error}</div>}

        <div className="composer">
          <div className="composer-box">
            <textarea ref={boxRef} value={prompt} onChange={e => setPrompt(e.target.value)} onKeyDown={onKey} rows={2} disabled={busy}
              aria-label="Your coding question"
              placeholder={inConversation ? 'Ask a follow-up, e.g. “now make it handle empty input”' : 'Describe what you want, e.g. “Write a function that merges overlapping intervals”'} />
            {showTests && (
              <textarea className="composer-tests" value={tests} onChange={e => setTests(e.target.value)} onKeyDown={onKey}
                rows={3} disabled={busy} aria-label="Tests (assert statements)" placeholder={TEST_PLACEHOLDER} />
            )}
            <div className="composer-bar">
              <button className="link-btn" onClick={() => setShowTests(s => !s)}>
                {showTests ? '− Remove tests' : '+ Add tests to verify the answer'}
              </button>
              <button className="btn btn--primary send-btn" disabled={busy || !prompt.trim()} onClick={ask} aria-label="Send">
                <Icon d={ICONS.send} size={18} />
              </button>
            </div>
          </div>
          <p className="composer-hint">Python only · Enter to send, Shift+Enter for a new line</p>
        </div>
      </section>
      <Sidebar summary={summary} notes={notes} onReset={reset} busy={busy} suggestion={suggestion}
        onPractise={practise} onAnother={another} open={panel} onClose={() => setPanel(false)} />
    </div>
  )
}
