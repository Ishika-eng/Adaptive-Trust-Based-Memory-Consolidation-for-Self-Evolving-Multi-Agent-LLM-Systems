// Small, safe Markdown subset for the assistant's explanations: paragraphs, lists, **bold**, *italic*,
// `code`, and $math$ (shown as code, since there is no LaTeX renderer). Builds React elements only.
const INLINE = /(`[^`\n]+`|\*\*[^*\n]+\*\*|\$[^$\n]+\$|\*[^*\n]+\*)/g

function inline(text) {
  return text.split(INLINE).map((part, i) => {
    if (!part) return null
    if (part.startsWith('`')) return <code key={i}>{part.slice(1, -1)}</code>
    if (part.startsWith('**')) return <strong key={i}>{part.slice(2, -2)}</strong>
    if (part.startsWith('$')) return <code key={i}>{part.slice(1, -1)}</code>
    if (part.startsWith('*')) return <em key={i}>{part.slice(1, -1)}</em>
    return part
  })
}

export function Markdown({ text }) {
  const blocks = (text || '').trim().split(/\n{2,}/)
  return (
    <div className="md">
      {blocks.map((block, i) => {
        const lines = block.split('\n')
        if (lines.every(l => /^\s*[-*]\s+/.test(l))) {
          return <ul key={i}>{lines.map((l, j) => <li key={j}>{inline(l.replace(/^\s*[-*]\s+/, ''))}</li>)}</ul>
        }
        if (lines.every(l => /^\s*\d+[.)]\s+/.test(l))) {
          return <ol key={i}>{lines.map((l, j) => <li key={j}>{inline(l.replace(/^\s*\d+[.)]\s+/, ''))}</li>)}</ol>
        }
        return <p key={i}>{inline(lines.join(' '))}</p>
      })}
    </div>
  )
}
