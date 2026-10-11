// Tiny Python syntax highlighter: builds React elements (never raw HTML), so code can't inject markup.
const KEYWORDS = new Set(('False None True and as assert async await break class continue def del elif else except ' +
  'finally for from global if import in is lambda nonlocal not or pass raise return try while with yield').split(' '))
const BUILTINS = new Set(('print len range int str float list dict set tuple sum min max sorted enumerate zip map filter ' +
  'any all abs isinstance open input reversed round type bool self').split(' '))
const TOKEN = /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?''')|([rRbBfF]{0,2}"(?:\\.|[^"\\\n])*"|[rRbBfF]{0,2}'(?:\\.|[^'\\\n])*')|(@\w+)|(\b\d[\d_]*(?:\.\d+)?\b)|(\b[A-Za-z_]\w*\b)/g

export function Highlight({ code }) {
  const out = []
  let last = 0, prevWord = '', m
  TOKEN.lastIndex = 0
  while ((m = TOKEN.exec(code)) !== null) {
    if (m.index > last) out.push(code.slice(last, m.index))
    const [text, comment, tri, str, deco, num, word] = m
    let cls = null
    if (comment) cls = 'tok-com'
    else if (tri || str) cls = 'tok-str'
    else if (deco) cls = 'tok-dec'
    else if (num) cls = 'tok-num'
    else if (word) {
      if (KEYWORDS.has(word)) cls = 'tok-kw'
      else if (prevWord === 'def') cls = 'tok-fn'
      else if (prevWord === 'class') cls = 'tok-cls'
      else if (BUILTINS.has(word)) cls = 'tok-bi'
      prevWord = word
    }
    out.push(cls ? <span key={m.index} className={cls}>{text}</span> : text)
    last = m.index + text.length
  }
  if (last < code.length) out.push(code.slice(last))
  return <>{out}</>
}
