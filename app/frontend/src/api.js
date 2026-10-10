export const API = 'http://127.0.0.1:8000'

async function readError(r) {
  const text = await r.text()
  try { return JSON.parse(text).detail ?? text } catch { return text }
}

export const getJSON = async (path) => {
  const r = await fetch(`${API}${path}`)
  if (!r.ok) throw new Error(await readError(r))
  return r.json()
}

export const postJSON = async (path, body) => {
  const r = await fetch(`${API}${path}`, {
    method: 'POST',
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!r.ok) throw new Error(await readError(r))
  return r.json()
}
