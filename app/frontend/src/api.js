export const API = 'http://127.0.0.1:8000'

export const getJSON = (path) => fetch(`${API}${path}`).then(r => r.json())
export const postJSON = (path, body) => fetch(`${API}${path}`, {
  method: 'POST',
  headers: body ? { 'Content-Type': 'application/json' } : undefined,
  body: body ? JSON.stringify(body) : undefined,
}).then(async r => {
  if (!r.ok) throw new Error(await r.text())
  return r.json()
})
