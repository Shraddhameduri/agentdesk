import { useEffect, useRef, useState } from 'react'

interface Approval {
  id: string
  action: string
  status: string
  payload: Record<string, any>
}

interface Source {
  doc_id: string
  title: string
  chunk_id: string
  score: number
}

interface Message {
  role: 'user' | 'assistant'
  text: string
  approval?: Approval | null
  sources?: Source[]
  provider?: string
}

function providerBadge(providerInfo: { provider?: string; model?: string }) {
  const { provider, model } = providerInfo
  if (provider === 'ollama') return <span className="demo-badge">LIVE — Ollama {model || ''} (local model)</span>
  if (provider === 'fake') return <span className="demo-badge">DEMO MODE — deterministic fake LLM, no API key needed</span>
  if (provider) return <span className="demo-badge">via {provider} provider</span>
  return null
}

export default function App() {
  const [apiKey, setApiKey] = useState(localStorage.getItem('copilot-key') || 'demo-key')
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [approvals, setApprovals] = useState<Approval[]>([])
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])
  useEffect(() => { refreshApprovals() }, [])
  useEffect(() => {
    // Badge reflects the actual live provider reported by /ready — never hardcoded.
    fetch('/ready').then(r => r.json()).then(setProviderInfo).catch(() => {})
  }, [])
  const [providerInfo, setProviderInfo] = useState<{ provider?: string; model?: string }>({})

  const headers = () => ({ 'X-API-Key': apiKey, 'Content-Type': 'application/json' })

  async function refreshApprovals() {
    try {
      const r = await fetch('/api/approvals?status=pending', { headers: { 'X-API-Key': apiKey } })
      if (r.ok) setApprovals(await r.json())
    } catch { /* offline */ }
  }

  async function resolveApproval(id: string, approve: boolean) {
    const r = await fetch(`/api/approvals/${id}/${approve ? 'approve' : 'reject'}`, {
      method: 'POST', headers: headers(), body: JSON.stringify({}),
    })
    if (r.ok) {
      const body = await r.json()
      setApprovals(a => a.filter(x => x.id !== id))
      setMessages(m => [...m, {
        role: 'assistant',
        text: approve
          ? `✅ Approved ${id}. Ticket ${body.executed?.id ?? ''} created.`
          : `🚫 Rejected ${id}. Nothing was created.`,
      }])
    }
  }

  async function send() {
    const text = input.trim()
    if (!text || busy) return
    localStorage.setItem('copilot-key', apiKey)
    setInput('')
    setMessages(m => [...m, { role: 'user', text }])
    setBusy(true)
    try {
      const r = await fetch('/api/chat/stream', {
        method: 'POST', headers: headers(), body: JSON.stringify({ message: text }),
      })
      if (!r.ok || !r.body) throw new Error(`HTTP ${r.status}`)
      const reader = r.body.getReader()
      const decoder = new TextDecoder()
      let buf = '', reply = '', meta: any = {}
      const pushAssistant = (partial: string) =>
        setMessages(m => {
          const last = m[m.length - 1]
          const msg: Message = { role: 'assistant', text: partial, approval: meta.approval, sources: meta.sources, provider: meta.provider }
          return last?.role === 'assistant' && (last as any)._stream ? [...m.slice(0, -1), { ...msg, _stream: true } as any] : [...m, { ...msg, _stream: true } as any]
        })
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        const parts = buf.split('\n\n')
        buf = parts.pop() || ''
        for (const part of parts) {
          const ev = part.match(/^event: (\w+)\ndata: (.*)$/s)
          if (!ev) continue
          const [, kind, data] = ev
          const payload = JSON.parse(data)
          if (kind === 'meta') meta = payload
          else if (kind === 'token') { reply += payload.text; pushAssistant(reply) }
        }
      }
      if (meta.approval) refreshApprovals()
    } catch (e: any) {
      setMessages(m => [...m, { role: 'assistant', text: `⚠️ Error: ${e.message}` }])
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="app">
      <header>
        <h1>🤖 Agentic Support Copilot</h1>
        {providerBadge(providerInfo)}
        <div className="keyrow">
          <label>API key <input value={apiKey} onChange={e => setApiKey(e.target.value)} /></label>
        </div>
      </header>

      {approvals.length > 0 && (
        <div className="approvals">
          <h3>⏳ Pending approvals ({approvals.length})</h3>
          {approvals.map(a => (
            <div key={a.id} className="approval-card">
              <div><b>{a.id}</b> — {a.action} — <i>{a.payload.subject}</i></div>
              <div className="muted">{a.payload.description}</div>
              <div className="btnrow">
                <button className="approve" onClick={() => resolveApproval(a.id, true)}>Approve</button>
                <button className="reject" onClick={() => resolveApproval(a.id, false)}>Reject</button>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="chat">
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <div className="bubble">
              <div dangerouslySetInnerHTML={{ __html: escapeHtml(m.text).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>') }} />
              {m.sources && m.sources.length > 0 && (
                <div className="sources">📚 {m.sources.map(s => s.doc_id).join(', ')}</div>
              )}
              {m.provider && <div className="provider">via {m.provider} provider</div>}
            </div>
            {m.approval && (
              <div className="approval-card inline">
                <div>📝 Approval <b>{m.approval.id}</b> pending — review in the panel above.</div>
              </div>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="composer">
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && send()}
          placeholder='Try: "Where is my order ORD-1002?" or "What is your returns policy?"'
          disabled={busy}
        />
        <button onClick={send} disabled={busy}>{busy ? '…' : 'Send'}</button>
      </div>
      <footer className="hint">Write actions (ticket create) always require approval. Nothing executes without it.</footer>
    </div>
  )
}

function escapeHtml(s: string) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}
