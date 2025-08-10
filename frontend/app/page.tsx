'use client'
import { useEffect, useState } from 'react'

export default function Home() {
  const [health, setHealth] = useState('...')
  useEffect(() => {
    const base = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000'
    fetch(`${base}/healthz`).then(r => r.json()).then(j => setHealth(j.status ?? 'unknown')).catch(() => setHealth('unreachable'))
  }, [])
  return (
    <main className="max-w-2xl mx-auto p-6">
      <h1 className="text-3xl font-bold">H-Pulse</h1>
      <p className="mt-2">Backend health: <span className="font-mono">{health}</span></p>
      <a href="/login" className="inline-block mt-6 px-4 py-2 rounded bg-blue-600 text-white">Login</a>
    </main>
  )
}