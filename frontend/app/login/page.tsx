'use client'
import { useState } from 'react'

export default function LoginPage() {
  const [email, setEmail] = useState('admin@hpulse.local')
  const [password, setPassword] = useState('Admin123!#')
  const [msg, setMsg] = useState('')

  const onLogin = async () => {
    const base = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000'
    const res = await fetch(`${base}/api/v1/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    })
    if (!res.ok) { setMsg('Login failed'); return }
    const data = await res.json()
    localStorage.setItem('token', data.access_token)
    setMsg('Logged in')
  }

  return (
    <main className="max-w-md mx-auto p-6">
      <h1 className="text-2xl font-semibold">Login</h1>
      <div className="mt-4 space-y-3">
        <input className="w-full border p-2" value={email} onChange={e => setEmail(e.target.value)} placeholder="Email" />
        <input className="w-full border p-2" type="password" value={password} onChange={e => setPassword(e.target.value)} placeholder="Password" />
        <button className="px-4 py-2 bg-blue-600 text-white rounded" onClick={onLogin}>Login</button>
        <p>{msg}</p>
      </div>
    </main>
  )
}