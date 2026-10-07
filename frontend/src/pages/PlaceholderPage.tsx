import { useState, type ReactNode } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Card } from '../components/ui/Card'
import { Button } from '../components/ui/Button'

function PageShell({ title, section, children }: { title: string; section: string; children: ReactNode }) {
  return (
    <div className="space-y-6">
      <header>
        <p className="text-xs text-[var(--color-ink-muted)]">{section}</p>
        <h1 className="text-2xl font-bold mt-1">{title}</h1>
      </header>
      {children}
    </div>
  )
}

export function SimulationPage() {
  const [result, setResult] = useState<unknown>(null)
  const [loading, setLoading] = useState(false)

  const run = async () => {
    setLoading(true)
    try {
      const res = await fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ arrival_rate: 5, clinicians: 3, replications: 5, horizon_hours: 4 }),
      })
      setResult(await res.json())
    } finally {
      setLoading(false)
    }
  }

  return (
    <PageShell title="Policy Simulation" section="Evaluation">
      <Card title="Quick run (5 replications)">
        <p className="text-sm text-[var(--color-ink-muted)] mb-3">Uses `/api/simulate`. Full charts per MASTER_PROMPT are next.</p>
        <Button onClick={run} disabled={loading}>
          Run simulation
        </Button>
        {result != null && (
          <pre className="mt-4 text-xs overflow-auto max-h-96 bg-[var(--color-surface-2)] p-3 rounded-lg">{JSON.stringify(result, null, 2)}</pre>
        )}
      </Card>
    </PageShell>
  )
}

export function EvaluationPage() {
  const calQuery = useQuery({
    queryKey: ['calibration'],
    queryFn: () => fetch('/api/calibration').then((r) => r.json()) as Promise<Record<string, unknown>>,
  })

  return (
    <PageShell title="Model Evaluation" section="Evaluation">
      <Card title="Calibration (held-out)">
        {calQuery.isLoading && <p className="text-sm">Loading…</p>}
        {calQuery.data && (
          <dl className="grid sm:grid-cols-3 gap-4 text-sm">
            {Object.entries(calQuery.data)
              .filter(([k, v]) => typeof v === 'number' && !k.includes('curve'))
              .slice(0, 9)
              .map(([k, v]) => (
                <div key={k}>
                  <dt className="text-xs uppercase text-[var(--color-ink-muted)]">{k}</dt>
                  <dd className="font-mono-nums text-xl font-semibold">{(v as number).toFixed(4)}</dd>
                </div>
              ))}
          </dl>
        )}
      </Card>
    </PageShell>
  )
}

export function AboutPage() {
  const rows = [
    { topic: 'Bayesian networks', page: 'Network & Learning', path: '/network' },
    { topic: 'Variable elimination', page: 'Inference Lab', path: '/inference' },
    { topic: 'Expected utility', page: 'Triage Desk', path: '/' },
    { topic: 'Value of information', page: 'Decision & VOI', path: '/decision' },
    { topic: 'Queue prioritization (cμ)', page: 'ER Queue Board', path: '/queue' },
  ]
  return (
    <PageShell title="About & Syllabus Map" section="Learn">
      <Card title="Syllabus topics → app locations">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-xs uppercase text-[var(--color-ink-muted)]">
              <th className="py-2">Topic</th>
              <th className="py-2">Page</th>
              <th className="py-2"></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.topic} className="border-t border-[var(--color-border)]">
                <td className="py-2">{r.topic}</td>
                <td className="py-2">{r.page}</td>
                <td className="py-2">
                  <a href={r.path} className="text-[var(--color-primary)] underline">
                    Open
                  </a>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
      <Card title="Limitations">
        <p className="text-sm text-[var(--color-ink-muted)]">
          Synthetic data only. Posteriors are probabilities, not diagnoses. Overrides are logged; recommendations are never auto-applied in production care.
        </p>
      </Card>
    </PageShell>
  )
}
