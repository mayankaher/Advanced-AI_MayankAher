import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'
import { assessmentFromDemoEvidence, assessmentToEvidence } from '../lib/evidence-form'
import { Card } from '../components/ui/Card'
import { Button } from '../components/ui/Button'
import { ProbBar } from '../components/ui/ProbBar'

const METHODS = ['ve', 'bp', 'rejection', 'lw', 'gibbs'] as const

export default function InferenceLabPage() {
  const casesQuery = useQuery({ queryKey: ['cases'], queryFn: api.getCases })
  const [caseId, setCaseId] = useState('E')
  const [method, setMethod] = useState<(typeof METHODS)[number]>('ve')
  const [result, setResult] = useState<Awaited<ReturnType<typeof api.infer>> | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const run = async () => {
    const c = casesQuery.data?.find((x) => x.id === caseId)
    if (!c) return
    setLoading(true)
    setError(null)
    try {
      const evidence = assessmentToEvidence(assessmentFromDemoEvidence(caseId, c.evidence))
      const res = await api.infer({ evidence, method, n_samples: method === 've' || method === 'bp' ? undefined : 10000 })
      setResult(res)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Request failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <header>
        <p className="text-xs text-[var(--color-ink-muted)]">AI Engine · Inference Lab</p>
        <h1 className="text-2xl font-bold mt-1">Inference methods</h1>
        <p className="text-sm text-[var(--color-ink-muted)]">Compare exact variable elimination with approximate methods (same API as backend).</p>
      </header>

      <Card title="Configuration">
        <div className="flex flex-wrap gap-3 items-end">
          <label className="text-sm">
            Demo case
            <select value={caseId} onChange={(e) => setCaseId(e.target.value)} className="block mt-1 rounded-[var(--radius-input)] border px-3 py-2">
              {(casesQuery.data ?? []).map((c) => (
                <option key={c.id} value={c.id}>
                  {c.id}
                </option>
              ))}
            </select>
          </label>
          <div className="flex flex-wrap gap-1">
            {METHODS.map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setMethod(m)}
                className={`px-3 py-1.5 rounded-full text-xs border ${method === m ? 'bg-[var(--color-primary)] text-white border-transparent' : 'border-[var(--color-border)]'}`}
              >
                {m.toUpperCase()}
              </button>
            ))}
          </div>
          <Button onClick={run} disabled={loading}>
            Run inference
          </Button>
        </div>
        {error && <p className="text-sm text-[var(--color-p1)] mt-3">{error}</p>}
      </Card>

      {result && (
        <div className="grid md:grid-cols-2 gap-4">
          <Card title={`Posterior (${result.method})`}>
            {Object.entries(result.posteriors.Severity ?? {}).map(([k, v]) => (
              <ProbBar key={k} label={k} value={v} emphasize={k === 'Critical'} />
            ))}
            {result.acceptance_rate != null && (
              <p className="text-xs mt-3 font-mono-nums">Rejection acceptance rate: {(result.acceptance_rate * 100).toFixed(2)}%</p>
            )}
          </Card>
          <Card title="VE trace (Severity)" subtitle="Exact only">
            {method === 've' && result.traces?.Severity ? (
              <ol className="text-xs font-mono-nums space-y-2 max-h-64 overflow-y-auto">
                {(result.traces.Severity as Array<{ description?: string }>).map((step, i) => (
                  <li key={i}>{step.description ?? JSON.stringify(step)}</li>
                ))}
              </ol>
            ) : (
              <p className="text-sm text-[var(--color-ink-muted)]">Run with method VE to see elimination trace.</p>
            )}
          </Card>
        </div>
      )}
    </div>
  )
}
