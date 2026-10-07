import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'
import { assessmentFromDemoEvidence, assessmentToEvidence } from '../lib/evidence-form'
import { Card } from '../components/ui/Card'
import { Button } from '../components/ui/Button'
import { ProbBar } from '../components/ui/ProbBar'
import { TriageLevelPill } from '../components/ui/TriageLevelPill'
import { TRIAGE_LEVELS } from '../lib/triage-levels'

export default function DecisionPage() {
  const modelQuery = useQuery({ queryKey: ['model'], queryFn: api.getModel })
  const casesQuery = useQuery({ queryKey: ['cases'], queryFn: api.getCases })
  const [caseId, setCaseId] = useState('F_incomplete')

  const decideQuery = useQuery({
    queryKey: ['decide', caseId],
    queryFn: async () => {
      const c = casesQuery.data!.find((x) => x.id === caseId)!
      const evidence = assessmentToEvidence(assessmentFromDemoEvidence(caseId, c.evidence))
      return api.decide(evidence)
    },
    enabled: !!casesQuery.data?.length,
  })

  const voiQuery = useQuery({
    queryKey: ['voi', caseId],
    queryFn: async () => {
      const c = casesQuery.data!.find((x) => x.id === caseId)!
      const evidence = assessmentToEvidence(assessmentFromDemoEvidence(caseId, c.evidence))
      return api.voi(evidence)
    },
    enabled: !!casesQuery.data?.length,
  })

  const matrix = modelQuery.data?.decision?.utility_matrix_rows_levels_cols_severity

  return (
    <div className="space-y-6">
      <header>
        <p className="text-xs text-[var(--color-ink-muted)]">AI Engine · Decision & VOI</p>
        <h1 className="text-2xl font-bold mt-1">Utility matrix & VOI explorer</h1>
      </header>

      <Card title="Utility matrix U[level][severity]" subtitle="From model spec (read-only)">
        {matrix ? (
          <div className="overflow-x-auto">
            <table className="text-sm font-mono-nums">
              <thead>
                <tr>
                  <th className="p-2 text-left">Level</th>
                  <th className="p-2">Low</th>
                  <th className="p-2">Moderate</th>
                  <th className="p-2">Critical</th>
                </tr>
              </thead>
              <tbody>
                {TRIAGE_LEVELS.map((level, i) => (
                  <tr key={level} className="border-t border-[var(--color-border)]">
                    <td className="p-2">
                      <TriageLevelPill level={level} size="sm" />
                    </td>
                    {matrix[i]?.map((u, j) => (
                      <td key={j} className="p-2 text-center">
                        {u}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-sm text-[var(--color-ink-muted)]">Loading…</p>
        )}
        <p className="text-xs text-[var(--color-ink-muted)] mt-3">Inline editing will POST to `/api/model/utility` in a later pass.</p>
      </Card>

      <Card title="VOI explorer">
        <label className="text-sm block mb-4">
          Patient
          <select value={caseId} onChange={(e) => setCaseId(e.target.value)} className="block mt-1 rounded-[var(--radius-input)] border px-3 py-2">
            {(casesQuery.data ?? []).map((c) => (
              <option key={c.id} value={c.id}>
                Case {c.id}
              </option>
            ))}
          </select>
        </label>
        {decideQuery.data && (
          <div className="mb-4">
            <TriageLevelPill level={decideQuery.data.recommended_level} />
            {Object.entries(decideQuery.data.p_severity).map(([k, v]) => (
              <ProbBar key={k} label={k} value={v} />
            ))}
          </div>
        )}
        {voiQuery.data && (
          <ul className="space-y-2 text-sm">
            {voiQuery.data.test_analyses.map((t) => (
              <li key={t.test} className="flex justify-between font-mono-nums border-b border-[var(--color-border)] py-1">
                <span>{t.test}</span>
                <span>
                  VOI {t.gross_voi.toFixed(2)} · net {t.net_voi.toFixed(2)}
                </span>
              </li>
            ))}
          </ul>
        )}
        <Button variant="secondary" className="mt-4" onClick={() => { decideQuery.refetch(); voiQuery.refetch() }}>
          Refresh
        </Button>
      </Card>
    </div>
  )
}
