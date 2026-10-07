import { useEffect, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../lib/api'
import { useQueueMeta } from '../context/QueueContext'
import { Card } from '../components/ui/Card'
import { Button } from '../components/ui/Button'
import { KpiCard } from '../components/ui/KpiCard'
import { TriageLevelPill } from '../components/ui/TriageLevelPill'
import { ConceptBadge } from '../components/ui/ConceptBadge'
import { rankExplanation } from '../lib/narrate'
import { ProbBar } from '../components/ui/ProbBar'

export default function QueueBoardPage() {
  const casesQuery = useQuery({ queryKey: ['cases'], queryFn: api.getCases })
  const { clinicians, setClinicians, setWaitingCount } = useQueueMeta()
  const [patientIds, setPatientIds] = useState<string[]>([])
  const [compareMode, setCompareMode] = useState<'bn' | 'pcrit' | 'fifo'>('bn')

  useEffect(() => {
    if (casesQuery.data && patientIds.length === 0) {
      setPatientIds(casesQuery.data.map((c) => c.id))
    }
  }, [casesQuery.data, patientIds.length])

  const patients = useMemo(() => {
    return (casesQuery.data ?? [])
      .filter((c) => patientIds.includes(c.id))
      .map((c) => ({ patient_id: c.id, evidence: c.evidence, wait_minutes: c.id.charCodeAt(0) % 25 }))
  }, [casesQuery.data, patientIds])

  useEffect(() => {
    setWaitingCount(patients.length)
  }, [patients.length, setWaitingCount])

  const rankQuery = useQuery({
    queryKey: ['queue-rank', patients, clinicians],
    queryFn: () => api.queueRank(patients, clinicians),
    enabled: patients.length > 0,
  })

  const ranked = rankQuery.data?.ranked ?? []

  const fifoOrder = useMemo(
    () => [...patients].sort((a, b) => a.wait_minutes - b.wait_minutes).map((p) => p.patient_id),
    [patients],
  )
  const pcritOrder = useMemo(
    () => [...ranked].sort((a, b) => b.p_critical - a.p_critical).map((r) => r.patient_id),
    [ranked],
  )

  const displayOrder =
    compareMode === 'fifo' ? fifoOrder : compareMode === 'pcrit' ? pcritOrder : ranked.map((r) => r.patient_id)

  const critLoad = ranked.filter((r) => r.p_critical > 0.5).length

  return (
    <div className="space-y-6">
      <header>
        <p className="text-xs text-[var(--color-ink-muted)]">Clinical · ER Queue Board</p>
        <h1 className="text-2xl font-bold mt-1">Live priority queue</h1>
        <p className="text-sm text-[var(--color-ink-muted)] mt-1">
          Rank waiting demo patients with the decision-theoretic cμ rule (API). Compare with P(Critical)-only and FIFO.
        </p>
      </header>

      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <KpiCard label="Patients waiting" value={patients.length} sub={['Demo cases loaded', `${clinicians} clinicians`]} />
        <KpiCard label="High P(Critical)" value={critLoad} unit="patients" sub={['P > 0.5', 'from posteriors']} />
        <KpiCard label="Avg priority index" value={ranked.length ? (ranked.reduce((s, r) => s + r.priority_index, 0) / ranked.length).toFixed(4) : '—'} />
        <KpiCard label="Ranking mode" value={compareMode === 'bn' ? 'cμ' : compareMode} />
      </div>

      <div className="flex flex-wrap gap-2 items-center">
        <label className="text-sm">
          Clinicians
          <input
            type="number"
            min={1}
            max={10}
            value={clinicians}
            onChange={(e) => setClinicians(Number(e.target.value))}
            className="ml-2 w-16 rounded-[var(--radius-input)] border border-[var(--color-border)] px-2 py-1"
          />
        </label>
        <Button variant="secondary" onClick={() => setPatientIds(casesQuery.data?.map((c) => c.id) ?? [])}>
          Import all demo cases
        </Button>
        <div className="flex rounded-[var(--radius-input)] border border-[var(--color-border)] overflow-hidden text-xs">
          {(
            [
              ['bn', 'BN + cμ'],
              ['pcrit', 'P(Crit) only'],
              ['fifo', 'FIFO'],
            ] as const
          ).map(([mode, label]) => (
            <button
              key={mode}
              type="button"
              className={`px-3 py-2 ${compareMode === mode ? 'bg-[var(--color-primary)] text-white' : 'bg-[var(--color-surface)]'}`}
              onClick={() => setCompareMode(mode)}
            >
              {label}
            </button>
          ))}
        </div>
        <ConceptBadge id="cmu" />
      </div>

      <Card title="Priority queue" subtitle="Watchlist">
        {rankQuery.isLoading && <p className="text-sm text-[var(--color-ink-muted)]">Ranking…</p>}
        {patients.length === 0 && (
          <p className="text-sm text-[var(--color-ink-muted)]">Add patients or import demo cases to watch the priority queue react.</p>
        )}
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs uppercase tracking-wide text-[var(--color-ink-muted)] border-b border-[var(--color-border)]">
                <th className="py-2 pr-3">Rank</th>
                <th className="py-2 pr-3">Patient</th>
                <th className="py-2 pr-3">Level</th>
                <th className="py-2 pr-3">P(Critical)</th>
                <th className="py-2 pr-3">Priority index</th>
                <th className="py-2">Why?</th>
              </tr>
            </thead>
            <tbody>
              {displayOrder.map((pid, idx) => {
                const row = ranked.find((r) => r.patient_id === pid)
                const above = displayOrder[idx - 1]
                const aboveRow = above ? ranked.find((r) => r.patient_id === above) : undefined
                return (
                  <tr key={pid} className="border-b border-[var(--color-border)]/60 hover:bg-[var(--color-surface-2)]">
                    <td className="py-3 font-mono-nums">{idx + 1}</td>
                    <td className="py-3 font-medium">Case {pid}</td>
                    <td className="py-3">{row && <TriageLevelPill level={row.recommended_level} size="sm" />}</td>
                    <td className="py-3 min-w-[140px]">{row && <ProbBar label="" value={row.p_critical} emphasize />}</td>
                    <td className="py-3 font-mono-nums">{row?.priority_index.toFixed(4) ?? '—'}</td>
                    <td className="py-3 text-xs text-[var(--color-ink-muted)] max-w-xs">
                      {row &&
                        rankExplanation(pid, row.priority_index, above, aboveRow?.priority_index)}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}
