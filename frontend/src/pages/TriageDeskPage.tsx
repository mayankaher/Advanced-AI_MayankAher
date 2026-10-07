import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { motion, useReducedMotion } from 'framer-motion'
import { api, type DecideResponse, type VOIResponse } from '../lib/api'
import {
  assessmentFromDemoEvidence,
  assessmentToEvidence,
  emptyAssessment,
  measuredCount,
  EVIDENCE_FIELD_COUNT,
  type AssessmentState,
} from '../lib/evidence-form'
import { summarizePosterior, voiSummary, whyThisAction } from '../lib/narrate'
import { TRIAGE_LEVELS } from '../lib/triage-levels'
import { Card } from '../components/ui/Card'
import { Button } from '../components/ui/Button'
import { ProbBar } from '../components/ui/ProbBar'
import { TriageLevelPill } from '../components/ui/TriageLevelPill'
import { Modal } from '../components/ui/Modal'
import { Drawer } from '../components/ui/Drawer'
import { PipelineStepper, type Step } from '../components/ui/Stepper'
import { ConceptBadge } from '../components/ui/ConceptBadge'
import { EvidenceForm } from '../components/domain/EvidenceForm'

const PIPELINE_DRAWER: Record<number, { title: string; body: string; formula: string }> = {
  0: {
    title: 'Evidence collected',
    body: 'Only measured fields are sent to the network. Missing vitals remain unobserved (barren nodes in inference).',
    formula: 'e = { observed variable → state }',
  },
  1: {
    title: 'Bayesian network',
    body: 'The model links demographics, latent condition/severity, symptoms, vitals, and optional tests via CPTs from model_spec.json.',
    formula: 'P(X₁,…,Xₙ) = ∏ᵢ P(Xᵢ | parents(Xᵢ))',
  },
  2: {
    title: 'Posterior inference',
    body: 'Variable elimination computes P(Severity | e) and P(Condition | e) exactly for this discrete network.',
    formula: 'P(S | e) = Σ_{C,…} P(S, C, …, e) / P(e)',
  },
  3: {
    title: 'Expected utility',
    body: 'Each triage level is scored by averaging utility over severity states weighted by the posterior.',
    formula: 'EU(a) = Σ_s P(s | e) · U(a, s)',
  },
  4: {
    title: 'Recommendation & VOI',
    body: 'The level with highest EU is recommended. VOI compares the expected gain of each unobserved test minus cost.',
    formula: 'net_VOI = VOI(test) − cost(test)',
  },
}

export default function TriageDeskPage() {
  const [assessment, setAssessment] = useState<AssessmentState>(() => emptyAssessment('new'))
  const [demoId, setDemoId] = useState<string>('')
  const [riskAversion, setRiskAversion] = useState(0)
  const [result, setResult] = useState<{ decide: DecideResponse; voi: VOIResponse; ms: number } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [whyOpen, setWhyOpen] = useState(false)
  const [overrideOpen, setOverrideOpen] = useState(false)
  const [overrideLevel, setOverrideLevel] = useState('')
  const [overrideReason, setOverrideReason] = useState('')
  const [accepted, setAccepted] = useState(false)
  const [drawerStep, setDrawerStep] = useState<number | null>(null)
  const [expandedTest, setExpandedTest] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const reduced = useReducedMotion()

  const modelQuery = useQuery({ queryKey: ['model'], queryFn: api.getModel })
  const casesQuery = useQuery({ queryKey: ['cases'], queryFn: api.getCases })

  const evidence = useMemo(() => assessmentToEvidence(assessment), [assessment])
  const measured = measuredCount(assessment)

  const loadDemo = useCallback(
    (id: string) => {
      const c = casesQuery.data?.find((x) => x.id === id)
      if (!c) return
      setAssessment(assessmentFromDemoEvidence(id, c.evidence))
      setDemoId(id)
      setAccepted(false)
      setResult(null)
    },
    [casesQuery.data],
  )

  useEffect(() => {
    if (casesQuery.data?.length && !demoId) loadDemo('E')
  }, [casesQuery.data, demoId, loadDemo])

  const runAssess = useMutation({
    mutationFn: async () => {
      abortRef.current?.abort()
      const ac = new AbortController()
      abortRef.current = ac
      const t0 = performance.now()
      const [decide, voi] = await Promise.all([
        api.decide(evidence, riskAversion),
        api.voi(evidence),
      ])
      if (ac.signal.aborted) throw new Error('cancelled')
      return { decide, voi, ms: Math.round(performance.now() - t0) }
    },
    onSuccess: (data) => {
      setResult(data)
      setError(null)
      setAccepted(false)
    },
    onError: (e: Error) => {
      if (e.message !== 'cancelled') setError(e.message)
    },
  })

  useEffect(() => {
    if (Object.keys(evidence).length === 0) return
    const t = window.setTimeout(() => runAssess.mutate(), 250)
    return () => window.clearTimeout(t)
  }, [evidence, riskAversion])

  const overrideMut = useMutation({
    mutationFn: () =>
      api.logOverride({
        patient_id: assessment.patientId,
        recommended_level: result!.decide.recommended_level,
        chosen_level: overrideLevel,
        reason: overrideReason,
        evidence,
      }),
    onSuccess: () => {
      setOverrideOpen(false)
      setOverrideReason('')
    },
  })

  const topContributions = result?.decide.evidence_contributions?.slice(0, 4) ?? []

  const pipelineSteps: Step[] = useMemo(
    () => [
      {
        id: 'evidence',
        label: 'Evidence',
        detail: `${measured} of ${EVIDENCE_FIELD_COUNT} fields`,
      },
      {
        id: 'bn',
        label: 'Bayesian network',
        detail: `${modelQuery.data?.topological_order.length ?? 18} nodes`,
      },
      {
        id: 'infer',
        label: 'Posterior (VE)',
        detail: result ? `${result.ms} ms round-trip` : 'Waiting…',
      },
      {
        id: 'eu',
        label: 'Expected utility',
        detail: result ? `Best: ${result.decide.best_level.replace('_', ' ')}` : '—',
      },
      {
        id: 'rec',
        label: 'Recommendation',
        detail: result ? voiSummary(result.voi).slice(0, 48) + '…' : '—',
      },
    ],
    [measured, modelQuery.data, result],
  )

  return (
    <div className="space-y-6 pb-8">
      <header>
        <p className="text-xs text-[var(--color-ink-muted)]">Clinical · Triage Desk</p>
        <h1 className="text-2xl font-bold mt-1">Patient assessment</h1>
        <p className="text-sm text-[var(--color-ink-muted)] mt-1 max-w-2xl">
          Enter evidence, run exact inference, and see the expected-utility triage recommendation with plain-language explanations.
        </p>
      </header>

      <div className="grid xl:grid-cols-2 gap-6 items-start">
        <Card title="Patient assessment" subtitle="Input">
          <div className="flex flex-wrap gap-2 mb-4">
            <label className="text-sm flex-1 min-w-[10rem]">
              <span className="text-xs text-[var(--color-ink-muted)]">Load demo case</span>
              <select
                value={demoId}
                onChange={(e) => loadDemo(e.target.value)}
                className="block mt-1 w-full rounded-[var(--radius-input)] border border-[var(--color-border)] px-3 py-2 bg-[var(--color-surface)]"
              >
                {(casesQuery.data ?? []).map((c) => (
                  <option key={c.id} value={c.id}>
                    Case {c.id}
                  </option>
                ))}
              </select>
            </label>
            <div className="flex items-end gap-2">
              <Button onClick={() => runAssess.mutate()} disabled={runAssess.isPending || !Object.keys(evidence).length}>
                Assess
              </Button>
              <Button
                variant="secondary"
                onClick={() => {
                  setAssessment(emptyAssessment('new'))
                  setDemoId('')
                  setResult(null)
                }}
              >
                Reset
              </Button>
            </div>
          </div>
          <EvidenceForm state={assessment} onChange={setAssessment} model={modelQuery.data} />
        </Card>

        <div className="space-y-4">
          {runAssess.isPending && !result && (
            <p className="text-sm text-[var(--color-ink-muted)] animate-pulse">Updating recommendation…</p>
          )}
          {error && (
            <div className="rounded-[var(--radius-input)] border border-[var(--color-p1)]/40 bg-[var(--color-p1-soft)] px-3 py-2 text-sm">
              {error}
              <Button variant="ghost" className="ml-2 !py-1" onClick={() => runAssess.mutate()}>
                Retry
              </Button>
            </div>
          )}

          {result && (
            <>
              <motion.div
                key={result.decide.recommended_level}
                initial={reduced ? false : { boxShadow: '0 0 0 0 rgba(47,111,237,0)' }}
                animate={reduced ? {} : { boxShadow: ['0 0 0 0 rgba(47,111,237,0)', '0 0 0 4px rgba(47,111,237,.25)', '0 0 0 0 rgba(47,111,237,0)'] }}
                transition={{ duration: 0.8 }}
                className="rounded-[var(--radius-card)] overflow-hidden border border-[var(--color-border)] shadow-[var(--shadow-card)]"
              >
                <div className="bg-[var(--color-slate-header)] text-white px-5 py-4 flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <p className="text-xs uppercase tracking-wide opacity-80">Recommended</p>
                    <div className="flex items-center gap-2 mt-1">
                      <TriageLevelPill level={result.decide.recommended_level} />
                      <ConceptBadge id="eu" />
                    </div>
                  </div>
                  <p className="font-mono-nums text-sm">
                    P(Critical) = {(result.decide.p_severity.Critical * 100).toFixed(2)}%
                    <ConceptBadge id="bayesian" />
                  </p>
                </div>
                <div className="bg-[var(--color-surface)] p-5 space-y-4">
                  <p className="text-sm leading-relaxed">{summarizePosterior(result.decide.p_severity, result.decide.p_condition)}</p>
                  <ul className="text-sm space-y-1">
                    {topContributions.map((c) => (
                      <li key={c.variable} className="flex justify-between gap-2">
                        <span>
                          {c.impact_on_p_critical >= 0 ? '▲' : '▼'} {c.variable} = {c.value}
                        </span>
                        <span className="font-mono-nums text-[var(--color-ink-muted)]">
                          ΔP(Crit) {c.impact_on_p_critical >= 0 ? '+' : ''}
                          {(c.impact_on_p_critical * 100).toFixed(2)}%
                        </span>
                      </li>
                    ))}
                  </ul>
                  <div className="flex flex-wrap gap-2 pt-2">
                    <Button
                      onClick={() => {
                        setAccepted(true)
                      }}
                      disabled={accepted}
                    >
                      {accepted ? 'Recommendation acknowledged' : 'Accept recommendation'}
                    </Button>
                    <Button variant="secondary" onClick={() => setOverrideOpen(true)}>
                      Override with reason
                    </Button>
                    <Button variant="ghost" onClick={() => setWhyOpen((v) => !v)}>
                      Why this?
                    </Button>
                  </div>
                  {whyOpen && (
                    <div className="text-sm space-y-2 border-t border-[var(--color-border)] pt-3 text-[var(--color-ink-muted)]">
                      <p>
                        <strong className="text-[var(--color-ink)]">Evidence:</strong> top factors above shift P(Critical).
                      </p>
                      <p>
                        <strong className="text-[var(--color-ink)]">Beliefs:</strong> {summarizePosterior(result.decide.p_severity, result.decide.p_condition)}
                      </p>
                      <p>
                        <strong className="text-[var(--color-ink)]">Action:</strong> {whyThisAction(result.decide)}
                      </p>
                    </div>
                  )}
                </div>
              </motion.div>

              <Card title="Severity posterior" subtitle="Probabilities">
                {Object.entries(result.decide.p_severity).map(([k, v]) => (
                  <ProbBar key={k} label={k} value={v} emphasize={k === 'Critical'} />
                ))}
              </Card>

              <Card title="Expected utility by triage level" subtitle="Decision">
                <div className="space-y-2">
                  {TRIAGE_LEVELS.map((level) => {
                    const eu = result.decide.eu_by_level[level]
                    const best = level === result.decide.best_level
                    const maxAbs = Math.max(...Object.values(result.decide.eu_by_level).map(Math.abs))
                    const width = maxAbs ? (Math.abs(eu) / maxAbs) * 100 : 0
                    return (
                      <div key={level} className={best ? 'ring-2 ring-[var(--color-primary)]/30 rounded-lg p-2 -mx-2' : ''}>
                        <div className="flex justify-between text-sm mb-1">
                          <TriageLevelPill level={level} size="sm" />
                          <span className="font-mono-nums">{eu?.toFixed(3)}</span>
                        </div>
                        <div className="h-2 rounded-full bg-[var(--color-surface-2)]">
                          <div className="h-full rounded-full bg-[var(--color-primary)]/70" style={{ width: `${width}%` }} />
                        </div>
                      </div>
                    )
                  })}
                </div>
                <p className="text-xs text-[var(--color-ink-muted)] mt-3" title="Expected utility averages the utility matrix row using P(Severity | evidence)">
                  EU(level) = Σ P(severity | e) × U(level, severity)
                </p>
              </Card>

              <Card title="Next best diagnostic test" subtitle="Value of information">
                <p className="text-sm mb-3">{voiSummary(result.voi)}</p>
                <ConceptBadge id="voi" />
                <ul className="mt-3 space-y-2">
                  {result.voi.test_analyses?.map((t) => (
                    <li key={t.test} className="border border-[var(--color-border)] rounded-[var(--radius-input)] overflow-hidden">
                      <button
                        type="button"
                        className="w-full flex justify-between px-3 py-2 text-sm hover:bg-[var(--color-surface-2)]"
                        onClick={() => setExpandedTest(expandedTest === t.test ? null : t.test)}
                      >
                        <span>{t.test}</span>
                        <span className="font-mono-nums text-[var(--color-ink-muted)]">
                          net {t.net_voi >= 0 ? '+' : ''}
                          {t.net_voi.toFixed(2)} (cost {t.cost.toFixed(1)})
                        </span>
                      </button>
                      {expandedTest === t.test && t.outcomes && (
                        <div className="px-3 pb-2 text-xs text-[var(--color-ink-muted)] space-y-1 border-t border-[var(--color-border)]">
                          {t.outcomes.map((o) => (
                            <p key={o.outcome}>
                              If {o.outcome}: P(outcome)={(o.p_outcome * 100).toFixed(1)}% → {o.best_level.replace('_', ' ')} (EU {o.best_eu.toFixed(3)})
                            </p>
                          ))}
                        </div>
                      )}
                    </li>
                  ))}
                </ul>
              </Card>

              <Card title="What-if" subtitle="Risk aversion">
                <label className="text-sm block">
                  Risk aversion λ = {riskAversion.toFixed(2)}
                  <input
                    type="range"
                    min={0}
                    max={2}
                    step={0.05}
                    value={riskAversion}
                    onChange={(e) => setRiskAversion(parseFloat(e.target.value))}
                    className="w-full mt-2"
                  />
                </label>
                <p className="text-xs text-[var(--color-ink-muted)] mt-2">Re-queries expected utility when λ changes (debounced 250 ms).</p>
              </Card>
            </>
          )}
        </div>
      </div>

      <Card title="How this was computed" subtitle="Pipeline">
        <PipelineStepper steps={pipelineSteps} activeIndex={result ? 4 : 1} onStepClick={setDrawerStep} />
        <p className="text-xs text-[var(--color-ink-muted)] mt-3">
          Click a step for details.{' '}
          <Link to="/inference" className="text-[var(--color-primary)] underline">
            Open Inference Lab
          </Link>
        </p>
      </Card>

      <Drawer open={drawerStep != null} onClose={() => setDrawerStep(null)} title={drawerStep != null ? PIPELINE_DRAWER[drawerStep]?.title ?? '' : ''}>
        {drawerStep != null && (
          <>
            <p className="mb-3">{PIPELINE_DRAWER[drawerStep].body}</p>
            <p className="font-mono-nums text-[var(--color-ink-muted)]">{PIPELINE_DRAWER[drawerStep].formula}</p>
            {drawerStep === 2 && result && (
              <p className="mt-3">
                Live: P(Critical) = {(result.decide.p_severity.Critical * 100).toFixed(2)}%, P(Moderate) ={' '}
                {(result.decide.p_severity.Moderate * 100).toFixed(2)}%
              </p>
            )}
          </>
        )}
      </Drawer>

      <Modal
        open={overrideOpen}
        onClose={() => setOverrideOpen(false)}
        title="Override with documented reason"
        footer={
          <>
            <Button variant="secondary" onClick={() => setOverrideOpen(false)}>
              Cancel
            </Button>
            <Button
              disabled={!overrideLevel || !overrideReason.trim() || overrideMut.isPending}
              onClick={() => overrideMut.mutate()}
            >
              Log override
            </Button>
          </>
        }
      >
        <div className="space-y-3 mt-2">
          <label className="block">
            Chosen level
            <select
              value={overrideLevel}
              onChange={(e) => setOverrideLevel(e.target.value)}
              className="block w-full mt-1 rounded-[var(--radius-input)] border border-[var(--color-border)] px-3 py-2"
            >
              <option value="">Select…</option>
              {TRIAGE_LEVELS.map((l) => (
                <option key={l} value={l}>
                  {l.replace('_', ' ')}
                </option>
              ))}
            </select>
          </label>
          <label className="block">
            Reason (required)
            <textarea
              value={overrideReason}
              onChange={(e) => setOverrideReason(e.target.value)}
              className="block w-full mt-1 min-h-[5rem] rounded-[var(--radius-input)] border border-[var(--color-border)] px-3 py-2"
            />
          </label>
        </div>
      </Modal>
    </div>
  )
}
