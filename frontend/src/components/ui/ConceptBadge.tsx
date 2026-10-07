import { useLearnMode } from '../../context/LearnModeContext'
import { useState } from 'react'

const CONCEPTS: Record<
  string,
  { topic: string; intuition: string; formula: string; code: string }
> = {
  bayesian: {
    topic: 'Bayesian inference',
    intuition: 'We update beliefs about hidden severity and condition using observed evidence and the network CPTs.',
    formula: 'P(Severity, Condition | e) ∝ P(e | S, C) P(S, C)',
    code: 'backend/app/pgm/variable_elimination.py',
  },
  eu: {
    topic: 'Expected utility',
    intuition: 'Each triage level gets an expected score by averaging utility over possible true severities.',
    formula: 'EU(a) = Σ_s P(s | e) · U(a, s)',
    code: 'backend/app/decision/triage.py',
  },
  voi: {
    topic: 'Value of information',
    intuition: 'A test is worth ordering if its expected utility gain exceeds its cost in utility units.',
    formula: 'VOI(test) = EU*(after test) − EU*(before); net = VOI − cost',
    code: 'backend/app/decision/voi.py',
  },
  cmu: {
    topic: 'cμ priority rule',
    intuition: 'Queue rank uses expected harm while waiting divided by expected clinician service time.',
    formula: 'priority ∝ harm_rate × P(Critical) / E[service_minutes | evidence]',
    code: 'backend/app/decision/priority.py',
  },
}

export function ConceptBadge({ id }: { id: keyof typeof CONCEPTS }) {
  const { learnMode } = useLearnMode()
  const [open, setOpen] = useState(false)
  if (!learnMode) return null
  const c = CONCEPTS[id]
  return (
    <span className="relative inline-block ml-1">
      <button
        type="button"
        className="text-[10px] uppercase tracking-wide px-1.5 py-0.5 rounded-full bg-[var(--color-primary-soft)] text-[var(--color-primary)] font-semibold"
        onClick={() => setOpen((v) => !v)}
      >
        {c.topic}
      </button>
      {open && (
        <div className="absolute z-20 left-0 mt-1 w-72 p-3 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] shadow-[var(--shadow-card)] text-left normal-case font-normal text-xs space-y-2">
          <p>{c.intuition}</p>
          <p className="font-mono-nums text-[var(--color-ink-muted)]">{c.formula}</p>
          <p className="text-[var(--color-ink-muted)]">{c.code}</p>
        </div>
      )}
    </span>
  )
}
