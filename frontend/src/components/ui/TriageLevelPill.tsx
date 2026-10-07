import { TRIAGE_LABELS, TRIAGE_SHORT, triagePillClass } from '../../lib/triage-levels'

export function TriageLevelPill({ level, size = 'md' }: { level: string; size?: 'sm' | 'md' }) {
  const label = TRIAGE_LABELS[level as keyof typeof TRIAGE_LABELS] ?? level
  const short = TRIAGE_SHORT[level as keyof typeof TRIAGE_SHORT] ?? level
  const pad = size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-3 py-1 text-sm'
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border font-medium ${pad} ${triagePillClass(level)}`}
      title={label}
    >
      <span aria-hidden>{short.split(' ')[0]}</span>
      <span>{label.replace(/^P\d\s/, '')}</span>
    </span>
  )
}
