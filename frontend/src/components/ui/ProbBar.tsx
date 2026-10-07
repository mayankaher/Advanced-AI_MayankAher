import { motion, useReducedMotion } from 'framer-motion'
import { severityBarColor } from '../../lib/triage-levels'

export function ProbBar({
  label,
  value,
  emphasize,
}: {
  label: string
  value: number
  emphasize?: boolean
}) {
  const reduced = useReducedMotion()
  const pct = Math.min(100, Math.max(0, value * 100))
  const color = severityBarColor(label)

  return (
    <div className="space-y-1.5" role="group" aria-label={`${label} probability ${pct.toFixed(1)} percent`}>
      <div className="flex justify-between items-baseline text-sm">
        <span className={emphasize ? 'font-semibold' : 'text-[var(--color-ink)]'}>{label}</span>
        <span className="font-mono-nums text-[var(--color-ink-muted)]">
          {(value * 100).toFixed(1)}
          <span className="text-xs ml-0.5">%</span>
        </span>
      </div>
      <div className="h-2.5 rounded-full bg-[var(--color-surface-2)] border border-[var(--color-border)] overflow-hidden">
        <motion.div
          className="h-full rounded-full"
          style={{ backgroundColor: color }}
          initial={false}
          animate={{ width: `${pct}%` }}
          transition={reduced ? { duration: 0 } : { duration: 0.4, ease: 'easeOut' }}
        />
      </div>
    </div>
  )
}
