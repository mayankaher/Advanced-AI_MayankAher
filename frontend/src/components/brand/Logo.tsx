import { Activity } from 'lucide-react'

export function Logo({ compact }: { compact?: boolean }) {
  return (
    <div className="flex items-center gap-2.5">
      <div className="flex h-9 w-9 items-center justify-center rounded-[var(--radius-input)] bg-[var(--color-primary-soft)] text-[var(--color-primary)]">
        <Activity size={20} strokeWidth={2.25} aria-hidden />
      </div>
      {!compact && (
        <div>
          <p className="font-bold text-[var(--color-ink)] leading-tight">TriageIQ</p>
          <p className="text-[10px] text-[var(--color-ink-muted)] leading-tight">Probabilistic triage decision support</p>
        </div>
      )}
    </div>
  )
}
