export function KpiCard({
  label,
  value,
  unit,
  delta,
  sub,
}: {
  label: string
  value: string | number
  unit?: string
  delta?: { text: string; positive?: boolean }
  sub?: [string, string]
}) {
  return (
    <div className="rounded-[var(--radius-card)] border border-[var(--color-border)] bg-[var(--color-surface)] p-5 shadow-[var(--shadow-card)]">
      <p className="text-xs uppercase tracking-wide text-[var(--color-ink-muted)] font-medium">{label}</p>
      <div className="flex items-end gap-2 mt-2">
        <span className="text-[40px] leading-none font-semibold tabular-nums text-[var(--color-ink)]">{value}</span>
        {unit && <span className="text-sm text-[var(--color-ink-muted)] mb-1">{unit}</span>}
        {delta && (
          <span
            className={`text-xs font-medium px-2 py-0.5 rounded-full mb-2 ${
              delta.positive ? 'bg-[var(--color-p4-soft)] text-[var(--color-p4)]' : 'bg-[var(--color-p1-soft)] text-[var(--color-p1)]'
            }`}
          >
            {delta.text}
          </span>
        )}
      </div>
      {sub && (
        <div className="flex gap-4 mt-3 text-xs text-[var(--color-ink-muted)] font-mono-nums">
          <span>{sub[0]}</span>
          <span>{sub[1]}</span>
        </div>
      )}
    </div>
  )
}
