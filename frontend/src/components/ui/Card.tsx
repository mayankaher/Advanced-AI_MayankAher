import type { ReactNode } from 'react'

export function Card({
  children,
  className = '',
  title,
  subtitle,
}: {
  children: ReactNode
  className?: string
  title?: string
  subtitle?: string
}) {
  return (
    <div
      className={`rounded-[var(--radius-card)] border border-[var(--color-border)] bg-[var(--color-surface)] shadow-[var(--shadow-card)] ${className}`}
    >
      {(title || subtitle) && (
        <div className="px-5 pt-5 pb-2">
          {subtitle && (
            <p className="text-xs uppercase tracking-wide text-[var(--color-ink-muted)] font-medium">{subtitle}</p>
          )}
          {title && <h2 className="text-lg font-semibold text-[var(--color-ink)] mt-0.5">{title}</h2>}
        </div>
      )}
      <div className="p-5 pt-3">{children}</div>
    </div>
  )
}
