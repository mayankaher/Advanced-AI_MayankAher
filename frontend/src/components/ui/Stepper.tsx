import type { ReactNode } from 'react'

export type Step = { id: string; label: string; detail: string; content?: ReactNode }

export function PipelineStepper({
  steps,
  activeIndex,
  onStepClick,
}: {
  steps: Step[]
  activeIndex: number
  onStepClick: (index: number) => void
}) {
  return (
    <nav aria-label="Computation pipeline" className="overflow-x-auto">
      <ol className="flex min-w-max gap-2 md:gap-0 md:flex-wrap md:justify-between">
        {steps.map((step, i) => {
          const active = i === activeIndex
          return (
            <li key={step.id} className="flex items-center flex-1 min-w-[140px]">
              <button
                type="button"
                onClick={() => onStepClick(i)}
                className={`text-left w-full rounded-[var(--radius-input)] border px-3 py-2 transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-[var(--color-primary)] ${
                  active
                    ? 'border-[var(--color-primary)] bg-[var(--color-primary-soft)]'
                    : 'border-[var(--color-border)] bg-[var(--color-surface)] hover:bg-[var(--color-surface-2)]'
                }`}
              >
                <span className="text-[10px] font-mono-nums text-[var(--color-ink-muted)]">{i + 1}</span>
                <p className="text-xs font-semibold mt-0.5">{step.label}</p>
                <p className="text-[11px] text-[var(--color-ink-muted)] mt-0.5 line-clamp-2">{step.detail}</p>
              </button>
              {i < steps.length - 1 && (
                <span className="hidden md:inline text-[var(--color-ink-muted)] px-1" aria-hidden>
                  →
                </span>
              )}
            </li>
          )
        })}
      </ol>
    </nav>
  )
}
