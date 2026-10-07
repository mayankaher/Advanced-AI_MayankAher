import type { ButtonHTMLAttributes, ReactNode } from 'react'

type Variant = 'primary' | 'secondary' | 'ghost'

export function Button({
  variant = 'primary',
  children,
  className = '',
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; children: ReactNode }) {
  const base =
    'inline-flex items-center justify-center gap-2 rounded-[var(--radius-input)] px-4 py-2.5 text-sm font-medium transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--color-primary)] disabled:opacity-45 disabled:pointer-events-none'
  const styles: Record<Variant, string> = {
    primary: 'bg-[var(--color-primary)] text-white hover:brightness-110',
    secondary:
      'border border-[var(--color-border)] bg-[var(--color-surface-2)] text-[var(--color-ink)] hover:bg-[var(--color-primary-soft)]',
    ghost: 'text-[var(--color-ink-muted)] hover:text-[var(--color-ink)] hover:bg-[var(--color-surface-2)]',
  }
  return (
    <button type="button" className={`${base} ${styles[variant]} ${className}`} {...props}>
      {children}
    </button>
  )
}
