import type { ButtonHTMLAttributes, ReactNode } from 'react'
import type { RunStatus } from '../api'

export function cx(...parts: (string | false | null | undefined)[]) {
  return parts.filter(Boolean).join(' ')
}

export function Card({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cx('rounded-2xl border border-slate-800 bg-slate-900/60 shadow-sm shadow-black/20', className)}>
      {children}
    </div>
  )
}

export function SectionTitle({
  step,
  title,
  subtitle,
  right,
}: {
  step?: number
  title: string
  subtitle?: string
  right?: ReactNode
}) {
  return (
    <div className="mb-4 flex items-start justify-between gap-4">
      <div className="flex items-start gap-3">
        {step !== undefined && (
          <span className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-full bg-sky-500/15 text-sm font-semibold text-sky-300 ring-1 ring-sky-400/30">
            {step}
          </span>
        )}
        <div>
          <h2 className="text-lg font-semibold text-slate-100">{title}</h2>
          {subtitle && <p className="text-sm text-slate-400">{subtitle}</p>}
        </div>
      </div>
      {right}
    </div>
  )
}

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger'

export function Button({
  variant = 'secondary',
  className,
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  const styles: Record<Variant, string> = {
    primary: 'bg-sky-500 text-slate-950 hover:bg-sky-400 shadow shadow-sky-500/20',
    secondary: 'bg-slate-800 text-slate-100 hover:bg-slate-700 ring-1 ring-slate-700',
    ghost: 'text-slate-300 hover:bg-slate-800 hover:text-slate-100',
    danger: 'bg-rose-500/10 text-rose-300 ring-1 ring-rose-500/30 hover:bg-rose-500/20',
  }
  return (
    <button
      className={cx(
        'inline-flex items-center justify-center gap-2 rounded-lg px-3.5 py-2 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-40',
        styles[variant],
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  )
}

export function Badge({ children, tone = 'slate' }: { children: ReactNode; tone?: 'slate' | 'sky' | 'amber' | 'emerald' | 'rose' }) {
  const tones = {
    slate: 'bg-slate-800 text-slate-300 ring-slate-700',
    sky: 'bg-sky-500/10 text-sky-300 ring-sky-400/30',
    amber: 'bg-amber-500/10 text-amber-300 ring-amber-400/30',
    emerald: 'bg-emerald-500/10 text-emerald-300 ring-emerald-400/30',
    rose: 'bg-rose-500/10 text-rose-300 ring-rose-400/30',
  }
  return (
    <span className={cx('inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-xs font-medium ring-1', tones[tone])}>
      {children}
    </span>
  )
}

const STATUS_TONE: Record<RunStatus, 'slate' | 'sky' | 'amber' | 'emerald' | 'rose'> = {
  queued: 'slate',
  running: 'sky',
  succeeded: 'emerald',
  failed: 'rose',
  cancelled: 'amber',
  interrupted: 'amber',
}

export function StatusBadge({ status }: { status: RunStatus }) {
  return (
    <Badge tone={STATUS_TONE[status]}>
      {status === 'running' && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-sky-300" />}
      {status}
    </Badge>
  )
}

export function Field({ label, hint, children }: { label: string; hint?: ReactNode; children: ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-sm font-medium text-slate-300">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-xs text-slate-500">{hint}</span>}
    </label>
  )
}

export const inputClass =
  'w-full rounded-lg border border-slate-700 bg-slate-950/60 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:border-sky-400 focus:outline-none focus:ring-2 focus:ring-sky-400/20'

export function Segmented<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T
  options: { value: T; label: string; disabled?: boolean; title?: string }[]
  onChange: (v: T) => void
}) {
  return (
    <div className="inline-flex rounded-lg bg-slate-950/60 p-1 ring-1 ring-slate-800">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          title={o.title}
          disabled={o.disabled}
          onClick={() => onChange(o.value)}
          className={cx(
            'rounded-md px-3 py-1.5 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-40',
            value === o.value ? 'bg-slate-700 text-white shadow' : 'text-slate-400 hover:text-slate-200',
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function Spinner({ className }: { className?: string }) {
  return (
    <span
      className={cx('inline-block h-4 w-4 animate-spin rounded-full border-2 border-slate-600 border-t-sky-400', className)}
    />
  )
}

export function ErrorNote({ children }: { children: ReactNode }) {
  return <div className="rounded-lg bg-rose-500/10 px-3 py-2 text-sm text-rose-200 ring-1 ring-rose-500/30">{children}</div>
}
