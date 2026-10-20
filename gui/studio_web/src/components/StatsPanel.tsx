import { useEffect, useState } from 'react'
import { fileUrl } from '../api'

type Stats = Record<string, unknown>

const KEY_METRICS: [string, string, (v: unknown) => string][] = [
  ['signal_e_mean_mono', 'Mean signal', (v) => `${Number(v).toFixed(1)} e⁻`],
  ['K_effective_e_per_DN', 'System gain K', (v) => `${Number(v).toFixed(3)} e⁻/DN`],
  ['full_well_effective_e', 'Full well', (v) => `${Math.round(Number(v)).toLocaleString()} e⁻`],
  ['bit_depth', 'ADC', (v) => `${v} bit`],
  ['black_level_DN', 'Black level', (v) => `${v} DN`],
  ['dn_noisy_min_mono', 'Min DN (noisy)', (v) => Number(v).toFixed(1)],
  ['dn_noisy_max_mono', 'Max DN (noisy)', (v) => Number(v).toFixed(1)],
  ['dark_mean_e_per_pixel', 'Dark signal', (v) => `${Number(v).toPrecision(3)} e⁻`],
  ['temperature_c', 'Sensor temperature', (v) => `${v} °C`],
  ['bayer_pattern', 'CFA pattern', (v) => String(v)],
  ['demosaic', 'Demosaic', (v) => String(v)],
  ['preview_white_balance_method', 'Preview white balance', (v) => String(v)],
]

const show = (v: unknown) => (typeof v === 'object' && v !== null ? JSON.stringify(v) : String(v))

export function StatsPanel({ runId, path }: { runId: string; path: string }) {
  const [stats, setStats] = useState<Stats | null>(null)
  const [all, setAll] = useState(false)
  useEffect(() => {
    fetch(fileUrl(runId, path))
      .then((r) => (r.ok ? r.json() : null))
      .then(setStats)
      .catch(() => setStats(null))
  }, [runId, path])
  if (!stats) return null
  const shown = KEY_METRICS.filter(([k]) => stats[k] !== undefined && stats[k] !== null)
  return (
    <div className="space-y-3">
      <dl className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-3">
        {shown.map(([k, label, f]) => (
          <div key={k} className="rounded-lg bg-slate-950/50 px-3 py-2 ring-1 ring-slate-800">
            <dt className="text-xs text-slate-500">{label}</dt>
            <dd className="font-mono text-sm text-slate-100">{f(stats[k])}</dd>
          </div>
        ))}
      </dl>
      <button className="text-xs text-sky-300 hover:underline" onClick={() => setAll((x) => !x)}>
        {all ? 'Hide' : 'Show'} all {Object.keys(stats).length} values from run_stats.json
      </button>
      {all && (
        <div className="max-h-80 overflow-auto rounded-lg ring-1 ring-slate-800">
          <table className="w-full text-left text-xs">
            <tbody>
              {Object.entries(stats).map(([k, v]) => (
                <tr key={k} className="border-b border-slate-800/60 last:border-0">
                  <td className="px-3 py-1.5 font-mono text-slate-400">{k}</td>
                  <td className="px-3 py-1.5 font-mono break-all text-slate-200">{show(v)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
