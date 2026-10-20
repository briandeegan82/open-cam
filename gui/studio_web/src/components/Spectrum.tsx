import { useEffect, useState } from 'react'
import { api, type Spectrum as SpectrumData } from '../api'

// Small SPD plot with a visible-spectrum gradient underneath.
export function SpectrumPlot({ illuminantId }: { illuminantId: string | null }) {
  const [data, setData] = useState<SpectrumData | null>(null)
  useEffect(() => {
    let live = true
    if (illuminantId) api.spectrum(illuminantId).then((d) => live && setData(d)).catch(() => live && setData(null))
    return () => {
      live = false
    }
  }, [illuminantId])
  if (!illuminantId || !data || data.wavelength_nm.length < 2) return null
  const W = 300
  const H = 80
  const lo = 380
  const hi = 780
  const pts = data.wavelength_nm
    .map((w, i) => [w, data.value[i]] as const)
    .filter(([w]) => w >= lo && w <= hi)
  const vmax = Math.max(...pts.map(([, v]) => v), 1e-12)
  const path = pts
    .map(([w, v], i) => `${i ? 'L' : 'M'}${(((w - lo) / (hi - lo)) * W).toFixed(1)},${(H - (v / vmax) * (H - 6)).toFixed(1)}`)
    .join(' ')
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H + 14}`} className="w-full">
        <defs>
          <linearGradient id="vis" x1="0" x2="1">
            <stop offset="0%" stopColor="#6d28d9" />
            <stop offset="18%" stopColor="#2563eb" />
            <stop offset="32%" stopColor="#06b6d4" />
            <stop offset="45%" stopColor="#22c55e" />
            <stop offset="55%" stopColor="#facc15" />
            <stop offset="65%" stopColor="#f97316" />
            <stop offset="80%" stopColor="#dc2626" />
            <stop offset="100%" stopColor="#450a0a" />
          </linearGradient>
          <clipPath id="under">
            <path d={`${path} L${W},${H} L0,${H} Z`} />
          </clipPath>
        </defs>
        <rect width={W} height={H} fill="url(#vis)" opacity="0.35" clipPath="url(#under)" />
        <path d={path} fill="none" stroke="#e2e8f0" strokeWidth="1.5" />
        {[400, 500, 600, 700].map((w) => (
          <text key={w} x={((w - lo) / (hi - lo)) * W} y={H + 12} fontSize="9" fill="#64748b" textAnchor="middle">
            {w}
          </text>
        ))}
      </svg>
      <p className="text-center text-[11px] text-slate-500">relative spectral power vs wavelength (nm)</p>
    </div>
  )
}
