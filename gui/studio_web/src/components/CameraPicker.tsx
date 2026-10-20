import { Search } from 'lucide-react'
import { useMemo, useState } from 'react'
import type { Camera } from '../api'
import { Badge, cx, inputClass } from './ui'

const fmt = (v: number | null | undefined, digits = 2) =>
  v === null || v === undefined ? '–' : Number(v).toFixed(digits).replace(/\.?0+$/, '')

export function CameraFacts({ camera }: { camera: Camera }) {
  const facts: [string, string][] = [
    ['Pixel pitch', `${fmt(camera.pixel_pitch_um)} µm`],
    ['f-number', `f/${fmt(camera.f_number, 1)}`],
    ['Lens', camera.lens_camera === 'realistic' ? `traced, ${fmt(camera.focal_length_mm, 1)} mm` : camera.lens_camera ?? '–'],
    ['CFA', `${camera.cfa ?? '–'}${camera.demosaic ? ` · ${camera.demosaic}` : ''}`],
    ['ADC', `${camera.bit_depth ?? '–'} bit`],
    ['Full well', camera.full_well_e ? `${Math.round(camera.full_well_e).toLocaleString()} e⁻` : '–'],
    ['Gain K', `${fmt(camera.gain_e_per_dn)} e⁻/DN`],
    ['Read noise', `${fmt(camera.read_noise_e)} e⁻`],
  ]
  return (
    <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm sm:grid-cols-4">
      {facts.map(([k, v]) => (
        <div key={k}>
          <dt className="text-xs text-slate-500">{k}</dt>
          <dd className="font-medium text-slate-200">{v}</dd>
        </div>
      ))}
    </dl>
  )
}

export function CameraPicker({
  cameras,
  value,
  onChange,
}: {
  cameras: Camera[]
  value: string
  onChange: (id: string) => void
}) {
  const [query, setQuery] = useState('')
  const [family, setFamily] = useState('All')
  const families = useMemo(() => ['All', ...Array.from(new Set(cameras.map((c) => c.family))).sort()], [cameras])
  const shown = cameras.filter(
    (c) =>
      (family === 'All' || c.family === family) &&
      `${c.display_name} ${c.id} ${c.cfa ?? ''}`.toLowerCase().includes(query.toLowerCase()),
  )
  const selected = cameras.find((c) => c.id === value)
  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-3 sm:flex-row">
        <div className="relative flex-1">
          <Search size={16} className="pointer-events-none absolute top-2.5 left-3 text-slate-500" />
          <input
            className={cx(inputClass, 'pl-9')}
            placeholder="Search cameras (e.g. iPhone, RCCB, Nikon)"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        <select className={cx(inputClass, 'sm:w-48')} value={family} onChange={(e) => setFamily(e.target.value)}>
          {families.map((f) => (
            <option key={f}>{f}</option>
          ))}
        </select>
      </div>
      <div className="grid max-h-64 grid-cols-1 gap-2 overflow-y-auto pr-1 sm:grid-cols-2 lg:grid-cols-3">
        {shown.map((c) => (
          <button
            key={c.id}
            type="button"
            disabled={!!c.error}
            onClick={() => onChange(c.id)}
            className={cx(
              'rounded-xl border px-3 py-2 text-left transition',
              c.id === value
                ? 'border-sky-400/70 bg-sky-500/10 ring-1 ring-sky-400/40'
                : 'border-slate-800 bg-slate-950/40 hover:border-slate-600',
            )}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="truncate text-sm font-medium text-slate-100">{c.display_name}</span>
              {c.cfa && <Badge>{c.cfa}</Badge>}
            </div>
            <div className="mt-0.5 text-xs text-slate-500">
              {c.error ? 'failed to load' : `${fmt(c.pixel_pitch_um)} µm · f/${fmt(c.f_number, 1)} · ${c.family}`}
            </div>
          </button>
        ))}
        {shown.length === 0 && <p className="text-sm text-slate-500">No cameras match.</p>}
      </div>
      {selected && !selected.error && (
        <div className="rounded-xl bg-slate-950/50 p-4 ring-1 ring-slate-800">
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <span className="font-semibold text-slate-100">{selected.display_name}</span>
            {selected.calibration_tier && <Badge tone="amber">calibration: {selected.calibration_tier}</Badge>}
            {selected.hdr && <Badge tone="sky">HDR</Badge>}
            {!selected.ircf && <Badge>no IR-cut filter data</Badge>}
            <span className="font-mono text-xs text-slate-500">{selected.path}</span>
          </div>
          <CameraFacts camera={selected} />
        </div>
      )}
    </div>
  )
}
