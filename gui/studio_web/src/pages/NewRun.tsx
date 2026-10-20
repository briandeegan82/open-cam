import { Clock, Cpu, Play, Sparkles, Terminal } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { api, formatDuration, type Camera, type Illuminant, type Mode, type Plan, type QualityPreset, type RunRequest, type Scene, type Status } from '../api'
import { CameraPicker } from '../components/CameraPicker'
import { CommandBlock } from '../components/CommandBlock'
import { SceneArt } from '../components/SceneArt'
import { SpectrumPlot } from '../components/Spectrum'
import { Badge, Button, Card, ErrorNote, Field, SectionTitle, Segmented, Spinner, cx, inputClass } from '../components/ui'
import { navigate } from '../router'

const DEFAULT_REQUEST: RunRequest = {
  scene_id: 'colorchecker',
  camera_id: 'iphone_8',
  mode: 'fast_analytic',
  illuminant_id: 'D65',
  target_illuminance_lux: 500,
  exposure_time_s: null,
  seed: 0,
  xres: 480,
  yres: 320,
  pixelsamples: 128,
  name: '',
}

const LUX_STOPS = [
  { lux: 1, label: 'Deep twilight' },
  { lux: 10, label: 'Street lighting' },
  { lux: 100, label: 'Dim room' },
  { lux: 500, label: 'Office' },
  { lux: 2000, label: 'Overcast day' },
  { lux: 20000, label: 'Daylight shade' },
  { lux: 100000, label: 'Direct sun' },
]

export function NewRun({ status, fromRun }: { status: Status | null; fromRun: string | null }) {
  const [scenes, setScenes] = useState<Scene[]>([])
  const [cameras, setCameras] = useState<Camera[]>([])
  const [illuminants, setIlluminants] = useState<Illuminant[]>([])
  const [presets, setPresets] = useState<QualityPreset[]>([])
  const [req, setReq] = useState<RunRequest>(DEFAULT_REQUEST)
  const [plan, setPlan] = useState<Plan | null>(null)
  const [showCli, setShowCli] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    Promise.all([api.scenes(), api.cameras(), api.illuminants(), api.presets()])
      .then(([s, c, i, p]) => {
        setScenes(s)
        setCameras(c)
        setIlluminants(i)
        setPresets(p)
      })
      .catch((e) => setError(String(e.message ?? e)))
  }, [])

  useEffect(() => {
    if (fromRun) api.run(fromRun).then((r) => setReq({ ...DEFAULT_REQUEST, ...r.request, name: '' })).catch(() => {})
  }, [fromRun])

  const scene = scenes.find((s) => s.id === req.scene_id)
  const set = (patch: Partial<RunRequest>) => setReq((r) => ({ ...r, ...patch }))

  // Keep the mode valid for the chosen scene and the installed tools.
  useEffect(() => {
    if (!scene) return
    if (!scene.modes.includes(req.mode)) set({ mode: scene.modes[0] })
  }, [scene, req.mode])

  useEffect(() => {
    const t = setTimeout(() => {
      api.plan(req).then(setPlan).catch(() => setPlan(null))
    }, 250)
    return () => clearTimeout(t)
  }, [req])

  const activePreset = presets.find((p) => p.xres === req.xres && p.yres === req.yres && p.pixelsamples === req.pixelsamples)
  const pbrtMissing = status !== null && !status.pbrt_available
  const blocked = req.mode === 'pbrt_accurate' && pbrtMissing
  const luxLog = Math.log10(req.target_illuminance_lux)
  const luxLabel = useMemo(
    () => LUX_STOPS.reduce((best, s) => (Math.abs(Math.log10(s.lux) - luxLog) < Math.abs(Math.log10(best.lux) - luxLog) ? s : best)).label,
    [luxLog],
  )

  const submit = async () => {
    setSubmitting(true)
    setError(null)
    try {
      const run = await api.createRun(req)
      navigate(`/runs/${run.id}`)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="grid gap-6 xl:grid-cols-[1fr_340px]">
      <div className="space-y-6">
        <Card className="p-5">
          <SectionTitle step={1} title="Choose a scene" subtitle="What the camera looks at." />
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            {scenes.map((s) => (
              <button
                key={s.id}
                type="button"
                onClick={() => set({ scene_id: s.id })}
                className={cx(
                  'group overflow-hidden rounded-xl border text-left transition',
                  s.id === req.scene_id ? 'border-sky-400/70 ring-2 ring-sky-400/30' : 'border-slate-800 hover:border-slate-600',
                )}
              >
                <div className="aspect-[3/2] overflow-hidden bg-slate-950">
                  <SceneArt id={s.id} />
                </div>
                <div className="p-3">
                  <div className="text-sm font-medium text-slate-100">{s.label.replace(/ \(.*\)$/, '')}</div>
                  <div className="mt-1 flex gap-1">
                    {s.modes.includes('fast_analytic') && <Badge tone="emerald">instant preview</Badge>}
                    <Badge>PBRT</Badge>
                  </div>
                </div>
              </button>
            ))}
          </div>
          {scene && <p className="mt-3 text-sm text-slate-400">{scene.description}</p>}
          <p className="mt-2 text-xs text-slate-500">
            Highway, urban, tunnel, cabin and the other scene builders are coming in the next Studio release; they are available from the CLI today.
          </p>
        </Card>

        <Card className="p-5">
          <SectionTitle step={2} title="Choose a camera" subtitle={`${cameras.length} camera recipes from config/camera_recipes.`} />
          {cameras.length ? <CameraPicker cameras={cameras} value={req.camera_id} onChange={(id) => set({ camera_id: id })} /> : <Spinner />}
        </Card>

        <Card className="p-5">
          <SectionTitle step={3} title="Lighting and exposure" />
          <div className="grid gap-6 md:grid-cols-2">
            <div className="space-y-4">
              <Field
                label="Light source"
                hint={scene && !scene.supports_illuminant ? 'This target uses its own built-in light; the spectrum is not configurable.' : undefined}
              >
                <select
                  className={inputClass}
                  disabled={!scene?.supports_illuminant}
                  value={req.illuminant_id ?? ''}
                  onChange={(e) => set({ illuminant_id: e.target.value || null })}
                >
                  {illuminants.map((i) => (
                    <option key={i.id} value={i.id}>
                      {i.label}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label={`Illuminance on the chart: ${req.target_illuminance_lux.toLocaleString(undefined, { maximumSignificantDigits: 3 })} lux`} hint={`About: ${luxLabel}`}>
                <input
                  type="range"
                  className="w-full"
                  min={0}
                  max={5}
                  step={0.01}
                  value={luxLog}
                  onChange={(e) => set({ target_illuminance_lux: Number(Math.pow(10, Number(e.target.value)).toPrecision(3)) })}
                />
              </Field>
              <Field label="Exposure time" hint="Auto uses the camera recipe's integration time.">
                <div className="flex items-center gap-3">
                  <Segmented
                    value={req.exposure_time_s === null ? 'auto' : 'manual'}
                    onChange={(v) => set({ exposure_time_s: v === 'auto' ? null : (cameras.find((c) => c.id === req.camera_id)?.integration_time_s ?? 0.01) })}
                    options={[
                      { value: 'auto', label: 'Recipe' },
                      { value: 'manual', label: 'Manual' },
                    ]}
                  />
                  {req.exposure_time_s !== null && (
                    <div className="flex items-center gap-2">
                      <input
                        type="number"
                        className={cx(inputClass, 'w-28')}
                        min={0.01}
                        step={0.1}
                        value={+(req.exposure_time_s * 1000).toPrecision(4)}
                        onChange={(e) => set({ exposure_time_s: Math.max(1e-5, Number(e.target.value) / 1000) })}
                      />
                      <span className="text-sm text-slate-400">ms</span>
                    </div>
                  )}
                </div>
              </Field>
            </div>
            {scene?.supports_illuminant && <SpectrumPlot illuminantId={req.illuminant_id} />}
          </div>
        </Card>

        <Card className="p-5">
          <SectionTitle step={4} title="Simulation quality" />
          <div className="space-y-5">
            <Field label="Method">
              <div className="grid gap-3 sm:grid-cols-2">
                {(
                  [
                    ['fast_analytic', 'Instant (analytic)', 'Seconds. Integrates the chart spectra directly, no ray tracing, no lens blur.'],
                    ['pbrt_accurate', 'Physically accurate (PBRT)', 'Spectral path tracing through the camera lens. Minutes to hours.'],
                  ] as [Mode, string, string][]
                ).map(([m, title, desc]) => {
                  const disabled = !scene?.modes.includes(m)
                  return (
                    <button
                      key={m}
                      type="button"
                      disabled={disabled}
                      onClick={() => set({ mode: m })}
                      className={cx(
                        'rounded-xl border p-3 text-left transition disabled:cursor-not-allowed disabled:opacity-40',
                        req.mode === m ? 'border-sky-400/70 bg-sky-500/10' : 'border-slate-800 hover:border-slate-600',
                      )}
                    >
                      <div className="flex items-center gap-2 text-sm font-medium text-slate-100">
                        {m === 'fast_analytic' ? <Sparkles size={16} /> : <Cpu size={16} />} {title}
                      </div>
                      <p className="mt-1 text-xs text-slate-400">{disabled ? 'Not available for this scene.' : desc}</p>
                    </button>
                  )
                })}
              </div>
            </Field>
            {pbrtMissing && req.mode === 'pbrt_accurate' && (
              <ErrorNote>PBRT is not built on this machine. Run tools/build_pbrt.sh (see docs/BUILD_PBRT.txt), or use the instant method.</ErrorNote>
            )}
            {req.mode === 'pbrt_accurate' && (
              <>
                <Field label="Render quality">
                  <div className="grid gap-3 sm:grid-cols-3">
                    {presets.map((p) => (
                      <button
                        key={p.id}
                        type="button"
                        onClick={() => set({ xres: p.xres, yres: p.yres, pixelsamples: p.pixelsamples })}
                        className={cx(
                          'rounded-xl border p-3 text-left transition',
                          activePreset?.id === p.id ? 'border-sky-400/70 bg-sky-500/10' : 'border-slate-800 hover:border-slate-600',
                        )}
                      >
                        <div className="text-sm font-medium text-slate-100">{p.label}</div>
                        <div className="font-mono text-xs text-slate-400">
                          {p.xres}×{p.yres} · {p.pixelsamples} spp
                        </div>
                        <p className="mt-1 text-xs text-slate-500">{p.description}</p>
                      </button>
                    ))}
                  </div>
                </Field>
                <details className="group">
                  <summary className="cursor-pointer text-sm text-slate-400 hover:text-slate-200">Custom resolution and samples</summary>
                  <div className="mt-3 grid grid-cols-3 gap-3">
                    <Field label="Width (px)">
                      <input type="number" className={inputClass} min={16} max={8192} value={req.xres} onChange={(e) => set({ xres: Number(e.target.value) })} />
                    </Field>
                    <Field label="Height (px)">
                      <input type="number" className={inputClass} min={16} max={8192} value={req.yres} onChange={(e) => set({ yres: Number(e.target.value) })} />
                    </Field>
                    <Field label="Samples per pixel">
                      <input type="number" className={inputClass} min={1} max={65536} value={req.pixelsamples} onChange={(e) => set({ pixelsamples: Number(e.target.value) })} />
                    </Field>
                  </div>
                </details>
              </>
            )}
            <div className="grid grid-cols-2 gap-3">
              <Field label="Noise seed" hint="Same seed, same noise pattern.">
                <input type="number" className={inputClass} value={req.seed} onChange={(e) => set({ seed: Math.trunc(Number(e.target.value)) })} />
              </Field>
              <Field label="Run name (optional)">
                <input className={inputClass} maxLength={80} placeholder="e.g. office-lux-test" value={req.name} onChange={(e) => set({ name: e.target.value })} />
              </Field>
            </div>
          </div>
        </Card>
      </div>

      <aside className="xl:sticky xl:top-6 xl:self-start">
        <Card className="space-y-4 p-5">
          <h2 className="text-lg font-semibold text-slate-100">Summary</h2>
          <dl className="space-y-2 text-sm">
            {[
              ['Scene', scene?.label.replace(/ \(.*\)$/, '') ?? '–'],
              ['Camera', cameras.find((c) => c.id === req.camera_id)?.display_name ?? req.camera_id],
              ['Light', scene?.supports_illuminant ? `${req.illuminant_id} · ${req.target_illuminance_lux.toLocaleString()} lux` : `${req.target_illuminance_lux.toLocaleString()} lux`],
              ['Exposure', req.exposure_time_s === null ? 'recipe default' : `${+(req.exposure_time_s * 1000).toPrecision(4)} ms`],
              ['Method', req.mode === 'fast_analytic' ? 'Instant (analytic)' : `PBRT ${req.xres}×${req.yres}, ${req.pixelsamples} spp`],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between gap-3">
                <dt className="text-slate-500">{k}</dt>
                <dd className="text-right text-slate-200">{v}</dd>
              </div>
            ))}
          </dl>
          {plan && (
            <div className="flex items-center gap-2 rounded-lg bg-slate-950/50 px-3 py-2 text-sm text-slate-300 ring-1 ring-slate-800">
              <Clock size={16} className="text-slate-500" /> Estimated time ≈ {formatDuration(plan.estimate_s)}
              {status && req.mode === 'pbrt_accurate' && <span className="text-xs text-slate-500">({status.cpu_count} CPU cores)</span>}
            </div>
          )}
          {error && <ErrorNote>{error}</ErrorNote>}
          <Button variant="primary" className="w-full py-2.5 text-base" disabled={submitting || blocked || !scene} onClick={submit}>
            {submitting ? <Spinner /> : <Play size={18} />} Run simulation
          </Button>
          <Button variant="ghost" className="w-full" onClick={() => setShowCli((x) => !x)}>
            <Terminal size={16} /> {showCli ? 'Hide' : 'Show'} equivalent CLI commands
          </Button>
          {showCli && plan && <CommandBlock script={plan.script} />}
        </Card>
      </aside>
    </div>
  )
}
