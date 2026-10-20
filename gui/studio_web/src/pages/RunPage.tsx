import { ArrowLeft, Download, FileDown, RotateCcw, Square, Trash2 } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { api, fileUrl, formatDuration, formatTime, type Run } from '../api'
import { CommandBlock } from '../components/CommandBlock'
import { ImageViewer, type ViewerImage } from '../components/ImageViewer'
import { StatsPanel } from '../components/StatsPanel'
import { Button, Card, ErrorNote, SectionTitle, Spinner, StatusBadge, cx } from '../components/ui'
import { navigate } from '../router'

const PREVIEWS: [string, string][] = [
  ['preview/noisy_demosaic_rgb8.png', 'Noisy · demosaiced'],
  ['preview/clean_demosaic_rgb8.png', 'Clean · demosaiced'],
  ['preview/noisy_rgb8.png', 'Noisy · raw colour'],
  ['preview/clean_rgb8.png', 'Clean · raw colour'],
  ['preview/noisy_mono_16.png', 'Noisy · raw mosaic'],
]

const DOWNLOADS: [string, string][] = [
  ['noisy.raw16', 'RAW16 mosaic'],
  ['electrons.npz', 'Electrons (NPZ)'],
  ['spectral.exr', 'Spectral render (EXR)'],
  ['preview/run_stats.json', 'Run statistics (JSON)'],
]

function useRun(id: string) {
  const [run, setRun] = useState<Run | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    let live = true
    let timer: number | undefined
    const tick = async () => {
      try {
        const r = await api.run(id)
        if (!live) return
        setRun(r)
        if (r.status === 'queued' || r.status === 'running') timer = window.setTimeout(tick, 1000)
      } catch (e) {
        if (live) setError(e instanceof Error ? e.message : String(e))
      }
    }
    tick()
    return () => {
      live = false
      window.clearTimeout(timer)
    }
  }, [id])
  return { run, error, setRun }
}

function LogPanel({ id, active }: { id: string; active: boolean }) {
  const [text, setText] = useState('')
  const offset = useRef(0)
  const pre = useRef<HTMLPreElement>(null)
  useEffect(() => {
    let live = true
    let timer: number | undefined
    offset.current = 0
    setText('')
    const tick = async () => {
      try {
        const r = await api.log(id, offset.current)
        if (!live) return
        if (r.text) {
          offset.current = r.offset
          setText((t) => (t + r.text).slice(-400_000))
        }
        if (r.status === 'queued' || r.status === 'running' || r.text) timer = window.setTimeout(tick, 700)
      } catch {
        timer = window.setTimeout(tick, 2000)
      }
    }
    tick()
    return () => {
      live = false
      window.clearTimeout(timer)
    }
  }, [id, active])
  useEffect(() => {
    const el = pre.current
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 80) el.scrollTop = el.scrollHeight
  }, [text])
  return (
    <pre ref={pre} className="h-72 overflow-auto rounded-xl bg-slate-950 p-3 font-mono text-[11px] leading-relaxed whitespace-pre-wrap text-slate-400 ring-1 ring-slate-800">
      {text || 'Waiting for output…'}
    </pre>
  )
}

function Elapsed({ run }: { run: Run }) {
  const [, force] = useState(0)
  useEffect(() => {
    if (run.status !== 'running') return
    const t = window.setInterval(() => force((x) => x + 1), 1000)
    return () => window.clearInterval(t)
  }, [run.status])
  if (!run.started) return null
  const end = run.finished ?? Date.now() / 1000
  return <span>{formatDuration(end - run.started)}</span>
}

export function RunPage({ id }: { id: string }) {
  const { run, error, setRun } = useRun(id)
  const [script, setScript] = useState<string | null>(null)
  const [tab, setTab] = useState<'results' | 'log' | 'cli'>('results')

  useEffect(() => {
    fetch(fileUrl(id, 'run.sh')).then((r) => (r.ok ? r.text() : null)).then(setScript).catch(() => {})
  }, [id, run?.status])

  if (error) return <ErrorNote>{error}</ErrorNote>
  if (!run) return <Spinner />

  const active = run.status === 'queued' || run.status === 'running'
  const images: ViewerImage[] = PREVIEWS.filter(([p]) => run.outputs.includes(p)).map(([p, label]) => ({
    key: p,
    label,
    url: `${fileUrl(run.id, `outputs/${p}`)}?t=${run.finished ?? 0}`,
  }))
  const done = run.steps.filter((s) => s.returncode !== null).length
  const req = run.request
  const statsPath = ['preview/run_stats.json', 'run_stats.json'].find((p) => run.outputs.includes(p))

  const cancel = async () => setRun(await api.cancelRun(run.id))
  const remove = async () => {
    if (!window.confirm('Delete this run and its outputs?')) return
    await api.deleteRun(run.id)
    navigate('/runs')
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <Button variant="ghost" onClick={() => navigate('/runs')}>
          <ArrowLeft size={16} /> History
        </Button>
        <h1 className="text-xl font-semibold text-slate-100">{req.name || run.id}</h1>
        <StatusBadge status={run.status} />
        <span className="text-sm text-slate-500">
          {formatTime(run.created)} · <Elapsed run={run} />
        </span>
        <div className="ml-auto flex gap-2">
          {active ? (
            <Button variant="danger" onClick={cancel}>
              <Square size={14} /> Cancel
            </Button>
          ) : (
            <>
              <Button onClick={() => navigate(`/new?from=${encodeURIComponent(run.id)}`)}>
                <RotateCcw size={16} /> Edit and re-run
              </Button>
              <Button variant="danger" onClick={remove}>
                <Trash2 size={16} />
              </Button>
            </>
          )}
        </div>
      </div>

      {active && (
        <Card className="p-5">
          <div className="mb-3 flex items-center justify-between text-sm">
            <span className="text-slate-300">
              {run.status === 'queued'
                ? `Waiting in queue${run.queue_position ? ` (position ${run.queue_position})` : ''}…`
                : run.steps[done]?.name ?? 'Starting…'}
            </span>
            <span className="text-slate-500">
              step {Math.min(done + 1, Math.max(run.steps.length, 1))} of {run.steps.length || '…'} · estimate ≈ {formatDuration(run.estimate_s)}
            </span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-slate-800">
            <div
              className="h-full rounded-full bg-gradient-to-r from-sky-500 to-cyan-300 transition-all"
              style={{ width: `${run.steps.length ? Math.max(4, (done / run.steps.length) * 100) : 4}%` }}
            />
          </div>
        </Card>
      )}

      {run.error && <ErrorNote>{run.error}. See the log for details.</ErrorNote>}

      <div className="flex gap-1 border-b border-slate-800">
        {(
          [
            ['results', 'Results'],
            ['log', 'Log'],
            ['cli', 'Reproduce from the CLI'],
          ] as const
        ).map(([k, label]) => (
          <button
            key={k}
            onClick={() => setTab(k)}
            className={cx(
              '-mb-px border-b-2 px-4 py-2 text-sm font-medium',
              tab === k ? 'border-sky-400 text-slate-100' : 'border-transparent text-slate-400 hover:text-slate-200',
            )}
          >
            {label}
          </button>
        ))}
      </div>

      {tab === 'results' && (
        <div className="grid gap-6 xl:grid-cols-[1fr_360px]">
          <Card className="p-5">
            {active ? (
              <div className="grid aspect-[3/2] place-items-center text-slate-500">
                <div className="flex items-center gap-3">
                  <Spinner /> Simulating…
                </div>
              </div>
            ) : (
              <ImageViewer images={images} />
            )}
          </Card>
          <div className="space-y-6">
            <Card className="p-5">
              <SectionTitle title="Settings" />
              <dl className="space-y-1.5 text-sm">
                {[
                  ['Scene', req.scene_id],
                  ['Camera', req.camera_id],
                  ['Method', req.mode === 'fast_analytic' ? 'Instant (analytic)' : `PBRT ${req.xres}×${req.yres}, ${req.pixelsamples} spp`],
                  ['Light', `${req.illuminant_id ?? 'scene'} · ${req.target_illuminance_lux.toLocaleString()} lux`],
                  ['Exposure', req.exposure_time_s === null ? 'recipe default' : `${+(req.exposure_time_s * 1000).toPrecision(4)} ms`],
                  ['Seed', String(req.seed)],
                ].map(([k, v]) => (
                  <div key={k} className="flex justify-between gap-3">
                    <dt className="text-slate-500">{k}</dt>
                    <dd className="text-right font-mono text-xs text-slate-200">{v}</dd>
                  </div>
                ))}
              </dl>
            </Card>
            {!active && (
              <Card className="p-5">
                <SectionTitle title="Downloads" />
                <div className="space-y-1">
                  {DOWNLOADS.filter(([p]) => run.outputs.includes(p)).map(([p, label]) => (
                    <a key={p} href={fileUrl(run.id, `outputs/${p}`)} download className="flex items-center justify-between rounded-lg px-2 py-1.5 text-sm text-slate-300 hover:bg-slate-800">
                      <span className="flex items-center gap-2">
                        <FileDown size={16} className="text-slate-500" /> {label}
                      </span>
                      <span className="font-mono text-xs text-slate-500">{p}</span>
                    </a>
                  ))}
                  <a href={fileUrl(run.id, 'request.json')} download className="flex items-center justify-between rounded-lg px-2 py-1.5 text-sm text-slate-300 hover:bg-slate-800">
                    <span className="flex items-center gap-2">
                      <Download size={16} className="text-slate-500" /> Settings (request.json)
                    </span>
                  </a>
                </div>
              </Card>
            )}
          </div>
          {statsPath && (
            <Card className="p-5 xl:col-span-2">
              <SectionTitle title="Sensor statistics" subtitle="From apply_emva_noise.py (run_stats.json)." />
              <StatsPanel runId={run.id} path={`outputs/${statsPath}`} />
            </Card>
          )}
        </div>
      )}

      {tab === 'log' && <LogPanel id={run.id} active={active} />}

      {tab === 'cli' && (
        <div className="space-y-4">
          <p className="text-sm text-slate-400">
            These are the exact commands this run executed. Run <span className="font-mono text-slate-300">bash run.sh</span>, or repeat the whole run (saved in history like a GUI run) with{' '}
            <span className="font-mono text-slate-300">opencam-studio run request.json</span>. The files are in{' '}
            <span className="font-mono text-slate-300">out/studio_runs/{run.id}/</span>.
          </p>
          {script ? <CommandBlock script={script} /> : <p className="text-sm text-slate-500">The script is written when the run starts.</p>}
        </div>
      )}
    </div>
  )
}
