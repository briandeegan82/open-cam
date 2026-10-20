import { ImageOff, Plus } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api, fileUrl, formatDuration, formatTime, type Run } from '../api'
import { Button, Card, ErrorNote, Spinner, StatusBadge } from '../components/ui'
import { navigate } from '../router'

export function History() {
  const [runs, setRuns] = useState<Run[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    let live = true
    let timer: number | undefined
    const tick = () =>
      api
        .runs()
        .then((r) => {
          if (!live) return
          setRuns(r)
          timer = window.setTimeout(tick, r.some((x) => x.status === 'queued' || x.status === 'running') ? 1500 : 10000)
        })
        .catch((e) => live && setError(String(e.message ?? e)))
    tick()
    return () => {
      live = false
      window.clearTimeout(timer)
    }
  }, [])
  if (error) return <ErrorNote>{error}</ErrorNote>
  if (!runs) return <Spinner />
  if (!runs.length)
    return (
      <Card className="grid place-items-center gap-3 p-12 text-center">
        <p className="text-slate-400">No runs yet.</p>
        <Button variant="primary" onClick={() => navigate('/new')}>
          <Plus size={16} /> New simulation
        </Button>
      </Card>
    )
  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-4">
      {runs.map((r) => {
        const thumb = r.outputs.includes('preview/noisy_demosaic_rgb8.png')
        return (
          <button key={r.id} onClick={() => navigate(`/runs/${r.id}`)} className="text-left">
            <Card className="overflow-hidden transition hover:border-slate-600">
              <div className="grid aspect-[3/2] place-items-center bg-slate-950">
                {thumb ? (
                  <img src={fileUrl(r.id, 'outputs/preview/noisy_demosaic_rgb8.png')} className="h-full w-full object-contain" alt="" loading="lazy" />
                ) : r.status === 'running' || r.status === 'queued' ? (
                  <Spinner />
                ) : (
                  <ImageOff className="text-slate-700" />
                )}
              </div>
              <div className="space-y-1 p-3">
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate text-sm font-medium text-slate-100">{r.request.name || `${r.request.scene_id} · ${r.request.camera_id}`}</span>
                  <StatusBadge status={r.status} />
                </div>
                <div className="text-xs text-slate-500">
                  {formatTime(r.created)}
                  {r.started && r.finished ? ` · ${formatDuration(r.finished - r.started)}` : ''} ·{' '}
                  {r.request.mode === 'fast_analytic' ? 'instant' : `PBRT ${r.request.pixelsamples} spp`}
                </div>
              </div>
            </Card>
          </button>
        )
      })}
    </div>
  )
}
