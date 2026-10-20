import { Maximize2, Minus, Plus } from 'lucide-react'
import { useCallback, useRef, useState, type PointerEvent, type WheelEvent } from 'react'
import { Button, Segmented, cx } from './ui'

export interface ViewerImage {
  key: string
  label: string
  url: string
}

// Zoom (wheel / buttons) + pan (drag) image viewer with an optional A/B split comparison.
// Zoomed pixels are drawn nearest-neighbour so sensor noise and CFA structure stay visible.
export function ImageViewer({ images }: { images: ViewerImage[] }) {
  const [mode, setMode] = useState<'single' | 'compare'>('single')
  const [a, setA] = useState(images[0]?.key ?? '')
  const [b, setB] = useState(images[1]?.key ?? images[0]?.key ?? '')
  const [split, setSplit] = useState(50)
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const drag = useRef<{ x: number; y: number; px: number; py: number } | null>(null)
  const box = useRef<HTMLDivElement>(null)

  const imgA = images.find((i) => i.key === a) ?? images[0]
  const imgB = images.find((i) => i.key === b) ?? images[0]

  const reset = useCallback(() => {
    setZoom(1)
    setPan({ x: 0, y: 0 })
  }, [])
  const zoomBy = (f: number) => setZoom((z) => Math.min(32, Math.max(1, z * f)))

  const onWheel = (e: WheelEvent) => {
    e.preventDefault()
    zoomBy(e.deltaY < 0 ? 1.25 : 0.8)
  }
  const onDown = (e: PointerEvent) => {
    if ((e.target as HTMLElement).dataset.handle) return
    ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
    drag.current = { x: e.clientX, y: e.clientY, px: pan.x, py: pan.y }
  }
  const onMove = (e: PointerEvent) => {
    if (!drag.current) return
    setPan({ x: drag.current.px + (e.clientX - drag.current.x) / zoom, y: drag.current.py + (e.clientY - drag.current.y) / zoom })
  }
  const onUp = () => (drag.current = null)

  if (!imgA) return <p className="text-sm text-slate-500">No preview images were produced.</p>

  const transform = { transform: `scale(${zoom}) translate(${pan.x}px, ${pan.y}px)` }
  const imgClass = cx('absolute inset-0 h-full w-full object-contain select-none', zoom > 2 && 'pixelated')
  const select = (value: string, set: (v: string) => void) => (
    <select
      className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-200"
      value={value}
      onChange={(e) => set(e.target.value)}
    >
      {images.map((i) => (
        <option key={i.key} value={i.key}>
          {i.label}
        </option>
      ))}
    </select>
  )

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <Segmented
          value={mode}
          onChange={setMode}
          options={[
            { value: 'single', label: 'Single' },
            { value: 'compare', label: 'Compare A/B', disabled: images.length < 2 },
          ]}
        />
        {mode === 'single' ? (
          <div className="flex flex-wrap gap-1">
            {images.map((i) => (
              <button
                key={i.key}
                onClick={() => setA(i.key)}
                className={cx(
                  'rounded-md px-2.5 py-1 text-xs font-medium',
                  i.key === imgA.key ? 'bg-sky-500/15 text-sky-200 ring-1 ring-sky-400/40' : 'text-slate-400 hover:text-slate-200',
                )}
              >
                {i.label}
              </button>
            ))}
          </div>
        ) : (
          <div className="flex items-center gap-2 text-xs text-slate-400">
            A {select(a, setA)} B {select(b, setB)}
          </div>
        )}
        <div className="ml-auto flex items-center gap-1">
          <Button variant="ghost" className="p-1.5" onClick={() => zoomBy(0.8)} title="Zoom out">
            <Minus size={16} />
          </Button>
          <span className="w-12 text-center font-mono text-xs text-slate-400">{zoom.toFixed(1)}×</span>
          <Button variant="ghost" className="p-1.5" onClick={() => zoomBy(1.25)} title="Zoom in">
            <Plus size={16} />
          </Button>
          <Button variant="ghost" className="p-1.5" onClick={reset} title="Fit">
            <Maximize2 size={16} />
          </Button>
        </div>
      </div>
      <div
        ref={box}
        className="relative aspect-[3/2] w-full cursor-grab touch-none overflow-hidden rounded-xl bg-[repeating-conic-gradient(#0f172a_0_25%,#111827_0_50%)] bg-[length:16px_16px] ring-1 ring-slate-800 active:cursor-grabbing"
        onWheel={onWheel}
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onDoubleClick={reset}
      >
        <div className="absolute inset-0 origin-center" style={transform}>
          <img src={(mode === 'compare' ? imgB : imgA).url} className={imgClass} draggable={false} alt="" />
        </div>
        {mode === 'compare' && (
          <>
            <div className="absolute inset-0" style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }}>
              <div className="absolute inset-0 origin-center" style={transform}>
                <img src={imgA.url} className={imgClass} draggable={false} alt="" />
              </div>
            </div>
            <div className="pointer-events-none absolute inset-y-0 w-0.5 bg-white/80" style={{ left: `${split}%` }} />
            <input
              data-handle="1"
              type="range"
              min={0}
              max={100}
              value={split}
              onChange={(e) => setSplit(Number(e.target.value))}
              className="absolute inset-x-4 bottom-3 z-10"
              aria-label="A/B split"
            />
            <span className="absolute top-2 left-2 rounded bg-black/60 px-1.5 py-0.5 text-xs text-white">A · {imgA.label}</span>
            <span className="absolute top-2 right-2 rounded bg-black/60 px-1.5 py-0.5 text-xs text-white">B · {imgB.label}</span>
          </>
        )}
      </div>
      <p className="text-xs text-slate-500">Scroll to zoom, drag to pan, double-click to fit. Previews are tone-mapped for display.</p>
    </div>
  )
}
