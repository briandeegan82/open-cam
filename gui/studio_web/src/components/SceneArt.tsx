// Procedural thumbnails for the scene gallery (drawn, not rendered).

const MACBETH = [
  '#735244', '#c29682', '#627a9d', '#576c43', '#8580b1', '#67bdaa',
  '#d67e2c', '#505ba6', '#c15a63', '#5e3c6c', '#9dbc40', '#e0a32e',
  '#383d96', '#469449', '#af363c', '#e7c71f', '#bb5695', '#0885a1',
  '#f3f3f2', '#c8c8c8', '#a0a0a0', '#7a7a79', '#555555', '#343434',
]

export function SceneArt({ id }: { id: string }) {
  const common = 'h-full w-full'
  if (id === 'colorchecker') {
    return (
      <svg viewBox="0 0 120 80" className={common}>
        <rect width="120" height="80" fill="#1e1e1e" />
        {MACBETH.map((c, i) => (
          <rect key={i} x={6 + (i % 6) * 18.5} y={6 + Math.floor(i / 6) * 17.5} width="16" height="15" rx="1.5" fill={c} />
        ))}
      </svg>
    )
  }
  if (id === 'slanted_edge') {
    return (
      <svg viewBox="0 0 120 80" className={common}>
        <rect width="120" height="80" fill="#d4d4d4" />
        <polygon points="56,0 120,0 120,80 64,80" fill="#262626" />
      </svg>
    )
  }
  if (id === 'iso_noise') {
    return (
      <svg viewBox="0 0 120 80" className={common}>
        <rect width="120" height="80" fill="#525252" />
        {Array.from({ length: 12 }, (_, i) => {
          const v = Math.round(20 + (i * 225) / 11)
          return (
            <rect key={i} x={8 + (i % 6) * 18} y={18 + Math.floor(i / 6) * 24} width="14" height="18" rx="1" fill={`rgb(${v},${v},${v})`} />
          )
        })}
      </svg>
    )
  }
  if (id === 'siemens_star') {
    const n = 18
    return (
      <svg viewBox="0 0 120 80" className={common}>
        <rect width="120" height="80" fill="#e5e5e5" />
        {Array.from({ length: n }, (_, i) => {
          const a0 = (i / n) * 2 * Math.PI
          const a1 = ((i + 0.5) / n) * 2 * Math.PI
          const r = 34
          return (
            <polygon
              key={i}
              points={`60,40 ${60 + r * Math.cos(a0)},${40 + r * Math.sin(a0)} ${60 + r * Math.cos(a1)},${40 + r * Math.sin(a1)}`}
              fill="#171717"
            />
          )
        })}
      </svg>
    )
  }
  return (
    <svg viewBox="0 0 120 80" className={common}>
      <rect width="120" height="80" fill="#334155" />
    </svg>
  )
}
