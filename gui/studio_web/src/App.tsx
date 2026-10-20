import { Aperture, GraduationCap, History as HistoryIcon, PlusCircle } from 'lucide-react'
import { useEffect, useState, type ReactNode } from 'react'
import { api, type Status } from './api'
import { Badge, cx } from './components/ui'
import { History } from './pages/History'
import { Learn } from './pages/Learn'
import { NewRun } from './pages/NewRun'
import { RunPage } from './pages/RunPage'
import { navigate, useRoute } from './router'

function NavItem({ to, icon, label, active }: { to: string; icon: ReactNode; label: string; active: boolean }) {
  return (
    <button
      onClick={() => navigate(to)}
      className={cx(
        'flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition',
        active ? 'bg-slate-800 text-white' : 'text-slate-400 hover:bg-slate-900 hover:text-slate-200',
      )}
    >
      {icon} {label}
    </button>
  )
}

export default function App() {
  const route = useRoute()
  const [status, setStatus] = useState<Status | null>(null)
  useEffect(() => {
    api.status().then(setStatus).catch(() => {})
  }, [])

  const section = route.path[0] ?? 'new'
  let page: ReactNode
  let title: string
  if (section === 'runs' && route.path[1]) {
    page = <RunPage key={route.path[1]} id={route.path[1]} />
    title = 'Run'
  } else if (section === 'runs') {
    page = <History />
    title = 'Run history'
  } else if (section === 'learn') {
    page = <Learn />
    title = 'Learn camera imaging'
  } else {
    page = <NewRun status={status} fromRun={route.query.get('from')} />
    title = 'New simulation'
  }

  return (
    <div className="flex h-full">
      <aside className="hidden w-60 shrink-0 flex-col border-r border-slate-800/80 bg-slate-950 p-4 md:flex">
        <div className="mb-8 flex items-center gap-2.5 px-2">
          <span className="grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-sky-500 to-cyan-400 text-slate-950 shadow-lg shadow-sky-500/20">
            <Aperture size={20} />
          </span>
          <div>
            <div className="text-sm leading-tight font-semibold text-white">Open Cam</div>
            <div className="text-xs leading-tight text-slate-500">Studio</div>
          </div>
        </div>
        <nav className="space-y-1">
          <NavItem to="/new" icon={<PlusCircle size={18} />} label="New simulation" active={section === 'new'} />
          <NavItem to="/runs" icon={<HistoryIcon size={18} />} label="Run history" active={section === 'runs'} />
          <NavItem to="/learn" icon={<GraduationCap size={18} />} label="Learn" active={section === 'learn'} />
        </nav>
        <div className="mt-auto space-y-2 px-2 text-xs text-slate-500">
          {status && (
            <>
              <div className="flex items-center justify-between">
                <span>PBRT renderer</span>
                <Badge tone={status.pbrt_available ? 'emerald' : 'amber'}>{status.pbrt_available ? 'ready' : 'not built'}</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span>CPU cores</span>
                <span className="text-slate-300">{status.cpu_count}</span>
              </div>
            </>
          )}
          <a className="block pt-2 hover:text-slate-300" href="/docs" target="_blank" rel="noreferrer">
            REST API docs
          </a>
        </div>
      </aside>
      <main className="min-w-0 flex-1 overflow-y-auto">
        <header className="sticky top-0 z-20 flex items-center gap-3 border-b border-slate-800/80 bg-slate-950/80 px-6 py-4 backdrop-blur">
          <h1 className="text-lg font-semibold text-white">{title}</h1>
          <nav className="ml-auto flex gap-1 md:hidden">
            <button className="rounded px-2 py-1 text-sm text-slate-300" onClick={() => navigate('/new')}>New</button>
            <button className="rounded px-2 py-1 text-sm text-slate-300" onClick={() => navigate('/runs')}>History</button>
            <button className="rounded px-2 py-1 text-sm text-slate-300" onClick={() => navigate('/learn')}>Learn</button>
          </nav>
        </header>
        <div className="mx-auto max-w-[1500px] p-6">{page}</div>
      </main>
    </div>
  )
}
