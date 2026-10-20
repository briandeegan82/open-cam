import { BookOpen, ExternalLink, MonitorPlay } from 'lucide-react'
import { useEffect, useState } from 'react'
import { api, type LearnTopic } from '../api'
import { Button, Card } from '../components/ui'

const REPO = 'https://github.com/briandeegan82/open-cam/blob/main/gui/tutorials'

export function Learn() {
  const [topics, setTopics] = useState<LearnTopic[]>([])
  const [launched, setLaunched] = useState<string | null>(null)
  useEffect(() => {
    api.learn().then(setTopics).catch(() => {})
  }, [])
  return (
    <div className="space-y-4">
      <p className="max-w-3xl text-sm text-slate-400">
        Eight interactive lessons, from lens geometry to a full scene-to-image run. Each opens as a desktop window on the computer running Studio and comes with a written tutorial.
      </p>
      <div className="grid gap-4 md:grid-cols-2">
        {topics.map((t) => {
          const [num, ...rest] = t.title.split(' - ')
          return (
            <Card key={t.id} className="flex items-start gap-4 p-5">
              <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-amber-500/10 font-mono text-sm font-semibold text-amber-300 ring-1 ring-amber-400/30">{num}</span>
              <div className="min-w-0 flex-1">
                <h3 className="font-medium text-slate-100">{rest.join(' - ')}</h3>
                <p className="mt-1 font-mono text-xs text-slate-500">{t.command}</p>
                <div className="mt-3 flex flex-wrap gap-2">
                  <Button
                    onClick={async () => {
                      await api.launchDemo(t.id)
                      setLaunched(t.id)
                      setTimeout(() => setLaunched(null), 4000)
                    }}
                  >
                    <MonitorPlay size={16} /> {launched === t.id ? 'Opening…' : 'Open demo'}
                  </Button>
                  {t.tutorial && (
                    <a href={`${REPO}/${t.tutorial}`} target="_blank" rel="noreferrer">
                      <Button variant="ghost">
                        <BookOpen size={16} /> Tutorial <ExternalLink size={12} />
                      </Button>
                    </a>
                  )}
                </div>
              </div>
            </Card>
          )
        })}
      </div>
    </div>
  )
}
