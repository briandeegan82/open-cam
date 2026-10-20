import { Check, Copy, Download } from 'lucide-react'
import { useState } from 'react'
import { Button } from './ui'

export function CommandBlock({ script, filename = 'run.sh' }: { script: string; filename?: string }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(script)
    } catch {
      const ta = document.createElement('textarea')
      ta.value = script
      document.body.appendChild(ta)
      ta.select()
      document.execCommand('copy')
      ta.remove()
    }
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }
  const download = () => {
    const url = URL.createObjectURL(new Blob([script], { type: 'text/x-shellscript' }))
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
  }
  return (
    <div className="overflow-hidden rounded-xl border border-slate-800 bg-slate-950">
      <div className="flex items-center justify-between border-b border-slate-800 px-3 py-1.5">
        <span className="font-mono text-xs text-slate-500">{filename}</span>
        <div className="flex gap-1">
          <Button variant="ghost" className="px-2 py-1 text-xs" onClick={copy}>
            {copied ? <Check size={14} /> : <Copy size={14} />} {copied ? 'Copied' : 'Copy'}
          </Button>
          <Button variant="ghost" className="px-2 py-1 text-xs" onClick={download}>
            <Download size={14} /> Download
          </Button>
        </div>
      </div>
      <pre className="max-h-80 overflow-auto p-3 font-mono text-xs leading-relaxed whitespace-pre-wrap break-all text-slate-300">
        {script}
      </pre>
    </div>
  )
}
