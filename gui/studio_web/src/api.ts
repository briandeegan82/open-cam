// Typed client for the Open Cam Studio FastAPI backend (gui/opencam_gui/studio/server.py).

export type Mode = 'fast_analytic' | 'pbrt_accurate'

export interface Status {
  pbrt_available: boolean
  cpu_count: number
  repo_root: string
  runs_dir: string
}

export interface Scene {
  id: string
  label: string
  kind: 'colorchecker' | 'iq_target'
  description: string
  modes: Mode[]
  supports_illuminant: boolean
}

export interface Camera {
  id: string
  display_name: string
  family: string
  sensor_class?: string | null
  calibration_tier?: string | null
  cfa?: string
  demosaic?: string
  pixel_pitch_um?: number
  f_number?: number
  integration_time_s?: number
  lens_camera?: string
  focal_length_mm?: number | null
  bit_depth?: number
  full_well_e?: number
  gain_e_per_dn?: number
  read_noise_e?: number
  hdr?: boolean
  ircf?: boolean
  path?: string
  error?: string
}

export interface Illuminant {
  id: string
  label: string
  path: string
}

export interface Spectrum {
  id: string
  wavelength_nm: number[]
  value: number[]
}

export interface QualityPreset {
  id: string
  label: string
  description: string
  xres: number
  yres: number
  pixelsamples: number
}

export interface RunRequest {
  scene_id: string
  camera_id: string
  mode: Mode
  illuminant_id: string | null
  target_illuminance_lux: number
  exposure_time_s: number | null
  seed: number
  xres: number
  yres: number
  pixelsamples: number
  name: string
}

export interface Step {
  name: string
  argv: string[]
  returncode: number | null
  started: number | null
  finished: number | null
}

export type RunStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled' | 'interrupted'

export interface Run {
  id: string
  request: RunRequest
  status: RunStatus
  created: number
  started: number | null
  finished: number | null
  error: string | null
  steps: Step[]
  outputs: string[]
  queue_position: number | null
  estimate_s: number
}

export interface Plan {
  steps: { name: string; argv: string[] }[]
  script: string
  estimate_s: number
}

export interface LearnTopic {
  id: string
  title: string
  command: string
  tutorial: string | null
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: init?.body ? { 'Content-Type': 'application/json' } : undefined,
    ...init,
  })
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* not JSON */
    }
    throw new Error(detail)
  }
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T)
}

export const api = {
  status: () => request<Status>('/api/status'),
  scenes: () => request<Scene[]>('/api/scenes'),
  cameras: () => request<Camera[]>('/api/cameras'),
  illuminants: () => request<Illuminant[]>('/api/illuminants'),
  spectrum: (id: string) => request<Spectrum>(`/api/illuminants/${encodeURIComponent(id)}/spectrum`),
  presets: () => request<QualityPreset[]>('/api/quality-presets'),
  plan: (req: RunRequest) => request<Plan>('/api/plan', { method: 'POST', body: JSON.stringify(req) }),
  runs: () => request<Run[]>('/api/runs'),
  run: (id: string) => request<Run>(`/api/runs/${encodeURIComponent(id)}`),
  createRun: (req: RunRequest) => request<Run>('/api/runs', { method: 'POST', body: JSON.stringify(req) }),
  cancelRun: (id: string) => request<Run>(`/api/runs/${encodeURIComponent(id)}/cancel`, { method: 'POST' }),
  deleteRun: (id: string) => request<void>(`/api/runs/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  log: (id: string, offset: number) =>
    request<{ text: string; offset: number; status: RunStatus }>(
      `/api/runs/${encodeURIComponent(id)}/log?offset=${offset}`,
    ),
  learn: () => request<LearnTopic[]>('/api/learn'),
  launchDemo: (id: string) => request<{ launched: string }>(`/api/learn/${id}/launch`, { method: 'POST' }),
}

export const fileUrl = (runId: string, path: string) =>
  `/api/runs/${encodeURIComponent(runId)}/files/${path.split('/').map(encodeURIComponent).join('/')}`

export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds)) return '–'
  if (seconds < 60) return `${Math.max(1, Math.round(seconds))} s`
  if (seconds < 3600) return `${Math.round(seconds / 60)} min`
  return `${(seconds / 3600).toFixed(1)} h`
}

export function formatTime(epoch: number): string {
  return new Date(epoch * 1000).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}
