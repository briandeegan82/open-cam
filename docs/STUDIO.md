# Open Cam Studio

Open Cam Studio is a browser interface to the Open Cam pipeline for people who
would rather not write YAML or shell commands. It runs locally. Pick a scene,
a camera and a quality level, press **Run simulation**, and inspect the clean
and noisy images, the sensor statistics and the downloadable RAW/EXR/NPZ files.

Studio is only a front end. Every run executes the same `tools/*.py` scripts as
the CLI (through `gui/opencam_gui/core/pipeline.py`), and every run saves the
exact shell commands it used, so you can repeat or script any GUI run.

## Start it

```bash
venv/bin/pip install -r requirements.txt
venv/bin/pip install -e gui            # adds fastapi + uvicorn and the opencam-studio command
venv/bin/opencam-studio                # serves http://127.0.0.1:8765 and opens your browser
```

Other forms: `opencam-gui studio`, `opencam-studio serve --port 9000 --no-browser`,
or `PYTHONPATH=gui venv/bin/python -m opencam_gui.studio`.

The built web bundle is committed in `gui/opencam_gui/studio/static/`, so you
do **not** need Node.js to use Studio.

On a headless render machine, keep the default localhost bind and use an SSH
tunnel: `ssh -L 8765:127.0.0.1:8765 rendering-box`, then browse to
`http://127.0.0.1:8765`. Studio has no login and it starts pipeline jobs, so
it warns if you bind it to a non-local address with `--host`.

## What Phase 1 covers

| Scene | Instant (analytic) | Physically accurate (PBRT) |
| --- | --- | --- |
| ColorChecker | yes (seconds, no lens blur) | yes (`tools/run_pipeline.py`) |
| Slanted edge, ISO noise chart, Siemens star | no | yes |

- **Cameras:** all `config/camera_recipes/*.yaml`, searchable, with pitch, f-number, lens, CFA, ADC, full well, gain and read noise.
- **Lighting:** every illuminant in `spectra/illuminant/interpolated/` with its spectrum, an illuminance slider (1 lux to 100 klux), and a recipe or manual exposure time.
- **Quality:** Preview (320×240, 32 spp), Standard (480×320, 128 spp) and Final (960×640, 1024 spp), or custom values, with a rough time estimate.
- **Results:** zoom/pan viewer with an A/B split slider, sensor statistics from `run_stats.json`, file downloads, the live log, and **Edit and re-run**.
- **Learn:** opens the eight Dear PyGui teaching demos and links their tutorials.

Highway, urban, tunnel, cabin/DMS and the other scene builders are planned
for the next phase (forms generated from each tool's argparse options).
Until then they are available from the CLI.

## Runs and scripting

Each run gets its own folder, by default `out/studio_runs/` (override it with
`--runs-dir` or `OPENCAM_STUDIO_RUNS`):

```
out/studio_runs/20261020-125542-office-d65-iphone-8/
  request.json    # the settings from the form
  run.json        # status, timings, per-step return codes, output list
  run.sh          # the exact commands that were executed
  pipeline.yaml   # PBRT ColorChecker runs only
  log.txt         # full tool output
  outputs/        # copies of the previews, RAW16, electrons NPZ, spectral EXR
```

The tools write fixed paths under `out/`, so Studio runs one job at a time
and copies each run's outputs into its folder before it starts the next one.
Leave the GUI out of it entirely:

```bash
bash out/studio_runs/<run>/run.sh                     # repeat the raw commands
venv/bin/opencam-studio plan request.json             # print the commands for a settings file
venv/bin/opencam-studio run request.json              # run it headless, recorded in run history
```

A minimal `request.json` is `{"scene_id": "colorchecker", "camera_id": "iphone_8"}`.
The other fields are `mode` (`fast_analytic` | `pbrt_accurate`), `illuminant_id`,
`target_illuminance_lux`, `exposure_time_s`, `seed`, `xres`, `yres`,
`pixelsamples` and `name`. The REST API that the web page uses is documented at
`http://127.0.0.1:8765/docs`.

## Developing the frontend

The UI is React + TypeScript + Tailwind, built with Vite, in `gui/studio_web/`:

```bash
cd gui/studio_web
npm ci
npm run dev        # http://localhost:5173 with hot reload; proxies /api to a running opencam-studio --no-browser
npm run lint
npm run build      # rewrites gui/opencam_gui/studio/static/ -- commit the result
```

CI rebuilds the bundle and fails if the committed `static/` folder is out of date.
