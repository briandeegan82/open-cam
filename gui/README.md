# Open Cam GUI & Tutorials

Interactive Dear PyGui demos and a guided learning path for the Open Cam
camera-simulation pipeline, in the same style as the `teaching-sims` project
(a sibling repo, if you have it checked out alongside this one): a
UI-agnostic physics/tooling core, a thin desktop front end, scripted lecture
scenarios, and a markdown tutorial per topic.

This package is **only an interface**. It does not reimplement any camera
physics or pipeline logic -- every plotted curve is produced by importing the
real functions from `../tools/*.py`, and every generated image comes from
running the real `tools/*.py` scripts (including `tools/run_pipeline.py`) as
subprocesses. See `opencam_gui/core/` for the thin adapter layer and
`../tutorials/` in this directory for the full explanation of what each demo
reuses.

## Learning path

| # | Topic | Demo | Tutorial |
| - | --- | --- | --- |
| 1 | Optics / point-spread function | `opencam-gui demo optics` | [tutorials/01-optics.md](tutorials/01-optics.md) |
| 2 | Sensor modelling (EMVA1288 PTC, DSNU, PRNU) | `opencam-gui demo sensor` | [tutorials/02-sensor-modeling.md](tutorials/02-sensor-modeling.md) |
| 3 | Image generation (scene -> sensor -> image) | `opencam-gui demo image-generation` | [tutorials/03-image-generation.md](tutorials/03-image-generation.md) |

Start with Optics, then Sensor Modelling, then Image Generation -- each
tutorial ends with a "bridge" section explaining why the next one builds on it.

## Quick start

```bash
cd ~/open-cam
python3 -m venv venv                     # skip if you already have one
venv/bin/pip install -r requirements.txt # core pipeline deps
venv/bin/pip install -e gui[dev]         # GUI-only deps, same venv

venv/bin/python -m opencam_gui.ui.cli demo optics
venv/bin/python -m opencam_gui.ui.cli demo sensor
venv/bin/python -m opencam_gui.ui.cli demo image-generation

# or, after the editable install:
opencam-gui demo optics --scenario uncorrected_lateral_ca
opencam-gui demo optics --list-scenarios
```

The Image Generation demo's "Fast preview (analytic, no PBRT)" mode works
without building PBRT at all; "Physically accurate (PBRT render)" needs the
binary from `../docs/BUILD_PBRT.txt` but always supports a **Dry run** either
way.

## What's in here

```
gui/
  opencam_gui/
    core/          # read-only adapters over tools/*.py -- no camera physics of its own
      repo.py          # locates the repo root, puts tools/ on sys.path
      catalog.py       # lists camera recipes / illuminants from config/ and spectra/
      camera.py        # loads a camera model, extracts optics/EMVA summaries
      dark_current.py  # documented mirror of apply_emva_noise.py's dark-current formula
      optics_engine.py # adapter over tools/apply_spectral_psf.py
      sensor_engine.py # adapter over tools/emva_theory.py
      pipeline.py      # builds and runs the real tools/*.py subprocess command plans
    topics/        # scripted lecture scenarios (dataclasses), one package per demo
    ui/
      cli.py           # `opencam-gui demo <topic>`
      desktop/         # the three Dear PyGui apps
  tutorials/       # markdown tutorials + index
  tests/           # pytest tests for opencam_gui/core (no display required)
```

## Running the tests

```bash
cd ~/open-cam
venv/bin/pip install -e gui[dev]
venv/bin/python -m pytest gui/tests -v
```

The tests only exercise `opencam_gui/core/` (pure computation and subprocess
command-building) -- they don't need a display and don't invoke PBRT.
