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
`tutorials/` in this directory for the full explanation of what each demo
reuses.

## Learning path

The seven demos form one graduate-course arc. Geometry forms the image, optics
blurs it, MTF measures the blur, the sensor adds noise, exposure and defects
distort the electrons, the ISP turns them into colour, and the last demo runs
the whole chain on a real scene.

| # | Topic | Demo | Tutorial |
| - | --- | --- | --- |
| 1 | Fundamental optics / imaging geometry | `opencam-gui demo geometry` | [tutorials/01-fundamental-optics.md](tutorials/01-fundamental-optics.md) |
| 2 | PSF, diffraction and aberrations | `opencam-gui demo optics` | [tutorials/02-psf-aberrations.md](tutorials/02-psf-aberrations.md) |
| 3 | Resolution and MTF | `opencam-gui demo mtf` | [tutorials/03-resolution-mtf.md](tutorials/03-resolution-mtf.md) |
| 4 | Sensor noise (EMVA1288) | `opencam-gui demo sensor` | [tutorials/04-sensor-noise.md](tutorials/04-sensor-noise.md) |
| 5 | Exposure and sensor defects | `opencam-gui demo exposure` | [tutorials/05-exposure-defects.md](tutorials/05-exposure-defects.md) |
| 6 | Colour and the ISP | `opencam-gui demo isp` | [tutorials/06-colour-isp.md](tutorials/06-colour-isp.md) |
| 7 | Image generation (scene to image) | `opencam-gui demo image-generation` | [tutorials/07-image-generation.md](tutorials/07-image-generation.md) |
| 8 | Image-quality lab (test scenes and metrics) | `opencam-gui demo iq-lab` | [tutorials/08-iq-lab.md](tutorials/08-iq-lab.md) |

`opencam-gui topics` prints the same list from the code. Each tutorial ends
with a bridge section explaining why the next one builds on it. A suggested
twelve-week lecture mapping lives in [tutorials/README.md](tutorials/README.md).

## Quick start

```bash
cd ~/open-cam
python3 -m venv venv                     # skip if you already have one
venv/bin/pip install -r requirements.txt # core pipeline deps
venv/bin/pip install -e gui[dev]         # GUI-only deps, same venv

venv/bin/python -m opencam_gui.ui.cli topics
venv/bin/python -m opencam_gui.ui.cli demo geometry
venv/bin/python -m opencam_gui.ui.cli demo mtf --list-scenarios
venv/bin/python -m opencam_gui.ui.cli demo isp --scenario tungsten

# or, after the editable install:
opencam-gui demo geometry --scenario phone_wide
opencam-gui demo optics --list-scenarios
```

Only Tutorial 07's "Physically accurate (PBRT render)" mode and Tutorial 08 (IQ lab)
need a PBRT binary from `../docs/BUILD_PBRT.txt`. Everything else, including Tutorial 03's
synthetic edge source and Tutorial 07's analytic ColorChecker path, runs on
NumPy alone. A **Dry run** is always available in the image-generation demo.

## What's in here

```
gui/
  opencam_gui/
    core/          # read-only adapters over tools/*.py -- no camera physics of its own
      repo.py              # locates the repo root, puts tools/ on sys.path
      catalog.py           # lists camera recipes / illuminants from config/ and spectra/
      camera.py            # loads a camera model, extracts optics/EMVA summaries
      dark_current.py      # documented mirror of apply_emva_noise.py's dark-current formula
      geometry_engine.py   # adapter over tools/imaging_geometry.py
      optics_engine.py     # adapter over tools/apply_spectral_psf.py
      mtf_engine.py        # adapter over tools/sfr_analysis.py
      sensor_engine.py     # adapter over tools/emva_theory.py
      exposure_engine.py   # adapter over tools/sensor_radiometry.py and apply_emva_noise.py
      isp_engine.py        # adapter over tools/colour_science.py and apply_emva_noise.py
      pipeline.py          # builds and runs the real tools/*.py subprocess command plans
    topics/        # scripted lecture scenarios (dataclasses), one package per demo
    ui/
      cli.py           # `opencam-gui demo <topic>` -- demos are a TOPICS registry
      desktop/         # DemoApp base plus one subclass per topic
        base.py
  tutorials/       # markdown tutorials + index
  tests/           # pytest tests for opencam_gui/core (no display required)
```

Adding an eighth demo is one entry in `TOPICS` in `opencam_gui/ui/cli.py`, a
`topics/<id>/scenarios.py`, and a `ui/desktop/<id>_app.py` subclassing
`DemoApp`. The physics goes in `tools/`; the `core/` adapter only reshapes
what `tools/` returns.

## Running the tests

```bash
cd ~/open-cam
venv/bin/pip install -e gui[dev]
venv/bin/python -m pytest gui/tests -v
```

The tests exercise `opencam_gui/core/` (pure computation and subprocess
command-building) and build the Dear PyGui widget trees headlessly -- they
don't need a display and don't invoke PBRT.
