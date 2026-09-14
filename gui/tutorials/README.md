# Tutorials

Hands-on guides for the Open Cam interactive demos. Each tutorial explains the
underlying principles, maps them to the UI, and walks through experiments you
can run in lecture or self-paced lab -- exactly in the style of
[teaching-sims](../../../teaching-sims/tutorials/README.md), applied to the
Open Cam camera-simulation pipeline instead of radar.

| Tutorial | Demo command | Time (approx.) |
| --- | --- | --- |
| [01 -- Optics / PSF](01-optics.md) | `opencam-gui demo optics` | 40-55 min |
| [02 -- Sensor Modelling (EMVA1288)](02-sensor-modeling.md) | `opencam-gui demo sensor` | 40-55 min |
| [03 -- Image Generation](03-image-generation.md) | `opencam-gui demo image-generation` | 45-60 min |

## Before you start

```bash
cd ~/open-cam
python3 -m venv venv                     # if you don't already have one
venv/bin/pip install -r requirements.txt # core pipeline deps (numpy, OpenEXR, ...)
venv/bin/pip install -e gui[dev]         # GUI-only deps (dearpygui, Pillow) into the same venv

venv/bin/python -m opencam_gui.ui.cli demo optics
venv/bin/python -m opencam_gui.ui.cli demo sensor
venv/bin/python -m opencam_gui.ui.cli demo image-generation
```

(Or, after the editable install above, just `opencam-gui demo optics` etc.)

Tips that apply to every demo:

- Every plotted curve, PSF kernel, or generated image comes from importing or
  subprocess-calling the real `tools/*.py` modules -- nothing is
  re-implemented for the GUI. If a number here disagrees with the pipeline,
  that is a bug worth reporting.
- Use the built-in **Lecture scenarios** buttons as checkpoints; they load a
  curated parameter set and teaching point, the same pattern as
  teaching-sims' scenario buttons.
- The **camera recipe** dropdown in every demo loads real
  `config/camera_recipes/*.yaml` files -- try comparing two real cameras
  side by side, not just the illustrative scenario defaults.
- Image Generation's "Fast preview (analytic, no PBRT)" mode works without a
  PBRT build; "Physically accurate (PBRT render)" needs one (see
  `../docs/BUILD_PBRT.txt`) but always supports **Dry run** either way.

Suggested course order: optics -> sensor modelling -> image generation (each
tutorial's "bridge" section explains why).
