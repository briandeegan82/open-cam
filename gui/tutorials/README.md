# Tutorials

Hands-on guides for the Open Cam interactive demos. Each tutorial explains the
underlying principles, maps them to the UI, and walks through experiments you
can run in lecture or self-paced lab.

The seven demos form one arc rather than seven independent topics. Geometry
forms the image, optics blurs it, MTF measures the blur, the sensor adds noise,
exposure and defects distort the electrons, the ISP turns them into colour, and
the last demo runs the whole chain on a real scene.

| Tutorial | Demo command | Time (approx.) | What it adds |
| --- | --- | --- | --- |
| [01 -- Fundamental optics](01-fundamental-optics.md) | `opencam-gui demo geometry` | 45-60 min | Thin lens, field of view, depth of field, equivalence, cos^4 falloff, distortion |
| [02 -- PSF and aberrations](02-psf-aberrations.md) | `opencam-gui demo optics` | 50-70 min | Diffraction, geometric aberration, lateral CA, stray light |
| [03 -- Resolution and MTF](03-resolution-mtf.md) | `opencam-gui demo mtf` | 45-60 min | Slanted-edge SFR, MTF50, diffraction cutoff, Nyquist and aliasing |
| [04 -- Sensor noise (EMVA1288)](04-sensor-noise.md) | `opencam-gui demo sensor` | 55-75 min | Photon transfer curve, read noise, dark current, DSNU, PRNU |
| [05 -- Exposure and defects](05-exposure-defects.md) | `opencam-gui demo exposure` | 50-70 min | Exposure triangle, EV, ISO as gain, blooming, hot pixels, kTC, ADC INL/DNL, banding |
| [06 -- Colour and the ISP](06-colour-isp.md) | `opencam-gui demo isp` | 50-70 min | Spectral colour, CFA and demosaic, white balance, CCM, delta-E, metamerism |
| [07 -- Image generation](07-image-generation.md) | `opencam-gui demo image-generation` | 45-60 min | The whole chain end to end on a real scene |

`opencam-gui topics` prints the same list from the code, so it cannot go stale.

## Suggested lecture mapping

A twelve-week graduate course, two hours a week, with the demos used live:

| Week | Topic | Demo | Notes |
| --- | --- | --- | --- |
| 1 | Image formation, thin lens, field of view | 01 | Experiments A and B |
| 2 | Depth of field, equivalence, the aperture trade-off | 01 | Experiments C-F |
| 3 | Diffraction and the point-spread function | 02 | Experiments A-C |
| 4 | Aberrations, chromatic effects, stray light | 02 | Experiments D-H |
| 5 | MTF, the slanted-edge method, theory overlays | 03 | Experiments A-E |
| 6 | Sampling, Nyquist, aliasing and the OLPF trade | 03 | Experiment F, plus 02's Experiment F for the contrast connection |
| 7 | Photons to electrons, shot and read noise, the PTC | 04 | Experiments A-C |
| 8 | Dark current, DSNU and PRNU, the EMVA1288 protocol | 04 | Experiments F-I |
| 9 | Exposure, EV, and what ISO actually is | 05 | Experiments A-C |
| 10 | Sensor defects and how to recognise them | 05 | Experiments D-H |
| 11 | Spectral colour, the CFA, demosaic and white balance | 06 | Experiments A-D |
| 12 | Colour correction, metamerism, alternative CFAs, whole pipeline | 06 then 07 | 06's E-H, then 07 end to end |

Shorter courses: 01, 03, 04 and 07 stand alone reasonably well as a four-lecture
introduction. Tutorials 02 and 03 are the natural pair (blur as a picture, then
blur as a number), as are 04 and 05 (noise statistics, then the defects the
statistics do not describe).

## Before you start

```bash
cd ~/open-cam
python3 -m venv venv                     # if you don't already have one
venv/bin/pip install -r requirements.txt # core pipeline deps (numpy, OpenEXR, ...)
venv/bin/pip install -e gui[dev]         # GUI-only deps (dearpygui, Pillow) into the same venv

venv/bin/python -m opencam_gui.ui.cli topics          # list the demos in order
venv/bin/python -m opencam_gui.ui.cli demo geometry
venv/bin/python -m opencam_gui.ui.cli demo mtf --list-scenarios
venv/bin/python -m opencam_gui.ui.cli demo isp --scenario tungsten
```

(Or, after the editable install above, just `opencam-gui demo geometry` etc.)

Tips that apply to every demo:

- Every plotted curve, PSF kernel, measured MTF and generated image comes from
  importing or subprocess-calling the real `tools/*.py` modules -- nothing is
  re-implemented for the GUI. If a number here disagrees with the pipeline, that
  is a bug worth reporting.
- Use the built-in **Lecture scenarios** buttons as checkpoints; each loads a
  curated parameter set and states its teaching point in the banner.
  `--list-scenarios` prints the same text without launching a window, which is
  useful for building a lecture plan.
- The **camera recipe** dropdown in every demo loads real
  `config/camera_recipes/*.yaml` files. Comparing two actual cameras is usually
  more instructive than sweeping a slider through values no camera has.
- **Presenter mode** hides the advanced controls, which keeps a live
  demonstration to the handful of sliders the point actually needs.
- Only Tutorial 07's "Physically accurate (PBRT render)" mode needs a PBRT build
  (see `../docs/BUILD_PBRT.txt`). Everything else, including Tutorial 03's
  synthetic edge source, runs on NumPy alone.

## For contributors

Adding an eighth demo is one entry in `TOPICS` in
[opencam_gui/ui/cli.py](../opencam_gui/ui/cli.py), a `topics/<id>/scenarios.py`,
and a `ui/desktop/<id>_app.py` subclassing `DemoApp` from
[ui/desktop/base.py](../opencam_gui/ui/desktop/base.py). The physics goes in
`tools/`, never in the GUI; the `core/` adapter only reshapes what `tools/`
returns into what the plots need. Headless tests in `gui/tests/` assert that
adapters agree with direct `tools` calls, and that the claims the scenarios make
in their teaching-point text are actually true.
