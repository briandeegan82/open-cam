# Tutorial 03 -- Image Generation (Scene -> Sensor -> Image)

**Demo:** `opencam-gui demo image-generation`
**Audience:** graduate / advanced undergraduate
**Goal:** run the real Open Cam pipeline end to end -- scene, illumination,
sensor -- and see how the choices from Tutorials 01 and 02 combine in an
actual generated image.

This demo does not render anything itself. Every click on **Generate** builds
a command plan and runs the real `tools/*.py` scripts (or the full
`tools/run_pipeline.py`) as subprocesses, streamed live into the log panel.
The two preview panels are the literal PNG files those scripts wrote to
`out/colorchecker_noisy_png/`.

---

## 1. Principles

### 1.1 Two execution modes

- **Fast preview (analytic, no PBRT)** -- ColorChecker only. Skips the PBRT
  renderer entirely: `build_colorchecker_scene.py` writes the chart's
  reflectance spectra directly, `spectral_sensor_forward.py` (`analytic`
  mode) converts them straight to electrons using the chosen illuminant and
  illuminance, and `apply_emva_noise.py` adds sensor noise. Seconds, not
  minutes -- good for iterating on illumination and camera choices.
- **Physically accurate (PBRT render)** -- the full path used by
  `tools/run_pipeline.py`: spectral Monte Carlo rendering through the actual
  lens model (pinhole, thin lens, or a traced multi-element `realistic` lens),
  then the same sensor forward + noise stages. Requires a built PBRT binary
  (`docs/BUILD_PBRT.txt`); IQ targets (slanted edge, ISO noise, Siemens star)
  only support this path.

Both modes call the *same* sensor forward and EMVA noise tools -- the only
difference is where the pre-sensor spectral image comes from.

### 1.2 Illumination: spectrum and intensity

- **Spectrum** selects a CSV from `spectra/illuminant/interpolated/` (D65
  daylight, A tungsten, F11 fluorescent, various phosphor-converted LEDs,
  ...). It changes the light's spectral power distribution, which interacts
  with each ColorChecker patch's own reflectance spectrum and the sensor's QE
  curves -- this is where colour-rendering differences between light sources
  come from.
- **Intensity** (lux) is the calibration target for `sensor_forward`'s
  `target_illuminance_lux`: it sets how many electrons the scene ultimately
  produces, independent of how bright the Monte Carlo render itself looks.
  Combined with **exposure time**, it is the same reciprocity relationship a
  real camera's exposure meter uses -- doubling either one doubles `mu_e`.

### 1.3 Where Tutorial 01 and 02 show up here

- The camera recipe's `lens.post_psf` settings (Tutorial 01) apply during the
  physically-accurate path's post-render stage -- or are already baked into
  the ray-traced blur if `lens.camera: realistic`.
- The camera recipe's `noise.emva` settings (Tutorial 02) are exactly what
  `apply_emva_noise.py` uses to turn electrons into DN here. A low-lux,
  small-pixel scenario should look exactly as noisy as the PTC from Tutorial
  02 predicts for that camera at that signal level.

---

## 2. UI map

| Panel / control | What it shows |
| --- | --- |
| Scene dropdown | ColorChecker, or an IQ target (slanted edge / ISO noise / Siemens star) |
| Generation mode | Fast preview (analytic) vs Physically accurate (PBRT) |
| Illumination spectrum / intensity | Illuminant CSV + target lux, with quick presets |
| Clean / Noisy preview | `clean_demosaic_rgb8.png` vs `noisy_demosaic_rgb8.png`, side by side |
| Command log | The literal subprocess commands and their stdout, live |
| Dry run | Prints the exact command plan without executing anything (always available) |

---

## 3. Guided walkthrough (lecture path)

### Experiment A -- Baseline render (~5 min)

1. Load **Studio daylight ColorChecker** and click **Generate**.
2. Watch the command log: scene build, sensor forward, EMVA noise, in that
   order -- the same three stages the fast path always runs.
3. Compare Clean vs Noisy: noise should be subtle at 1000 lux on a
   full-frame sensor.

### Experiment B -- Illuminant spectrum (~8 min)

1. Load **Tungsten indoor lighting** (illuminant A) and Generate.
2. Compare the result against the D65 baseline -- patches should read
   warmer under A even at similar exposure.
3. Try **Fluorescent office lighting** (F11, a narrow tri-band spectrum) and
   look for patches whose colour rendering suffers more than under smooth
   daylight spectra.

### Experiment C -- Low light and sensor noise (~10 min)

1. Load **Low-light phone shot** (iPhone 8, 20 lux) and Generate.
2. Compare against the Nikon Z6 baseline: visibly noisier, smaller sensor,
   lower full well -- consistent with the PTC comparison from Tutorial 02.
3. Raise `Exposure time (s)` while keeping lux fixed and regenerate: SNR
   should improve the same way it would from raising illuminance instead
   (reciprocity).

### Experiment D -- Highlight clipping (~6 min)

1. Load **Bright overcast day** (15000 lux) and Generate.
2. Check `run_stats` output text for the DN range -- the brightest patches
   should sit close to the ADC's full range.
3. Push the lux preset to "Direct sun" and confirm highlight patches clip.

### Experiment E -- Physically accurate path (requires a PBRT build)

1. Toggle **Physically accurate (PBRT render)**.
2. Turn on **Dry run** first and Generate -- inspect the exact PBRT-driving
   commands without needing PBRT installed.
3. If PBRT is built (see `docs/BUILD_PBRT.txt`), turn Dry run off and compare
   render time and output against the fast path for the same scene.
4. Try a Siemens star or slanted-edge target -- these only support the
   physically-accurate path, since their scene builder doesn't emit the
   reflectance manifest the analytic path needs.

---

## 4. Self-paced lab sheet

| Task | Measurement | Your value |
| --- | --- | --- |
| D65 vs A, same lux: which patches shift hue most? | qualitative | |
| iPhone 8 at 20 lux vs 200 lux: visible noise difference | qualitative | |
| Doubling exposure time vs doubling lux at low light | same result? (y/n) | |
| Bright overcast (15000 lux): which patches clip first? | qualitative | |
| Dry run command count, fast vs physically-accurate mode | count | |

---

## 5. Check your understanding

1. Why can the analytic fast path work without a PBRT render at all?
2. Why do IQ targets (slanted edge, Siemens star, ISO noise) require the
   physically-accurate path?
3. What's the physical difference between changing illumination spectrum
   and changing illumination intensity, in terms of what happens at the
   sensor?
4. If two scenarios produce the same `mu_e` at the sensor but different lux
   and exposure-time combinations, should their noise look the same? Why?

---

## 6. Where to go next

Everything here calls the *same* pipeline `tools/run_pipeline.py` drives in
production. To go further: edit a `config/camera_recipes/*.yaml` and rerun
this demo to see your change reflected immediately, or drop into
`tools/run_pipeline.py --dry-run` directly for the full config surface (GPU
rendering, strict physical-accuracy validation, IQ-target batch generation)
that this teaching GUI intentionally keeps out of view.
