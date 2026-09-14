# Tutorial 02 -- Sensor Modelling (EMVA1288 Photon Transfer)

**Demo:** `opencam-gui demo sensor`
**Audience:** graduate / advanced undergraduate
**Goal:** connect electron counts, shot noise, read noise, and dark current to
the same closed-form model `tools/validate_emva_model.py` uses to check a
camera config against its datasheet.

Every curve in this demo is produced by `tools/emva_theory.py` -- the exact
functions the EMVA validator calls, not a re-derivation. The dark-current
temperature helper is a documented, read-only mirror of the formula inline in
`tools/apply_emva_noise.py` (see `opencam_gui/core/dark_current.py` for the
citation); the real image-generation pipeline always computes it itself.

---

## 1. Principles

### 1.1 From electrons to DN

The sensor model (matching `tools/apply_emva_noise.py`) is:

```
DN = e / K + black_level          (K = system gain, e-/DN)
```

where `e` is Poisson-distributed shot noise on the mean signal `mu_e`, plus
independent Gaussian read noise `sigma_d` (electrons), optionally clipped at
the full well before the ADC.

### 1.2 The photon transfer curve (PTC)

Plotting `Var(DN)` against `mean(DN)` on log-log axes is the standard EMVA1288
diagnostic:

- **Shot-noise-limited region**: `Var(DN) ~ mean_e / K^2` -- a straight line of
  slope 1. This is the majority of a good sensor's useful range.
- **Read-noise-limited region**: at low signal, `Var(DN) -> sigma_d^2 / K^2`,
  flattening the curve. The crossover happens at `mu_e ~ sigma_d^2` electrons.
- **Dark floor** (`mu_e = 0`): electrons are `max(0, N(0, sigma_d))` before the
  ADC, giving a small but nonzero mean and variance even with the shutter closed.

### 1.3 Dark current and temperature

Two models, selected by `dark_activation_energy_eV`:

- **Doubling rule** (`= 0`): `I(T) = I_ref x 2^((T - T_ref) / doubling_per_C)`.
- **Arrhenius with T^1.5** (`> 0`): generation-dominated dark current,
  `I(T) = I_ref x (T/T_ref)^1.5 x exp(Ea/kB x (1/T_ref - 1/T))`, typical
  `Ea = 0.55-0.70 eV` for CMOS.

Dark current adds Poisson shot-noise variance (`mu_dark_e`) on top of the
signal's shot noise -- it does not, in this model, shift the *mean* signal DN,
which is why the dark-current scenario shows up as a rising **variance** at
the dark floor rather than a shifted mean.

### 1.4 SNR

`SNR (dB) = 20 log10((mean_DN - black_level) / sqrt(Var(DN)))`. It climbs
steeply through the read-noise-limited region, then flattens toward
`10 log10(mu_e)` once photon shot noise dominates (a `sqrt(N)` law).

---

## 2. UI map

| Panel / control | What it shows |
| --- | --- |
| Photon transfer curve | `Var(DN)` vs `mean(DN)`, log-log, with the dark floor marked |
| SNR vs mean signal | Derived from the same curve arrays |
| Status line | Gain, read noise, full well, shot/read crossover, dark floor |
| Monte Carlo verification | One-click theory vs 20000-trial simulation at a chosen mu_e |

**Core sliders:** read noise sigma_d, gain K, full well, black level, Poisson toggle.
**Advanced:** dark current rate, temperature, Arrhenius Ea, exposure time.
**Camera recipe dropdown:** loads all of the above from a real
`config/camera_recipes/*.yaml` via `tools/camera_model.load_camera_model`.

---

## 3. Guided walkthrough (lecture path)

### Experiment A -- Ideal photon-limited sensor (~6 min)

1. Load **Ideal photon-limited sensor**.
2. Confirm the PTC is a straight line across almost the whole plot -- read
   noise (`sigma_d = 0.5 e-`) is negligible almost everywhere.
3. Read the shot/read crossover from the status line: it should sit near the
   very bottom of the signal range.

### Experiment B -- Read-noise-limited sensor (~8 min)

1. Load **Read-noise-limited low-light sensor** (`sigma_d = 8 e-`).
2. Identify the flat (read-noise-limited) region at low signal and the
   slope-1 (shot-noise-limited) region at high signal.
3. Raise `sigma_d` further and watch the crossover point move right on the
   status line -- more of the sensor's range becomes read-noise-limited.

### Experiment C -- Hot sensor / dark current (~8 min)

1. Load **Hot sensor: dark current takes over** (Arrhenius, Ea = 0.63 eV, 55 C, 1 s exposure).
2. Note `mu_dark` in the status line and the dark-floor marker's variance
   sitting above the theoretical zero-signal point of a cold sensor.
3. Drag temperature from 20 C to 60 C and watch the dark floor rise.
4. Drop exposure time back toward 1 ms -- the same temperature now
   contributes far less dark charge (`mu_dark` scales linearly with
   integration time).

### Experiment D -- Real camera comparison (~10 min)

1. Load **Real camera: Nikon Z6**, note the full well and gain from the
   status line.
2. Load **Real camera: iPhone 8** and compare: much smaller full well means
   the entire usable signal range sits further left on the same log-log axes.
3. Use the camera recipe dropdown to load either camera directly (not just
   the scenario's illustrative starting values) and confirm the numbers match
   `config/camera_recipes/<name>.yaml`.

### Experiment E -- Monte Carlo verification (~5 min)

1. Set `mu_e to verify` to a mid-range signal level.
2. Click **Run Monte Carlo check** and compare the theory and 20000-trial
   simulation means/variances -- they should agree to a few percent, exactly
   the check `tools/validate_emva_model.py` performs automatically in CI.

---

## 4. Self-paced lab sheet

| Task | Measurement | Your value |
| --- | --- | --- |
| `sigma_d = 2.6 e-`: shot/read crossover | electrons | |
| Same sensor: mean DN at that crossover | DN | |
| Ea = 0.63 eV, 20 C -> 50 C: dark current multiplier | dimensionless | |
| Nikon Z6: mean DN at 50% full well | DN | |
| iPhone 8: mean DN at 50% full well | DN | |
| mu_e = full_well/2: theory vs Monte Carlo variance, percent difference | % | |

---

## 5. Check your understanding

1. Why does the PTC flatten instead of dropping to zero at very low signal?
2. Why does the dark-current scenario change the *variance* at the dark floor
   but not the mean DN in this model?
3. A sensor's read noise doubles. Sketch qualitatively how the PTC and SNR
   curves change.
4. Why is doubling the exposure time *not* equivalent to doubling the target
   illuminance for read noise, even though both double `mu_e`?

---

## 6. Bridge to the next tutorial

You now have both halves of the imaging chain: the PSF that blurs the signal
and the EMVA1288 model that turns it into noisy DN. The Image Generation
tutorial runs both together on a real scene -- pick a light source, an
illuminance, and a camera, and generate an actual image through the same
`tools/*.py` scripts.
