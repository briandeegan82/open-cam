# Tutorial 04 -- Sensor Noise Modelling (EMVA1288 Photon Transfer, DSNU, PRNU)

**Demo:** `opencam-gui demo sensor`
**Audience:** graduate / advanced undergraduate; Tutorial 03 assumed
**Goal:** connect electron counts, shot noise, read noise, dark current, DSNU,
and PRNU to the same closed-form model `tools/validate_emva_model.py` uses to
check a camera config against its datasheet -- including the EMVA1288
protocol that calculates DSNU and PRNU from stacks of dark and flat-field
frames.

Every curve in this demo is produced by `tools/emva_theory.py` -- the exact
functions the EMVA validator calls, not a re-derivation. DSNU1288 and
PRNU1288 on the second tab are the same spatial estimators (temporal
averaging, residual-temporal correction). The dark-current temperature
helper is a documented, read-only mirror of the formula inline in
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

### 1.5 DSNU and PRNU -- how EMVA1288 calculates them

The photon transfer curve above is **temporal** noise: it is the variance of
repeated readings of the *same* pixel. DSNU and PRNU are **spatial** (fixed
pattern) noise: they are the same every frame, so they do not appear on the
PTC. EMVA1288 isolates them by averaging `L` identical frames of a uniform
field, then measuring the spatial standard deviation of that average image.

**Temporal mean image.** For a stack `y_i[m,n]`, `i = 1..L`:

```
y_bar[m,n] = (1/L) * sum_i y_i[m,n]
```

Temporal noise in `y_bar` is reduced by `sqrt(L)`. The leftover spatial
structure is the fixed pattern, plus a residual `sigma_temporal / sqrt(L)`.

**Residual-temporal correction** (EMVA 1288 §7):

```
s^2_y.bar  = spatial variance of y_bar
sigma^2_y  = mean over pixels of the per-pixel temporal variance
s^2_y      = s^2_y.bar - sigma^2_y / L
```

**DSNU1288** from a stack of *dark* frames:

```
DSNU1288 = s_y.dark / K     (electrons)
```

DSNU is an **additive** per-pixel offset (dark-current variation). It is
visible with the shutter closed and does not grow with signal.

**PRNU1288** from a second stack at ~50% saturation (flat field), after
removing the dark spatial variance:

```
PRNU1288 = sqrt(s^2_y.50 - s^2_y.dark) / (mu_y.50 - mu_y.dark)
```

PRNU is a **multiplicative** per-pixel gain (photodiode area, fill factor,
microlens alignment). It is invisible in the dark and grows linearly with
mean signal. After infinite averaging the two combine as

```
sigma_spatial(mu) = sqrt( DSNU^2 + (PRNU * mu)^2 )
```

The DSNU map generator matches `tools/apply_emva_noise.py`: Gaussian
zero-mean offset (the textbook EMVA statistical model) or the pipeline's
log-normal dark-current map (long positive tail = hot pixels). PRNU is
`g = max(0, 1 + N(0, PRNU^2))`. The measurement stacks keep the dark
histogram linear (an analog offset / optical black), which is an EMVA1288
requirement -- you do not measure DSNU on a clipped dark floor.

---

## 2. UI map

| Panel / control | What it shows |
| --- | --- |
| Photon transfer curve | `Var(DN)` vs `mean(DN)`, log-log, with the dark floor marked |
| SNR vs mean signal | Derived from the same curve arrays |
| DSNU / PRNU tab: maps | Live grayscale images (fixed scale so slider amplitude is visible) |
| DSNU / PRNU tab: spatial std | `sqrt(DSNU^2 + (PRNU x mu)^2)` vs signal, plus measured points |
| DSNU / PRNU tab: histograms | Pixel distribution -- Gaussian vs log-normal hot-pixel tail |
| Status line | Gain, read noise, full well, shot/read crossover, dark floor, FPN |
| Monte Carlo verification | One-click theory vs 20000-trial simulation at a chosen mu_e |
| Measure DSNU1288 / PRNU1288 | Runs the EMVA protocol on simulated dark + 50% stacks |

**Core sliders:** read noise sigma_d, gain K, full well, black level, Poisson toggle, PRNU fraction, DSNU (e-), frames L to average.
**Advanced:** dark current rate, temperature, Arrhenius Ea, exposure time, DSNU model (gaussian / lognormal).
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

### Experiment F -- How DSNU is calculated (~8 min)

1. Switch to the **DSNU / PRNU** tab. Load **No FPN: averaging kills spatial noise**.
2. Both maps should be essentially flat (no frozen pattern). Click
   **Measure DSNU1288 / PRNU1288**.
3. Read the two dark numbers: the *uncorrected* spatial std of the mean image
   is leftover read noise (`sigma_d / sqrt(L)`); **DSNU1288** after the
   `s^2_y.bar - sigma^2 / L` correction should sit near zero.
4. Drop `L` to 10 and measure again -- the uncorrected number grows, DSNU1288
   stays near zero. That difference *is* the EMVA calculation.

### Experiment G -- DSNU is additive (~8 min)

1. Load **DSNU only: additive dark pattern**. The DSNU map is a frozen speckle;
   the PRNU map is flat.
2. Measure: DSNU1288 should recover the slider (~3 e-); PRNU1288 ~ 0.
3. Raise `L` -- DSNU1288 does **not** fall as `1/sqrt(L)`. Averaging cannot
   remove a pattern that is the same in every frame.
4. The spatial-std curve is a horizontal line at the DSNU floor: the pattern
   amplitude does not grow with signal.

### Experiment H -- How PRNU is calculated (~8 min)

1. Load **PRNU only: multiplicative gain map**. The DSNU map is now flat; the
   PRNU map shows the gain field (percent from mean).
2. The spatial-std curve starts at 0 and rises linearly (`PRNU x mu`).
3. Measure: PRNU1288 should recover ~3%. The formula is
   `sqrt(s^2_50 - s^2_dark) / (mu_50 - mu_dark)` -- dark spatial variance is
   subtracted so leftover DSNU is not counted as PRNU.
4. Drag PRNU down toward 0.5% (a typical consumer sensor) and watch the slope
   flatten; the maps get quieter.

### Experiment I -- Hot pixels and a real camera (~8 min)

1. Load **Hot pixels: log-normal DSNU**. The histogram has a long positive
   tail -- a few leaky pixels dominate DSNU. This is the generative model in
   `tools/apply_emva_noise.py`, not the Gaussian textbook field.
2. Load **Real camera: Nikon Z6** vs **iPhone 8** and compare PRNU (0.7% vs
   1.8%) on the spatial-std curve: at the same electron count the phone's
   multiplicative pattern is more than twice as strong.

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
| No-FPN, L=50, sigma_d=8 e-: uncorrected dark spatial std | e- (expect ~1.1) | |
| Same, DSNU1288 after temporal correction | e- (expect ~0) | |
| DSNU-only scenario: measured DSNU1288 | e- (expect ~3) | |
| PRNU-only, 3%: measured PRNU1288 | % | |
| Spatial std at 50% well: `sqrt(DSNU^2 + (PRNU * mu)^2)` | e- | |

---

## 5. Check your understanding

1. Why does the PTC flatten instead of dropping to zero at very low signal?
2. Why does the dark-current scenario change the *variance* at the dark floor
   but not the mean DN in this model?
3. A sensor's read noise doubles. Sketch qualitatively how the PTC and SNR
   curves change.
4. Why is doubling the exposure time *not* equivalent to doubling the target
   illuminance for read noise, even though both double `mu_e`?
5. Why does averaging more dark frames *not* reduce DSNU1288 the way it
   reduces the uncorrected spatial std of the mean image?
6. A colleague computes PRNU as `s_y.50 / mu_y.50` without subtracting the
   dark spatial variance. When is that approximately OK, and when does it
   over-estimate PRNU?
7. Sketch `sigma_spatial` vs `mu` for (a) DSNU only, (b) PRNU only, (c) both.

---

## 6. Bridge to the next tutorial

This tutorial took the electron count `mu_e` as given and asked what noise sits
on it. The next one asks where `mu_e` came from in the first place -- scene
luminance, aperture, shutter and ISO -- and then goes underneath the clean
Poisson-and-Gaussian model to the defects it does not describe: blooming, hot
pixels, kTC reset noise, ADC non-linearity and readout banding. The photon
transfer curve you just built becomes the map for reading an exposure.
