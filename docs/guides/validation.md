# Validation — Interpreting Reports and Plots

The pipeline produces several validation outputs that verify physical correctness at different stages. This guide explains what each check does and how to interpret the results.

---

## Overview of Validation Stages

| Stage | Tool | Output | What it checks |
|-------|------|--------|---------------|
| EMVA noise model | `validate_emva_model.py` | JSON report + PTC plot | Noise statistics vs analytic EMVA predictions |
| ColorChecker render | `validate_colorchecker.py` | Console + patch summary | Spectral radiance plausibility |
| Demosaic quality | `validate_demosaic_linear.py` | JSON metrics | PSNR and MAE of demosaiced image vs ground truth |

All three run automatically at the end of `run_pipeline.py` when `validate.enabled`, `validate_emva.enabled`, and `validate_demosaic.enabled` are all `true`.

---

## EMVA Noise Validation

### What It Does

`validate_emva_model.py` runs a Monte Carlo simulation of the noise pipeline at several signal levels (set by `validation.ptc_mu_e_levels`). For each level it:

1. Generates `monte_carlo_trials` synthetic pixel values using the noise chain in `apply_emva_noise.py`
2. Measures the empirical mean $\mu_\text{DN}$ and variance $\sigma^2_\text{DN}$
3. Computes the analytic prediction from `emva_theory.temporal_variance_dn_squared()`
4. Checks that the relative error is within `validation.variance_rtol`

### The JSON Report

```json
{
  "camera_model": "iphone_8",
  "ptc_levels": [
    {
      "mu_e": 0.0,
      "mu_dn_measured": 16.0,
      "mu_dn_predicted": 16.0,
      "sigma2_dn_measured": 0.298,
      "sigma2_dn_predicted": 0.303,
      "rtol": 0.017,
      "passed": true
    },
    {
      "mu_e": 2000.0,
      "sigma2_dn_measured": 88.4,
      "sigma2_dn_predicted": 87.9,
      "rtol": 0.006,
      "passed": true
    }
  ],
  "overall_passed": true
}
```

| Field | Description |
|-------|-------------|
| `mu_e` | Signal level tested [e⁻] |
| `mu_dn_measured` / `_predicted` | Mean DN — checks black level and gain $K$ |
| `sigma2_dn_measured` / `_predicted` | Temporal variance [DN²] |
| `rtol` | Relative error: $|\sigma^2_\text{meas} - \sigma^2_\text{pred}| / \sigma^2_\text{pred}$ |
| `passed` | Whether `rtol < variance_rtol` |

### The PTC Plot (`ptc_plot.png`)

The photon transfer curve (PTC) plots $\log_2(\sigma^2_\text{DN})$ vs $\log_2(\mu_\text{DN})$ (or linear scale). Interpreting the regions:

```
    σ² [DN²]
    │                                       ← saturation (clipping)
    │                              ●
    │                   ●   slope = 1
    │          ●
    │   ●                              ← shot noise dominant (Poisson)
    │─────────────────────────         ← read noise floor
    └──────────────────────── μ [DN]
         black                 full-well
```

- **Read noise floor** (low signal): $\sigma^2 \approx \sigma_d^2/K^2 + \sigma_q^2$ — should be flat
- **Shot noise region** (mid signal): slope = 1 in log-log; gradient = $1/K$
- **Saturation** (high signal): variance drops as clipping suppresses variability
- **Slope** of the linear region = $1/K$ → lets you read off system gain directly

### Common Failure Modes

| Symptom | Likely cause |
|---------|-------------|
| `rtol > variance_rtol` at all levels | Wrong `K` (system gain) — check `overall_system_gain_K_e_per_DN` |
| Floor too high | Read noise `sigma_d_e` too large |
| Floor correct but slope wrong | `K` incorrect |
| PTC bends before `full_well_e` | `full_well_e` set too high; or INL causing early saturation |
| Variance much higher than predicted at dark level | kTC noise enabled unintentionally, or `dsnu_std_e` too large |

---

## Datasheet Parameter Check

When `validation.datasheet.enabled: true`, the camera model's measured parameters are compared against the `datasheet` reference values in the model YAML:

```yaml
validation:
  datasheet:
    overall_system_gain_K_e_per_DN: 4.77
    temporal_dark_noise_sigma_d_e: 2.6
    full_well_e: 4800
    black_level_DN: 16
    parameter_rtol: 0.02    # 2% tolerance
```

This catches accidental edits that break the model calibration.

---

## Demosaic Quality Validation

### What It Does

`validate_demosaic_linear.py` compares the demosaiced output to the ground-truth full-colour electron array (before CFA sampling and noise). It computes:

- **PSNR** (peak signal-to-noise ratio) per channel and overall
- **MAE** (mean absolute error) per channel in normalised units

A border crop (default 2 pixels) is excluded to avoid boundary effects from the demosaic filter at image edges.

### The JSON Report (`demosaic_linear_metrics.json`)

```json
{
  "crop": 2,
  "psnr_r": 38.2,
  "psnr_g": 41.5,
  "psnr_b": 36.9,
  "psnr_overall": 39.0,
  "mae_r": 0.0021,
  "mae_g": 0.0015,
  "mae_b": 0.0028
}
```

### Interpreting PSNR

| PSNR | Interpretation |
|------|---------------|
| > 45 dB | Excellent — demosaic artefacts below visible threshold |
| 35–45 dB | Good — minor colour fringing at sharp edges |
| 25–35 dB | Moderate — visible zipper artefacts in high-frequency regions |
| < 25 dB | Poor — significant colour moire |

For a bilinear demosaic, PSNR of 35–42 dB is typical. Green channel PSNR is always higher because there are twice as many green samples in the Bayer pattern.

### Why PSNR Varies by Camera

- **Small pixel pitch** → more significant spatial crosstalk blur → better PSNR (blurring reduces high-frequency detail that demosaic would miscolour)
- **High noise** → worse PSNR (shot noise adds to demosaic error)
- **High sharpness scene content** → worse PSNR (more aliasing at the Bayer limit)

---

## ColorChecker Validation

`validate_colorchecker.py` checks the spectral plausibility of the PBRT render:

- Verifies that patch radiance values for achromatic patches (patches 19–24: white to black) fall in expected ratios
- Checks for saturated or clipped patches (indicating exposure calibration issues)
- Prints a per-patch luminance summary

Failures here indicate a scene-building or illuminant calibration problem, not a sensor noise problem.

---

## Running Validation Independently

Each validator can be run standalone without the full pipeline:

```bash
# EMVA validation only
python tools/validate_emva_model.py \
    --camera-model-config config/camera_models/nikon_z6.yaml \
    --report-json out/emva_nikon_z6.json

# Demosaic validation only
python tools/validate_demosaic_linear.py \
    --electrons-npz out/sensor_forward_electrons.npz \
    --raw-tif out/raw_output.tif \
    --crop 4 \
    --json-out out/demosaic_metrics.json

# Batch EMVA across all cameras
python tools/batch_validate_emva.py
```

---

## Unit Tests

The test suite in `tests/` validates the physics computations at a lower level:

```bash
python -m pytest tests/ -q --ignore=tests/test_optics_psf.py
```

| Test file | What it tests |
|-----------|--------------|
| `test_camera_model_loader.py` | YAML deep-merge, required field presence |
| `test_demosaic.py` | Bayer pattern sampling and bilinear demosaic |
| `test_emva_validation.py` | Analytic vs Monte Carlo noise variance |
| `test_pipeline_config_semantics.py` | Pipeline YAML key validation |
| `test_optics_psf.py` | PSF sigma calculation (requires numpy ≥2.0) |

All tests except `test_optics_psf.py` should pass on a clean install.

---

## Interpreting the EMVA Summary

`config/camera_models/EMVA_SUMMARY.md` contains a table of all 60+ camera models with their key parameters. The `source` column indicates the parameter reliability:

| Source label | Reliability |
|-------------|-------------|
| `manual_or_default` | Manually set from datasheet or EMVA report |
| `heuristic_scaling_rules_v1` | Estimated by pixel-pitch scaling rules |
| `inferred_web_table_v1` | Inferred from web-published measurements |

Models with `heuristic_scaling_rules_v1` may have EMVA parameters that differ by 10–20% from true measured values. For research use requiring high accuracy, source better data or use a `manual_or_default` model.

---

## See Also

- [Noise Model Theory](../theory/03_noise_model.md) — EMVA 1288 physics
- [Camera Models](camera_models.md) — `validation` section in camera model schema
- `tools/emva_theory.py` — analytic prediction functions
