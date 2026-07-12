# Quick Start — First Run Walkthrough

This guide walks through a complete pipeline run using the default `iphone_8` camera model and a ColorChecker scene.

---

## Prerequisites

- Python environment activated: `source .venv/bin/activate`
- PBRT built at `third_party/pbrt-v4/build/pbrt`
- Output directory exists: `mkdir -p out`

If PBRT is not yet built, see [Building PBRT](building_pbrt.md). You can skip the PBRT render and use the analytic forward model instead (see Step 3 below).

---

## Step 1: Review the Pipeline Config

Open `config/pipeline.yaml`. Key settings to check before your first run:

```yaml
paths:
  camera_model_name: iphone_8    # active camera model

render:
  pixelsamples: 64               # lower for faster test; raise for quality
  gpu_enabled: true              # set false if no NVIDIA GPU

exposure_time_override_s: 0.06   # 60 ms exposure

sensor_forward:
  mode: pbrt_exr                 # pbrt_exr = use PBRT render; sensor_forward = analytic
  enabled: true
  target_illuminance_lux: 200    # scene illuminance for calibration
```

For a quick test without GPU, set `gpu_enabled: false` and `pixelsamples: 16`.

---

## Step 2: Build the Scene

Generate the ColorChecker `.pbrt` scene file:

```bash
python tools/build_colorchecker_scene.py \
    --output scenes/generated/colorchecker.pbrt \
    --film spectral \
    --spectral-nbuckets 32 \
    --xres 960 --yres 640
```

This generates a PBRT scene containing a 24-patch ColorChecker chart under a D65 illuminant, viewed with the configured lens model.

---

## Step 3: Run the Full Pipeline

```bash
python tools/run_pipeline.py config/pipeline.yaml
```

The pipeline runs these stages in order:

1. **Scene build** — generates the `.pbrt` file if not already present
2. **PBRT render** — produces `out/colorchecker_spectral.exr` (32-channel multispectral)
3. **EXR → electrons** — converts radiance to per-pixel electron counts (`out/sensor_forward_electrons.npz`)
4. **PSF blur** (if enabled) — applies chromatic diffraction PSF
5. **Noise pipeline** — applies full EMVA noise chain, outputs `out/raw_output.tif`
6. **Demosaic + preview** — produces `out/preview_demosaic.png`
7. **EMVA validation** — checks noise statistics against model predictions
8. **Demosaic validation** — computes PSNR vs ground truth

Progress is printed to the console. Total runtime:
- CPU with 64 samples: ~5–10 minutes (depending on resolution)
- GPU with 64 samples: ~30–60 seconds

---

## Step 4: Inspect the Outputs

```
out/
├── colorchecker_spectral.exr        # 32-channel PBRT render
├── sensor_forward_electrons.npz     # electron arrays per channel
├── raw_output.tif                   # RAW16 Bayer mosaic
├── preview_demosaic.png             # sRGB demosaiced preview
├── preview_demosaic_linear.png      # linear (no gamma) preview
├── emva_validation_report.json      # noise model validation
├── ptc_plot.png                     # photon transfer curve
└── demosaic_linear_metrics.json     # PSNR, MAE per channel
```

### Preview Image

`preview_demosaic.png` is the main visual output — a simulated RAW-like preview with sensor noise applied. At 200 lux and 60 ms exposure the iPhone 8 sensor will show visible noise (ISO ~800 equivalent), which is the expected behaviour.

### EMVA Validation Report

Open `out/emva_validation_report.json`:

```json
{
  "temporal_variance_dn2_predicted": 45.2,
  "temporal_variance_dn2_measured": 44.8,
  "temporal_variance_rtol": 0.009,
  "passed": true
}
```

`passed: true` means the measured noise matches the analytic EMVA prediction within the tolerance set by `validation.variance_rtol` (default 7%).

### Photon Transfer Curve

`ptc_plot.png` shows $\sigma_\text{DN}^2$ vs $\mu_\text{DN}$ for several light levels. The curve should be:
- Linear slope ≈ $1/K$ in the mid-signal region (shot noise dominated)
- Flat floor at low signal = read noise floor ($\sigma_d^2/K^2$)
- Dropping at saturation (clipping)

---

## Step 5: Try a Different Camera

Change the camera model in `pipeline.yaml`:

```yaml
paths:
  camera_model_name: nikon_z6
```

The Nikon Z6 has a 5.94 µm pixel pitch, 14-bit ADC, and very low read noise (2.2 e⁻). The preview image will be visibly cleaner at the same exposure settings.

Available cameras are listed in `config/camera_models/EMVA_SUMMARY.md`. You can also use any recipe from `config/camera_recipes/`.

---

## Step 6: Try the Analytic Forward Model

Skip PBRT entirely and use the built-in spectral forward model:

```yaml
sensor_forward:
  mode: sensor_forward    # use analytic model
  enabled: true
```

This computes electron maps from the camera model's spectral sensitivities and the configured illuminant, without rendering. It is faster (seconds vs minutes) but does not produce spatially varying scene content — the output is a spatially uniform image of the ColorChecker patches.

---

## Step 7: Research Lens Presets

For optics experiments, use one of the research recipes:

```yaml
paths:
  camera_model_name: research_realistic_wide22   # 22mm f/1.8 wide angle
```

Or override the lens at run time:

```yaml
lens_type_override: thinlens

lens_overrides:
  thinlens_fov_deg: 60.0
  thinlens_focal_distance: 4.0
```

---

## Common Adjustments

| Goal | Setting |
|------|---------|
| Brighter image | Increase `exposure_time_override_s` |
| Less noise | Use a camera model with larger pixel pitch |
| Faster render | Decrease `render.pixelsamples` |
| Higher quality render | Increase `render.pixelsamples` (≥256 for PTC validation) |
| Test GPU rendering | `render.gpu_enabled: true` |
| Skip PBRT render | `sensor_forward.mode: sensor_forward` |
| Specific illuminant | Set `render.illuminant` to a CSV path |

---

## See Also

- [Pipeline Config](pipeline_config.md) — complete key reference
- [Camera Models](camera_models.md) — adding or customising cameras
- [Validation](validation.md) — interpreting the reports
