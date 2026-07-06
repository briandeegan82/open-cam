# Pipeline Config Reference — `config/pipeline.yaml`

This document describes every key in `config/pipeline.yaml`. The file is read by `tools/run_pipeline.py`, which orchestrates the full rendering and noise pipeline.

---

## `paths` — File Paths

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `scene_builder` | string | `tools/build_colorchecker_scene.py` | Python script that generates the PBRT scene |
| `pbrt` | string | `third_party/pbrt-v4/build/pbrt` | Path to PBRT v4 binary |
| `noise_tool` | string | `tools/apply_emva_noise.py` | Noise pipeline script |
| `sensor_forward_tool` | string | `tools/spectral_sensor_forward.py` | Analytic forward model script |
| `pbrt_exr_to_electrons_tool` | string | `tools/pbrt_spectral_exr_to_electrons.py` | PBRT EXR → electrons conversion |
| `psf_tool` | string | `tools/apply_spectral_psf.py` | Post-PSF blur tool |
| `validate_tool` | string | `tools/validate_colorchecker.py` | ColorChecker validation script |
| `validate_emva_tool` | string | `tools/validate_emva_model.py` | EMVA noise model validation |
| `validate_demosaic_tool` | string | `tools/validate_demosaic_linear.py` | Demosaic quality validation |
| `camera_model_name` | string | `iphone_8` | Name of the active camera model (matches `config/camera_models/<name>.yaml`) |
| `scene_file` | string | `scenes/generated/colorchecker.pbrt` | Output path for generated scene file |
| `exr_out` | string | `out/colorchecker_spectral.exr` | PBRT render output EXR |
| `sensor_forward_electrons_npz` | string | `out/sensor_forward_electrons.npz` | Electron array output |
| `emva_validation_report` | string | `out/emva_validation_report.json` | EMVA validation JSON report |
| `demosaic_metrics_json` | string | `out/demosaic_linear_metrics.json` | Demosaic quality metrics |
| `out_dir` | string | `out` | Directory for all output files |

---

## `render` — PBRT Render Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `film` | string | `spectral` | PBRT film type. Must be `spectral` for the pipeline to work |
| `film_output` | string\|null | `null` | Override EXR output path (null = use `paths.exr_out`) |
| `spectral_nbuckets` | int | `32` | Number of spectral wavelength buckets |
| `spectral_lambda_min` | float | `360.0` | Minimum wavelength [nm] |
| `spectral_lambda_max` | float | `830.0` | Maximum wavelength [nm] |
| `xres` | int | `960` | Image width in pixels |
| `yres` | int | `640` | Image height in pixels |
| `pixelsamples` | int | `64` | Monte Carlo samples per pixel. Use ≥256 for accurate PTC validation; 16–64 for quick iteration |
| `cam_dist` | float | `4.0` | Camera distance from scene origin [m] |
| `light_scale` | float | `1.0` | Multiplier on scene light power |
| `illuminant` | string\|null | `spectra/illuminant/interpolated/D65.csv` | CSV file for scene illuminant spectrum. null = PBRT default |
| `builder_extra_args` | list | `[]` | Extra CLI arguments passed to the scene builder script |
| `gpu_enabled` | bool | `false` | Enable GPU rendering (injects `--gpu`, pbrt's OptiX wavefront GPU path). Requires an OptiX-enabled PBRT build. `--wavefront` alone runs on the CPU |
| `pbrt_args` | list | `["--stats"]` | Extra arguments passed directly to PBRT. `--gpu` is added automatically when `gpu_enabled: true` |

---

## `validate` — ColorChecker Validation

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Run `validate_colorchecker.py` after the pipeline |

---

## `validate_emva` — EMVA Noise Validation

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Run `validate_emva_model.py` after noise pipeline |

---

## `validate_demosaic` — Demosaic Quality Validation

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Run `validate_demosaic_linear.py` after demosaic |
| `crop` | int | `2` | Pixel border crop to exclude edge artefacts from metrics |

---

## `lens_type_override` — Lens Type

Overrides the `lens.camera` field from the active camera model.

| Value | Description |
|-------|-------------|
| `null` | Use the value from the camera model |
| `pinhole` | Ideal pinhole (no aberrations, no vignetting) |
| `thinlens` | Ideal thin lens (depth of field, no aberrations) |
| `realistic` | Ray-traced multi-element lens from a `.dat` prescription file |

---

## `lens_overrides` — Per-Lens-Type Overrides

Override specific lens parameters at run time. All keys default to `null` (use camera model value).

| Key | Applies to | Description |
|-----|-----------|-------------|
| `realistic_lensfile` | `realistic` | Path to `.dat` lens prescription file |
| `realistic_aperture_diameter_mm` | `realistic` | Override aperture diameter [mm] |
| `realistic_focus_distance` | `realistic` | Focus distance [m] |

> **Note**: `pinhole_fov_deg` and `thinlens_*` keys are only meaningful when `lens_type_override` is `pinhole` or `thinlens` respectively.

---

## `strict_physical_accuracy`

Controls validation strictness for physical accuracy checks.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `strict_qe_validation` | bool | `false` | Fail if QE data does not cover the full spectral range |
| `strict_calibration_validation` | bool | `false` | Fail if calibration tier is below minimum |
| `calibration_tier_policy` | string | `semi_strict` | One of: `permissive`, `semi_strict`, `strict` |

---

## `exposure_time_override_s`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `exposure_time_override_s` | float\|null | `0.06` | Integration time [s] at run time. Overrides `sensor.integration_time_s` in the camera model. Set to `null` to use the camera model value |

---

## `sensor_forward` — Forward Model Mode

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `mode` | string | `pbrt_exr` | `pbrt_exr`: use PBRT render; `sensor_forward`: use analytic model |
| `enabled` | bool | `true` | Run sensor forward model stage |
| `target_illuminance_lux` | float\|null | `200` | Override scene illuminance [lux] for calibration. `null` uses camera model value |

---

## `noise` — Noise Pipeline

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Run the EMVA noise pipeline |
| `seed` | int | `0` | Random seed for reproducibility |
| `exposure_scale` | float\|null | `null` | Multiply electron count before noise (for ISO simulation). `null` = no scaling |
| `preview_percentile` | float | `99.5` | Percentile for preview image white clipping |
| `preview_no_normalize` | bool | `true` | Disable automatic normalisation in preview |
| `preview_white_balance_enabled` | bool | `false` | Apply white balance in preview |
| `preview_color_correction_enabled` | bool | `false` | Apply colour correction matrix in preview |

---

## Full Example

```yaml
paths:
  camera_model_name: nikon_z6
  out_dir: out/nikon_z6_test

render:
  pixelsamples: 256
  gpu_enabled: false
  xres: 1920
  yres: 1280

lens_type_override: realistic
lens_overrides:
  realistic_lensfile: null         # use camera model default
  realistic_focus_distance: 4.0

exposure_time_override_s: 0.02    # 20 ms

sensor_forward:
  mode: pbrt_exr
  enabled: true
  target_illuminance_lux: 500

noise:
  enabled: true
  seed: 42

validate_emva:
  enabled: true
validate_demosaic:
  enabled: true
  crop: 4
```

---

## See Also

- [Camera Models](camera_models.md) — camera model YAML key reference
- [Quick Start](quickstart.md) — running the pipeline
- [Tools Reference](tools_reference.md) — individual tool CLI options
