# Tools Reference

CLI reference for every script in `tools/`. All tools accept `--repo-root` to change the base path (default: the repository root, auto-detected).

---

## `run_pipeline.py` — Pipeline Orchestrator

Runs the complete pipeline from scene build through noise to validation, driven by `config/pipeline.yaml`.

```bash
python tools/run_pipeline.py [OPTIONS] [config_yaml]
```

| Option | Default | Description |
|--------|---------|-------------|
| `config_yaml` (positional) | `config/pipeline.yaml` | Pipeline config file |
| `--repo-root PATH` | auto-detected | Repository root path |
| `--camera-model-name NAME` | from pipeline.yaml | Override camera model name at run time |
| `--dry-run` | false | Print commands only, do not execute |
| `--name NAME` | null | Optional run name for manifest filename |
| `--skip-render` | false | Reuse the existing rendered EXR (`paths.exr_out`); skip the pbrt render and post-render PSF blur. Fails if the EXR is missing. Use to re-run only the sensor/noise/demosaic stages without a GPU render. |
| `--emit-demosaic-linear16` | false | Pass through to the noise step: also write linear 16-bit demosaiced images (see `apply_emva_noise.py`). |

**Example:**

```bash
# Standard run
python tools/run_pipeline.py config/pipeline.yaml

# Override camera without editing YAML
python tools/run_pipeline.py --camera-model-name nikon_z6

# Preview what would run
python tools/run_pipeline.py --dry-run
```

---

## `apply_emva_noise.py` — EMVA Noise Pipeline

Applies the complete EMVA 1288 noise chain to an electron array, outputting a RAW16 TIFF and preview PNGs.

```bash
python tools/apply_emva_noise.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--repo-root PATH` | auto | Repository root |
| `--config PATH` | null | YAML config path (legacy; prefer `--camera-model-config`) |
| `--camera-model-config PATH` | null | Camera model YAML path |
| `--electrons-npz PATH` | null | Pre-computed electron array (.npz with H×W×3 float32) |
| `--exposure-scale FLOAT` | null | Multiply electrons before noise (ISO simulation) |
| `--seed INT` | 0 | RNG seed for reproducibility |
| `--out-dir PATH` | `out/` | Output directory |
| `--preview-percentile FLOAT` | 99.5 | White clip percentile for preview |
| `--preview-no-normalize` | false | Disable preview auto-normalise |
| `--preview-white-balance` | false | Apply white balance in preview |
| `--preview-color-correction` | false | Apply colour correction matrix in preview |
| `--emit-demosaic-linear16` | false | Also write linear (no-gamma) 16-bit demosaiced images alongside the 8-bit sRGB previews |

**Outputs written to `--out-dir`:**
- `raw_output.tif` — RAW16 Bayer mosaic
- `clean_demosaic_rgb8.png` / `noisy_demosaic_rgb8.png` — 8-bit sRGB (gamma-encoded) demosaiced previews
- `clean_demosaic_linear_rgb16.png` / `noisy_demosaic_linear_rgb16.png` — 16-bit **linear** (no gamma) demosaiced images, written only when `--emit-demosaic-linear16` is set

### Linear 16-bit demosaiced output

When `--emit-demosaic-linear16` is set, the tool writes 16-bit RGB PNGs that share the *same* black/white stretch as the 8-bit sRGB previews but omit the sRGB transfer curve. Each 16-bit value is:

```
value16 = clip( (DN − black_dn) / (white_dn − black_dn), 0, 1 ) × 65535
```

where `white_dn` is the preview percentile clip, or — when `--preview-no-normalize` is set — the full ADC scale `2^bit_depth − 1`. The stretch is therefore a fixed affine map from the ADC range to `[0, 65535]` with **no gamma**; 16-bit is used because linear light bands badly in 8-bit shadows. No white balance or colour-correction is applied unless those preview flags are enabled, so the values are in the sensor's native (un-white-balanced) colour space.

> Note: PNG's Pillow backend cannot encode 16-bit multi-channel images, so these files are written by a small built-in encoder (`_write_png16_rgb`) rather than through imageio. Read them with a 16-bit-aware loader (e.g. OpenCV, or `imageio` with the FreeImage plugin) — the default Pillow path may down-convert to 8-bit on read.

---

## `apply_spectral_psf.py` — Post-Render PSF Blur

Applies a wavelength-dependent Gaussian PSF (diffraction + geometric) and optional stray light to a multispectral PBRT EXR.

```bash
python tools/apply_spectral_psf.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--repo-root PATH` | auto | Repository root |
| `--camera-model-config PATH` | null | Camera model YAML path |
| `--exr-in PATH` | from pipeline | Input multispectral EXR |
| `--exr-out PATH` | from pipeline | Output blurred EXR |

The PSF parameters are read from `lens.post_psf` in the camera model:
- `sigma_pixels`: geometric residual blur radius
- `stray_light.veiling_glare_fraction`: uniform fog level
- `stray_light.halo_sigma_pixels` and `halo_strength`: large diffuse halo

**When to use:** Only necessary when the lens type is `pinhole` or `thinlens` (which have no physical PSF in PBRT). When using `realistic` lens type, PBRT already models the geometric PSF; diffraction blur can still be added if desired.

---

## `pbrt_spectral_exr_to_electrons.py` — EXR to Electrons

Converts PBRT `Film "spectral"` EXR output to per-pixel electron count arrays.

```bash
python tools/pbrt_spectral_exr_to_electrons.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--repo-root PATH` | auto | Repository root |
| `--camera-model-config PATH` | null | Camera model YAML |
| `--exr PATH` | `out/colorchecker_spectral.exr` | Input multispectral EXR |
| `--out-npz PATH` | `out/sensor_forward_electrons.npz` | Output electron array |
| `--exposure-time-s FLOAT` | from model | Integration time override [s] |

The conversion chain: radiance $L$ → irradiance $E = L\pi/(4N^2)$ → photon flux $\phi = E\lambda/hc$ → electrons $e = \int\phi\cdot\text{QE}\,d\lambda \cdot A_\text{pixel} \cdot t_\text{int} \cdot \text{FF}$.

---

## `spectral_sensor_forward.py` — Analytic Forward Model

Computes electron maps analytically from spectral reflectance data and camera model parameters, without PBRT.

```bash
python tools/spectral_sensor_forward.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--repo-root PATH` | auto | Repository root |
| `--config PATH` | null | Sensor-forward YAML config |
| `--camera-model-config PATH` | null | Camera model YAML (preferred) |
| `--out-npz PATH` | `out/sensor_forward_electrons.npz` | Output electron array |
| `--target-lux FLOAT` | from model | Scene illuminance override [lux] |
| `--exposure-time-s FLOAT` | from model | Integration time override [s] |

Useful for rapid noise parameter exploration without waiting for a PBRT render.

---

## `validate_emva_model.py` — EMVA Noise Validation

Validates that the Monte Carlo noise pipeline produces statistics matching the analytic EMVA 1288 predictions.

```bash
python tools/validate_emva_model.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--repo-root PATH` | auto | Repository root |
| `--config PATH` | null | YAML config |
| `--camera-model-config PATH` | null | Camera model YAML |
| `--report-json PATH` | null | Output JSON report path |
| `--trials INT` | from model | Monte Carlo trials |
| `--seed INT` | from model | RNG seed |

Produces a JSON report and PTC plot. See [Validation Guide](validation.md) for interpretation.

---

## `validate_demosaic_linear.py` — Demosaic Quality Validation

Measures PSNR and mean absolute error between demosaiced output and ground-truth noiseless image.

```bash
python tools/validate_demosaic_linear.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--repo-root PATH` | auto | Repository root |
| `--config PATH` | null | YAML config |
| `--camera-model-config PATH` | null | Camera model YAML |
| `--electrons-npz PATH` | from pipeline | Noiseless ground-truth electron array |
| `--raw-tif PATH` | from pipeline | Noisy RAW16 TIFF from noise pipeline |
| `--crop INT` | 2 | Border crop to exclude edge artefacts |
| `--json-out PATH` | null | Output metrics JSON |

---

## `validate_colorchecker.py` — ColorChecker Render Validation

Validates the spectral render by checking ColorChecker patch radiance against expected values.

```bash
python tools/validate_colorchecker.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--repo-root PATH` | auto | Repository root |
| `--render` | false | Run PBRT on the ColorChecker scene before validating |
| `--exr PATH` | `out/colorchecker.exr` | Input EXR to validate |

---

## `build_colorchecker_scene.py` — Scene Builder

Generates a PBRT scene file for the 24-patch X-Rite ColorChecker under a configurable illuminant.

```bash
python tools/build_colorchecker_scene.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--output PATH` | `scenes/generated/colorchecker.pbrt` | Output .pbrt file |
| `--film STR` | `spectral` | PBRT Film type |
| `--spectral-nbuckets INT` | 32 | Number of spectral wavelength buckets |
| `--xres INT` | 960 | Image width |
| `--yres INT` | 640 | Image height |
| `--cam-dist FLOAT` | 4.0 | Camera distance [m] |
| `--samples INT` | 64 | Pixel samples |

---

## `emva_theory.py` — Analytic EMVA Predictions

Library module (not a standalone script) providing analytic variance functions used by the validation tools.

Key functions:
- `temporal_variance_electrons_squared(mu_e, sigma_d_e, *, use_poisson, sigma_ktc_e, mu_dark_e)` — predicted temporal variance in e²
- `temporal_variance_dn_squared(...)` — same in DN²
- `monte_carlo_temporal_dn_stats(...)` — Monte Carlo estimate for comparison
- `photon_transfer_curve_checks(...)` — validate PTC shape

---

## `sensor_radiometry.py` — Radiometry Helpers

Library module providing physical constants and radiometry computations.

Key functions:
- `photon_flux_density_from_irradiance(E_lambda, lambda_nm)` — W/(m²·nm) → photons/(s·m²·nm)
- `cosine_illuminance_factor(normal, sun_direction)` — Lambert cosine factor
- `cos4_vignetting_from_pinhole(x, y, f, pixel_pitch)` — cos⁴ vignetting map

---

## `assign_lens_models.py` — Lens Model Assignment

Assigns class-appropriate lens models to all camera recipes. See [Lens Models](lens_models.md).

```bash
python tools/assign_lens_models.py [--dry-run]
```

---

## `audit_qe_import_health.py` — QE Data Audit

Checks that all camera models have valid, complete QE spectral data files.

```bash
python tools/audit_qe_import_health.py
```

---

## `batch_validate_emva.py` — Batch EMVA Validation

Runs `validate_emva_model.py` across all (or a filtered subset of) camera models and produces a summary report.

```bash
python tools/batch_validate_emva.py [--filter PATTERN]
```

---

## `camera_model.py` — Camera Model Loader

Library module (not standalone) providing `load_camera_model(name)` which loads, deep-merges, and validates a camera model YAML. Used by all pipeline tools.

---

## `split_camera_models.py` — Monolithic → Recipe Split

Splits legacy monolithic camera model YAMLs into the three-layer recipe/sensor/lens structure.

```bash
python tools/split_camera_models.py [--src-dir PATH] [--lens-dir PATH] [--sensor-dir PATH] [--recipes-dir PATH] [--dry-run]
```

---

## `run_pipeline.py` — Shell Environment

`pipeline_shell_env.py` exports the pipeline config as shell environment variables for use in bash scripts:

```bash
eval $(python tools/pipeline_shell_env.py config/pipeline.yaml)
echo $PBRT_BIN   # third_party/pbrt-v4/build/pbrt
```

---

## `exr_multispectral.py` — EXR I/O Utilities

Library module providing helpers for reading/writing multispectral OpenEXR files with `S0.NNNnm` channel naming.

---

## `munsell_mat.py` / `extract_munsell_mat.py` — Munsell Spectra

Tools for loading and processing Munsell reflectance spectra from `.mat` files for use in scene building:

```bash
python tools/extract_munsell_mat.py --mat spectra/munsell/munsell.mat --out spectra/munsell/
```

---

## `build_munsell_scenes.py` / `build_straylight_test_scene.py`

Scene builders for Munsell colour patches and stray-light test targets. Usage is similar to `build_colorchecker_scene.py`.

---

## See Also

- [Quick Start](quickstart.md) — running the full pipeline
- [Pipeline Config](pipeline_config.md) — `pipeline.yaml` key reference
- [Validation](validation.md) — interpreting tool outputs
