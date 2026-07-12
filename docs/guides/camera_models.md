# Camera Models — YAML Schema and Inheritance

This guide explains the camera model YAML system: the schema, inheritance mechanism, per-camera recipe files, and how to add a new camera.

---

## Architecture Overview

Camera configuration is split across three layers:

```
config/camera_models/default.yaml       ← root defaults (all parameters)
config/camera_models/<name>.yaml        ← per-camera overrides (single sensor model)
config/camera_recipes/<name>.yaml       ← recipe: references lens + sensor models
config/sensor_models/<name>.yaml        ← sensor parameters only
config/lens_models/<name>.yaml          ← lens parameters only
```

The pipeline loads the camera model by name from `paths.camera_model_name` in `pipeline.yaml`. The file `config/camera_models/<name>.yaml` is deep-merged on top of `config/camera_models/default.yaml`.

---

## Deep-Merge Inheritance

All camera models inherit from `default.yaml`. The merge is **deep** — nested keys are merged recursively, not replaced entirely. Only keys that differ from the default need to be specified.

Example: if `default.yaml` has:

```yaml
noise:
  emva:
    sigma_d_e: 2.6
    prnu_std_fraction: 0.01
```

and `iphone_8.yaml` has:

```yaml
noise:
  emva:
    sigma_d_e: 1.6
```

the merged model has `sigma_d_e: 1.6` and `prnu_std_fraction: 0.01` (inherited).

---

## Camera Model Schema

### `model`

```yaml
model:
  name: iphone_8
  display_name: iPhone 8
```

### `lens`

```yaml
lens:
  camera: realistic                        # pinhole | thinlens | realistic
  realistic_lensfile: config/lenses/wide_22mm.dat
  realistic_aperture_diameter_mm: 4.0
  realistic_focus_distance: 1.0
  post_psf:
    enabled: false
    mode: gaussian
    sigma_pixels: 0.75                     # geometric residual blur [pixels]
    stray_light:
      enabled: false
      veiling_glare_fraction: 0.01
      halo_sigma_pixels: 8.0
      halo_strength: 0.04
```

### `sensor`

```yaml
sensor:
  quantum_efficiency:
    red_csv: spectra/QE/interpolated/QE_red.csv
    green_csv: spectra/QE/interpolated/QE_green.csv
    blue_csv: spectra/QE/interpolated/QE_blue.csv
    ircf_csv: spectra/QE/interpolated/QE_IRCF.csv
  pixel_pitch_um: 1.4           # pixel pitch [µm]
  integration_time_s: 0.01      # sensor integration time [s]
  f_number: 2.0                 # lens f-number
  fill_factor: 0.95             # fraction of pixel area that is photosensitive
```

### `noise.emva` — EMVA 1288 Parameters

| Key | Unit | Description |
|-----|------|-------------|
| `overall_system_gain_K_e_per_DN` | e⁻/DN | System gain $K$ |
| `sigma_d_e` | e⁻ | Temporal dark noise (read noise) |
| `black_level_DN` | DN | ADC black level offset |
| `use_poisson_shot_noise` | bool | Enable Poisson shot noise draw |
| `prnu_std_fraction` | fraction | PRNU standard deviation |
| `prnu_std_fraction_r/g/b` | fraction | Per-channel PRNU (optional; defaults to `prnu_std_fraction`) |
| `dsnu_std_e` | e⁻ | DSNU standard deviation |
| `dsnu_mean_e` | e⁻ | DSNU mean (defaults to `dsnu_std_e` if absent) |
| `dark_current_e_per_s` | e⁻/s | Dark current at reference temperature |
| `dark_current_reference_temp_c` | °C | Reference temperature for dark current |
| `temperature_c` | °C | Sensor operating temperature |
| `dark_activation_energy_eV` | eV | Arrhenius activation energy (0.63 eV for Si). Set 0.0 for legacy doubling rule |
| `dark_current_doubling_per_c` | °C | Dark current doubling temperature (used only when `dark_activation_energy_eV: 0.0`) |
| `row_fpn_std_e` | e⁻ | Row fixed pattern noise standard deviation |
| `column_fpn_std_e` | e⁻ | Column fixed pattern noise standard deviation |
| `ktc_noise.enabled` | bool | Enable kTC reset noise (for sensors without CDS) |
| `ktc_noise.node_capacitance_fF` | fF | Optional; estimated from K if absent |

### `noise.adc` — ADC Parameters

| Key | Unit | Description |
|-----|------|-------------|
| `bit_depth` | bits | ADC resolution (8–16) |
| `full_well_e` | e⁻ | Full well capacity (hard clip level) |
| `clipping` | string | `hard` (only option currently) |
| `inl_quadratic_fraction` | fraction | Peak INL as fraction of full scale |
| `dnl_std_lsb` | LSB | DNL random variation standard deviation |

### `cfa` — Colour Filter Array

```yaml
cfa:
  enabled: true
  pattern: GBRG            # RGGB | BGGR | GRBG | GBRG
  demosaic: bilinear
  demosaic_srgb: true      # apply sRGB OETF to preview output
  spatial_crosstalk:
    enabled: true
    sigma_pixels: 0.5      # scalar or use per-channel keys below
    sigma_pixels_r: 0.6    # optional: red channel crosstalk
    sigma_pixels_g: 0.4
    sigma_pixels_b: 0.3
```

### `sensor_forward.model` — Analytic Forward Model

```yaml
sensor_forward:
  model:
    electrons_scale: 25000.0
    include_surround: true
    surround_reflectance: 0.22
    calibration:
      mode: photon_counting         # photon_counting | photopic_lux
      irradiance_scale_W_m2nm_per_unit: 0.001
      illuminant_override_csv: spectra/illuminant/interpolated/D55.csv
      target_illuminance_lux: 500.0
      optics_transmittance: 1.0
      use_aperture_factor: true
    cosine_shading: true
    chart_normal: [0.0, 0.0, 1.0]
    vignetting_cos4: false
    optics_transmittance_spatial:
      enabled: false
      mode: radial_power
      edge_factor: 0.85
      exponent: 2.0
    pbrt_spectral_exr:
      radiance_to_irradiance: thin_lens
      radiance_to_irradiance_scale: null
      extra_irradiance_scale: 1.0
      radiometric_autocalibration: mean_photopic_lux
```

### `validation` — Noise Validation Tolerances

```yaml
validation:
  monte_carlo_trials: 20000
  random_seed: 0
  variance_rtol: 0.07          # relative tolerance for variance check (7%)
  mean_abs_dn_atol: 0.22       # absolute tolerance for mean DN check
  ptc_mu_e_levels:             # signal levels [e-] for PTC
    - 0.0
    - 400.0
    - 2000.0
    - 3500.0
    - 4700.0
  datasheet:
    enabled: true
    overall_system_gain_K_e_per_DN: 4.77
    temporal_dark_noise_sigma_d_e: 2.6
    full_well_e: 4800
    black_level_DN: 16
    parameter_rtol: 0.02
```

---

## Camera Recipe Files

Camera recipes in `config/camera_recipes/` are lightweight files that reference lens and sensor models by name, plus supply any EMVA parameters:

```yaml
schema_version: 1
model:
  name: iphone_8
  display_name: iPhone 8
lens_model: phone_wide_f18          # references config/lens_models/phone_wide_f18.yaml
sensor_model: iphone_8              # references config/sensor_models/iphone_8.yaml
source:
  emva_param_method: inferred_web_table_v1
  sensor_class: phone
  calibration_tier: inferred
```

---

## EMVA Parameter Sources

Camera models use one of three parameter source tiers:

| Tier | `source.calibration_tier` | Description |
|------|--------------------------|-------------|
| Verified | `measured` | Parameters from published EMVA 1288 report |
| Inferred | `inferred` | Parameters estimated from public datasheets + scaling |
| Default | `manual_or_default` | Generic class defaults from heuristic rules |

The `strict_physical_accuracy.calibration_tier_policy` in `pipeline.yaml` controls whether low-tier cameras cause a warning or error.

---

## Adding a New Camera

1. **Create the camera model YAML** at `config/camera_models/<name>.yaml`:

```yaml
model:
  name: my_new_camera
  display_name: My Camera Model
sensor:
  pixel_pitch_um: 3.5
  f_number: 2.8
  fill_factor: 0.62
noise:
  emva:
    overall_system_gain_K_e_per_DN: 1.2
    sigma_d_e: 2.8
    black_level_DN: 256
    prnu_std_fraction: 0.012
    dsnu_std_e: 0.5
    dark_current_e_per_s: 0.3
  adc:
    bit_depth: 14
    full_well_e: 20000
```

2. **Set in `pipeline.yaml`**:

```yaml
paths:
  camera_model_name: my_new_camera
```

3. **Run validation**:

```bash
python tools/validate_emva_model.py --camera my_new_camera
```

### Finding EMVA Parameters

Primary sources in order of reliability:
1. Published EMVA 1288 reports (manufacturer or third-party test labs)
2. DxOMark sensor test data
3. Camera review site measurements (DPReview, Photons to Photos)
4. Scaling heuristics from `tools/split_camera_models.py`

Key parameter relationships:
- $K \approx V_\text{ref} / (2^N \cdot \text{FW})$ where $V_\text{ref}$ is ADC reference voltage
- $\sigma_d \approx 1.5\text{–}3.5$ e⁻ for modern BSI CMOS
- $\text{FW} \approx 1000\text{–}3000 \cdot p^2$ [e⁻] where $p$ is pitch in µm
- PRNU ≈ 0.5–2% for consumer sensors

---

## See Also

- `config/camera_models/EMVA_SUMMARY.md` — parameter table for all 60+ cameras
- [Noise Model Theory](../theory/03_noise_model.md) — what each parameter means physically
- [Lens Models](lens_models.md) — lens assignment and prescription files
- `tools/audit_qe_import_health.py` — check QE file coverage
