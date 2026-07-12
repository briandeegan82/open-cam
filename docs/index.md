# open-cam Documentation

**open-cam** is a physically-accurate camera simulation pipeline. It models the complete image-formation chain from scene radiance to RAW16 sensor output: spectral rendering with PBRT v4, radiometric conversion, sensor noise, CFA sampling, and demosaic. All stages are grounded in first-principles physics.

---

## Table of Contents

### Theory

Background chapters explaining the physics implemented by each pipeline stage.

| Chapter | Topic |
|---------|-------|
| [Radiometry](theory/01_radiometry.md) | Light, radiance, irradiance, the thin-lens formula, vignetting |
| [Sensor Physics](theory/02_sensor_physics.md) | Photoelectric effect, quantum efficiency, electron count integral |
| [Noise Model](theory/03_noise_model.md) | EMVA 1288 noise chain: shot noise, read noise, PRNU, DSNU, dark current, ADC |
| [Optics & PSF](theory/04_optics.md) | Diffraction, Airy disk, Gaussian PSF, chromatic aberration, stray light |
| [Colour & CFA](theory/05_color_cfa.md) | Bayer patterns, spatial crosstalk, demosaic, channel crosstalk, sRGB OETF |

### Guides

Step-by-step instructions for installation, configuration, and use.

| Guide | Topic |
|-------|-------|
| [Installation](guides/installation.md) | Python environment, PBRT build, dependencies |
| [Building PBRT](guides/building_pbrt.md) | CPU and GPU (OptiX) PBRT v4 build instructions |
| [GPU vs CPU Rendering](guides/gpu_vs_cpu_rendering.md) | Backend limitations, spectral accuracy parity, how the pipeline invokes PBRT |
| [Quick Start](guides/quickstart.md) | First run walkthrough with annotated outputs |
| [Pipeline Config](guides/pipeline_config.md) | Complete `pipeline.yaml` key reference |
| [Camera Models](guides/camera_models.md) | YAML schema, inheritance, adding new cameras |
| [Lens Models](guides/lens_models.md) | Lens prescriptions, PBRT realistic lenses, assignment rules |
| [Highway Scenes](guides/highway_scenes.md) | Autonomous-driving highway scene generator: road, cars, signs, sun+sky |
| [Tools Reference](guides/tools_reference.md) | CLI reference for every script in `tools/` |
| [Validation](guides/validation.md) | EMVA, PTC, demosaic validation — interpreting outputs |

---

## Pipeline Overview

```
Scene (PBRT .pbrt)
        │  PBRT v4 spectral render
        ▼
Spectral EXR  (S0.360nm … S0.830nm channels, L [W/(m²·sr·nm)])
        │  pbrt_spectral_exr_to_electrons.py
        ▼
Electrons NPZ  (per-pixel, per-colour-channel float32 maps)
        │  apply_spectral_psf.py  (optional chromatic PSF + stray light)
        │  apply_emva_noise.py
        ▼
RAW16 TIFF  +  preview PNG
        │  validate_colorchecker.py / validate_emva_model.py
        ▼
Validation Reports  (JSON + PNG PTC plots)
```

The orchestrator `tools/run_pipeline.py` runs all stages in order, driven by `config/pipeline.yaml` and a camera model YAML.

---

## Key Design Decisions

- **Spectral throughout**: PBRT outputs per-wavelength radiance. Electron counts are computed by integrating `QE(λ)` against the photon flux spectrum — no RGB approximation.
- **EMVA 1288 noise**: Every noise source (shot noise, read noise, PRNU, DSNU, dark current, FPN, ADC) is modelled with the statistical distribution specified by the EMVA standard.
- **Physical units everywhere**: Irradiance in W/(m²·nm), electrons as discrete integer draws, DN as integer ADC output.
- **60+ real camera models**: Parameters sourced from published datasheets, EMVA reports, or heuristic scaling rules. See `config/camera_models/EMVA_SUMMARY.md`.
