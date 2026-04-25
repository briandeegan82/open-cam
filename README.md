# Open Cam

Open Cam is a physically motivated camera simulation pipeline. It renders a ColorChecker scene with PBRT, converts spectral radiance into sensor electrons, and applies EMVA-style sensor noise and optional CFA/demosaic to generate RAW and preview outputs.

## What You Get

- Spectral or RGB PBRT render (`.exr`)
- Optional electrons intermediate (`.npz`)
- Noisy RAW Bayer output (`.raw16`)
- Preview PNGs (clean/noisy, optional demosaic)
- Validation reports and run manifest JSON files

Main output folder: `out/`

## Quick Start (Recommended)

1. Create Python environment and install dependencies:

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

2. Build PBRT (see `docs/BUILD_PBRT.txt` for full details):

```bash
cd third_party/pbrt-v4
git submodule update --init --recursive
env -u PBRT_OPTIX_PATH cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
cd ../..
```

3. Run the full pipeline:

```bash
venv/bin/python tools/run_pipeline.py --config config/pipeline.yaml
```

4. Inspect key outputs:

- `out/colorchecker_noisy_png/noisy_demosaic_rgb8.png`
- `out/colorchecker_noisy_png/clean_demosaic_rgb8.png`
- `out/colorchecker_noisy_png/run_stats.json`
- `out/run_pipeline_<timestamp>.json`

## Requirements

- Linux/macOS with Python 3
- PBRT binary at `third_party/pbrt-v4/build/pbrt`
- Python packages in `requirements.txt`:
  - `numpy`, `scipy`, `PyYAML`, `imageio`, `OpenEXR`, `openpyxl`

`OpenEXR` is required for multispectral EXR channel handling (`S0.*nm`).

## How the Pipeline Works

`tools/run_pipeline.py` orchestrates these stages from `config/pipeline.yaml`:

1. Build scene: `tools/build_colorchecker_scene.py`
2. Render with PBRT
3. Optional post-PSF blur: `tools/apply_spectral_psf.py`
4. Render validation: `tools/validate_colorchecker.py`
5. Optional EMVA parameter validation: `tools/validate_emva_model.py`
6. Optional spectral-to-electrons forward model:
   - `tools/spectral_sensor_forward.py` (`analytic`)
   - `tools/pbrt_spectral_exr_to_electrons.py` (`pbrt_exr`)
7. Noise + CFA + preview generation: `tools/apply_emva_noise.py`
8. Optional demosaic metrics: `tools/validate_demosaic_linear.py`

## Common Commands

### Full run

```bash
venv/bin/python tools/run_pipeline.py --config config/pipeline.yaml
```

### GPU render (PBRT stage)

Enable GPU for the PBRT render stage via `render.gpu_enabled`:

```yaml
render:
  gpu_enabled: true
  pbrt_args: ["--stats"]
```

This causes `tools/run_pipeline.py` to invoke PBRT with `--wavefront` (unless already present in `render.pbrt_args`).

Use `--dry-run` to confirm command wiring before launching a full run:

```bash
venv/bin/python tools/run_pipeline.py --config config/pipeline.yaml --dry-run
```

See `docs/BUILD_PBRT.txt` for GPU build prerequisites (CUDA + OptiX).

### Dry run (show commands only)

```bash
venv/bin/python tools/run_pipeline.py --config config/pipeline.yaml --dry-run
```

### Bash runner (same pipeline)

```bash
scripts/generate_colorchecker_image.sh 0
```

Use `PIPELINE_CONFIG=<path>` to point to a custom pipeline YAML.

### Build image-quality targets

Generate and render additional targets (slanted edge, ISO-style grayscale patches, Siemens star):

```bash
scripts/generate_iq_targets.sh all
```

Run one target through the full camera pipeline (render + optional post-PSF + optional sensor-forward + EMVA noise), using the same config flow as `generate_colorchecker_image.sh`:

```bash
scripts/generate_slanted_edge_image.sh 0
scripts/generate_iso_noise_image.sh 0
scripts/generate_siemens_star_image.sh 0
```

Equivalent generic entrypoint:

```bash
scripts/generate_iq_target_image.sh slanted_edge 0
```

Generate only one target:

```bash
scripts/generate_iq_targets.sh slanted_edge
```

If you only want scene files (no render), call the builder directly:

```bash
venv/bin/python tools/build_image_quality_targets.py --target all
```

Notes:

- Full per-target outputs are stored under `out/iq_targets/`.
- The full per-target pipeline uses `sensor_forward.mode: pbrt_exr` for scene-specific electrons generation.
- `EMVA_FROM_EXR=1` forces `tools/apply_emva_noise.py` to run without `--electrons-npz`.

### Standalone tools (explicit camera model)

When running stage tools directly, always pass `--camera-model-config` so behavior is deterministic.

```bash
venv/bin/python tools/apply_emva_noise.py \
  --repo-root . \
  --camera-model-config config/camera_recipes/nikon_z6.yaml \
  --linear-exr out/colorchecker_spectral.exr
```

```bash
venv/bin/python tools/spectral_sensor_forward.py \
  --repo-root . \
  --camera-model-config config/camera_recipes/nikon_z6.yaml
```

## Configuration Guide

### `config/pipeline.yaml` (run-level orchestration)

Key sections:

- `paths.*`: tool paths, outputs, and camera recipe selection
- `render.*`: film mode, resolution, light scale, camera render options
- `sensor_forward.*`: enable/disable and `mode` (`analytic` or `pbrt_exr`)
- `noise.*`: EMVA stage toggles and preview controls
- `validate*.*`: per-stage validation toggles

Camera selection rules:

- Set exactly one of:
  - `paths.camera_model_name`
  - `paths.camera_model_config`
- `--camera-model-config` on CLI overrides pipeline selection.

### Camera recipes and model composition

- `config/camera_recipes/*.yaml` selects a lens model + sensor model.
- `config/lens_models/*.yaml` defines optics and optional post-PSF.
- `config/sensor_models/*.yaml` defines QE, sensor/noise, CFA, ADC, and forward-model settings.

## Key Output Files

- Render EXR: `paths.exr_out` (default `out/colorchecker_spectral.exr`)
- Electrons NPZ: `paths.sensor_forward_electrons_npz`
- RAW output: `out/colorchecker_noisy.raw16`
- Preview images and stats: `out/colorchecker_noisy_png/`
- EMVA validation report: `out/emva_validation_report.json`
- Demosaic metrics: `out/demosaic_linear_metrics.json`
- Run manifest: `out/run_pipeline_<timestamp>.json`
- IQ-target outputs (per target): `out/iq_targets/<target>_noisy.raw16`, `out/iq_targets/<target>_noisy_png/`, `out/iq_targets/<target>_electrons.npz`

## Typical Workflows

### Compare clean vs noisy

After a run, compare:

- `out/colorchecker_noisy_png/clean_demosaic_rgb8.png`
- `out/colorchecker_noisy_png/noisy_demosaic_rgb8.png`

### Switch camera quickly

In `config/pipeline.yaml`:

```yaml
paths:
  camera_model_name: nikon_z6
```

### Use analytic forward model with RGB render

In `config/pipeline.yaml`:

- `render.film: rgb`
- `sensor_forward.mode: analytic`
- set `paths.exr_out` accordingly (for example `out/colorchecker.exr`)

### Override the scene illuminant spectrum

In `config/pipeline.yaml`:

- `render.illuminant: spectra/illuminant/interpolated/D65.csv`

This is passed to scene generation as `build_colorchecker_scene.py --illuminant ...`.

## Validation and Tests

Run unit tests:

```bash
PYTHONPATH=tools:. venv/bin/python -m unittest discover -s tests -v
```

Useful validation commands:

```bash
venv/bin/python tools/validate_colorchecker.py --repo-root . --exr out/colorchecker_spectral.exr
```

```bash
venv/bin/python tools/validate_emva_model.py \
  --repo-root . \
  --camera-model-config config/camera_recipes/nikon_z6.yaml \
  --json-out out/emva_validation_report.json
```

## Munsell Dataset Tools (Optional)

Optional scripts for Joensuu Munsell matte data:

- `tools/munsell_mat_to_sqlite.py`
- `tools/extract_munsell_mat.py`
- `tools/build_munsell_scenes.py`

Quick example:

```bash
venv/bin/python tools/munsell_mat_to_sqlite.py --summary
```

## Known Limitations

- `pbrt_exr` sensor-forward mode expects multispectral EXR and currently supports `photon_counting` calibration mode.
- `cfa.demosaic` currently implements bilinear demosaic only.
- Post-PSF is a simple Gaussian/stray-light approximation, not a full chromatic lens aberration model.
- PBRT spectral sampling remains Monte Carlo based; use validation and repeated runs for sensitive comparisons.

## Troubleshooting

- Missing PBRT binary: build PBRT and verify `third_party/pbrt-v4/build/pbrt`.
- Missing OpenEXR support: reinstall deps with `venv/bin/pip install -r requirements.txt`.
- Pipeline command mismatch: run with `--dry-run` and inspect generated command list.
- Unexpected camera config behavior: ensure only one of `paths.camera_model_name` and `paths.camera_model_config` is set.

## Contributing

Before opening a PR:

- `PYTHONPATH=tools:. venv/bin/python -m unittest discover -s tests -v`
- `venv/bin/python tools/run_pipeline.py --config config/pipeline.yaml --dry-run`

Keep generated artifacts (`out/`, `scenes/generated/`) out of commits unless a change explicitly requires them.
