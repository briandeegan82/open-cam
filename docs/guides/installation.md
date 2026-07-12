# Installation

This guide covers setting up the open-cam pipeline: Python environment, PBRT v4 build, and verification.

---

## System Requirements

- **OS**: Linux (Ubuntu 20.04+ recommended) or macOS 12+
- **Python**: 3.10 or newer (uses `float | None` union type syntax)
- **CMake**: 3.16+
- **C++ compiler**: GCC 9+ or Clang 12+ with C++17 support
- **GPU (optional)**: NVIDIA GPU with CUDA 11.7+ and OptiX 7.3+ for GPU rendering

---

## 1. Clone the Repository

```bash
git clone <repo-url> open-cam
cd open-cam
git submodule update --init --recursive   # pulls third_party/pbrt-v4
```

---

## 2. Python Environment

Create an isolated environment to avoid dependency conflicts:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.24 | Array maths, spectral integration |
| `scipy` | ≥1.10 | Gaussian filters (PSF), optimisation |
| `PyYAML` | ≥6.0 | Camera model and pipeline config parsing |
| `imageio` | ≥2.31 | EXR and PNG I/O |
| `OpenEXR` | ≥3.3 | Multispectral EXR channel access |
| `openpyxl` | ≥3.1 | QE workbook parsing |
| `Pillow` | ≥10.0 | Preview image encoding |
| `matplotlib` | ≥3.8 | PTC plots, validation figures |

### Verify Python Install

```bash
python -m pytest tests/ -q --ignore=tests/test_optics_psf.py
```

All tests should pass. `test_optics_psf.py` requires a numpy version fix and can be ignored initially.

---

## 3. Build PBRT v4

See [Building PBRT](building_pbrt.md) for the full build guide. Quick version:

```bash
cd third_party/pbrt-v4
mkdir build && cd build

# CPU-only build:
env -u PBRT_OPTIX_PATH cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

The binary will be at `third_party/pbrt-v4/build/pbrt`.

---

## 4. Verify Spectral Film

Check that your PBRT build has spectral film support:

```bash
third_party/pbrt-v4/build/pbrt --help 2>&1 | grep -i spectral
```

You should see `Film "spectral"` listed. If not, see [Building PBRT](building_pbrt.md) for the correct CMake flags.

---

## 5. Spectra Data

The pipeline ships with pre-interpolated spectra:

```
spectra/
    illuminant/interpolated/   # D50, D55, D65, D75 illuminants
    QE/interpolated/           # QE_red.csv, QE_green.csv, QE_blue.csv, QE_IRCF.csv
    munsell/                   # Munsell reflectance spectra
    xrite/                     # ColorChecker reflectance spectra
```

These are ready to use and do not need to be regenerated unless you add new camera spectral data.

---

## 6. Output Directory

Create the output directory before the first run:

```bash
mkdir -p out
```

The pipeline writes all outputs to `out/` by default (controlled by `paths.out_dir` in `pipeline.yaml`).

---

## 7. Validate Complete Installation

Run the full pipeline end-to-end with the default camera model and a low sample count:

```bash
# Edit pipeline.yaml to set pixelsamples: 16 for a quick test
python tools/run_pipeline.py config/pipeline.yaml
```

Expected outputs in `out/`:
- `colorchecker_spectral.exr` — multispectral render
- `sensor_forward_electrons.npz` — electron arrays
- `raw_output.tif` — RAW16 mosaic
- `preview_demosaic.png` — sRGB preview
- `emva_validation_report.json` — noise validation
- `ptc_plot.png` — photon transfer curve

If PBRT is not yet built, the analytic forward model path still runs:

```bash
# In pipeline.yaml: sensor_forward.mode: sensor_forward (not pbrt_exr)
python tools/run_pipeline.py config/pipeline.yaml
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'OpenEXR'`

OpenEXR Python bindings depend on the OpenEXR C++ library:

```bash
# Ubuntu/Debian:
sudo apt install libopenexr-dev
pip install OpenEXR

# macOS (Homebrew):
brew install openexr
pip install OpenEXR
```

### `ImportError: numpy` version issues with `np.trapezoid`

If you see `AttributeError: module 'numpy' has no attribute 'trapezoid'`, you have NumPy < 2.0. The code uses `np.trapezoid` (new name) with a fallback to `np.trapz`. Upgrade:

```bash
pip install 'numpy>=2.0'
```

### PBRT `Film "spectral"` not found

This means PBRT was built without spectral film support. See [Building PBRT](building_pbrt.md) and ensure you do **not** pass `-DPBRT_USE_COLOR_FILM=ON`.

### GPU render fails with CUDA error

- Verify OptiX SDK is installed and `PBRT_OPTIX_PATH` is set
- Confirm CUDA version matches OptiX requirements
- Fall back to CPU: set `gpu_enabled: false` in `pipeline.yaml`
