# Building PBRT v4

PBRT v4 is the physically-based ray tracer used by open-cam for spectral scene rendering. It is included as a vendored clone at `third_party/pbrt-v4/`.

---

## Prerequisites

### All platforms
- CMake ≥ 3.16
- C++17-capable compiler (GCC 9+, Clang 12+, MSVC 2019+)
- Git (for submodules)

### GPU build (optional)
- NVIDIA GPU with compute capability ≥ 6.0
- CUDA Toolkit 11.7+
- [OptiX SDK](https://developer.nvidia.com/optix) 7.3+ (requires free NVIDIA developer account)

---

## Step 1: Initialise Submodules

PBRT has its own submodule dependencies (e.g. `glfw`, `nanovdb`):

```bash
cd third_party/pbrt-v4
git submodule update --init --recursive
```

> **Note**: `third_party/pbrt-v4` is a vendored clone, not a git submodule of open-cam. Initialise its own submodules from within its directory.

---

## Step 2: CPU-Only Build

If `PBRT_OPTIX_PATH` is set in your environment but OptiX headers are not installed, CMake will enable GPU targets and the build fails. Always explicitly unset it for CPU builds:

```bash
cd third_party/pbrt-v4
env -u PBRT_OPTIX_PATH cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

The compiled binary will be at:

```
third_party/pbrt-v4/build/pbrt
```

---

## Step 3: GPU Build (Optional)

Install the OptiX SDK and set the path, then configure with an explicit CUDA
toolkit and host compiler:

```bash
export PBRT_OPTIX_PATH=$HOME/NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64  # your install

cd third_party/pbrt-v4
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12
cmake --build build -j$(nproc)
```

To verify GPU support is compiled in, check that `--gpu` is listed:

```bash
third_party/pbrt-v4/build/pbrt --help 2>&1 | grep -- --gpu
```

If `--gpu` does not appear, the binary is a CPU-only build (OptiX was not found
at configure time). Note: `--wavefront` is present even in CPU-only builds and
runs the wavefront integrator on the CPU — it does **not** use the GPU.

### CUDA toolkit selection (important)

The CUDA toolkit version matters, and the failure modes are non-obvious:

- **CUDA 12.2–12.8**: the build fails with `error: '::cuda' has not been
  declared` in `wavefront/media.cpp`, `surfscatter.cpp`, `subsurface.cpp`. This
  is a libcu++/CCCL regression — under `-rdc=true`, nvcc emits host-side
  registration for device-only `cuda::std` CPO constants (pulled in via
  `<cuda/atomic>` in `wavefront/workqueue.h`). Not fixable by changing the host
  compiler.
- **CUDA 12.0**: builds cleanly, but generates PTX `.version 8.0`. On a recent
  driver (e.g. R595 / CUDA 13.x era) OptiX rejects it at runtime with
  `OptiX: COMPILER: Invalid PTX input: Failed to parse input PTX string`.
- **CUDA 13.2** (recommended here): builds cleanly *and* emits PTX `.version 9.2`
  that the recent driver's OptiX accepts. Use `g++-12` as the CUDA host compiler.

Rule of thumb: use a CUDA toolkit new enough that its PTX ISA matches your
driver, and avoid the 12.2–12.8 range. Confirm the embedded PTX version with:

```bash
grep -m1 '^\.version' build/CMakeFiles/optix.cu.dir/src/pbrt/gpu/optix/optix.ptx
```

A stale `PBRT_OPTIX_PATH` pointing at a non-existent directory will make CMake
enable GPU targets that then fail to compile (`fatal error: optix.h: No such
file or directory`). Either point it at a real SDK or `env -u PBRT_OPTIX_PATH`
for CPU builds.

---

## Step 4: Verify Spectral Film

The open-cam pipeline requires `Film "spectral"` (OpenEXR multispectral output). Verify it is available:

```bash
third_party/pbrt-v4/build/pbrt --help 2>&1 | head -30
```

Do **not** pass `-DPBRT_USE_COLOR_FILM=ON` — this disables the spectral film and replaces it with an RGB film only.

---

## Spectral Film Notes

- PBRT's `Film "spectral"` outputs one EXR channel per wavelength bucket, named `S0.360nm`, `S0.361nm`, …
- Values are in **radiance units** [W/(m²·sr·nm)] integrated over the bucket width, normalised by the film's sensor response
- The wavelength range is controlled in the scene file with `Option "lambda_min" … "lambda_max" …` (defaults 360–830 nm)
- RGB scene inputs (colours, textures) are interpreted in the `ColorSpace "srgb"` unless another is specified with the `ColorSpace` directive before `Film`
- Piecewise-linear spectra use wavelengths in nanometres; PBRT clips to the active spectral range

### Spectral Integration in PBRT

PBRT integrates radiance over each spectral bucket using a hero-wavelength stratified sampler. With `pixelsamples: 64` and `spectral_nbuckets: 32`, each bucket gets approximately 2 samples per pixel on average. For photon-transfer-curve validation, 256+ samples per pixel is recommended to keep render noise below sensor noise.

---

## Pipeline Configuration

The PBRT binary path and render settings are in `config/pipeline.yaml`:

```yaml
paths:
  pbrt: third_party/pbrt-v4/build/pbrt

render:
  film: spectral
  spectral_nbuckets: 32
  spectral_lambda_min: 360.0
  spectral_lambda_max: 830.0
  pixelsamples: 64
  gpu_enabled: true            # adds --gpu automatically (needs OptiX build)
  pbrt_args: ["--stats"]       # extra args; --gpu added by pipeline if gpu_enabled
```

When `gpu_enabled: true`, `run_pipeline.py` automatically prepends `--gpu` to the PBRT command (unless it is already in `pbrt_args`), selecting pbrt's OptiX GPU wavefront renderer. This requires a pbrt binary built with OptiX (Step 3); a CPU-only build will reject `--gpu`.

---

## Building Scene Files

Scene `.pbrt` files are generated by Python tools:

```bash
# ColorChecker test scene
python tools/build_colorchecker_scene.py --output scenes/generated/colorchecker.pbrt

# IQ targets, Munsell patches, etc.
bash scripts/generate_iq_target_image.sh
```

---

## Running PBRT Manually

```bash
# Render with CPU
third_party/pbrt-v4/build/pbrt scenes/generated/colorchecker.pbrt

# Render with GPU (requires an OptiX-enabled build)
third_party/pbrt-v4/build/pbrt --gpu scenes/generated/colorchecker.pbrt

# Render with stats output
third_party/pbrt-v4/build/pbrt --stats scenes/generated/colorchecker.pbrt
```

The output EXR is written to the path specified by `Film "outfile"` in the `.pbrt` scene file.

---

## Reference

- [PBRT v4 file format](https://pbrt.org/fileformat-v4.html)
- [PBRT v4 GitHub](https://github.com/mmp/pbrt-v4)
- [OptiX SDK downloads](https://developer.nvidia.com/optix)
