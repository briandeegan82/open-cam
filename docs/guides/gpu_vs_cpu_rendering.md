# GPU vs CPU Rendering (PBRT v4)

open-cam can render scenes with either PBRT's CPU renderer or its OptiX GPU
renderer (`--gpu`). This guide documents what actually differs between the two
paths, with a focus on **spectral rendering accuracy**, and shows how the
open-cam pipeline invokes PBRT.

All line references are to the vendored `third_party/pbrt-v4` tree.

---

## Summary

- **Spectral accuracy is identical** between CPU and GPU to floating-point
  noise level. Both use the same 4-wavelength hero sampling, the same visible
  importance distribution, the same dispersion handling, and the same
  `SpectralFilm` bucket accumulation — compiled from one `PBRT_CPU_GPU`
  codebase.
- The real lever on spectral quality is **samples per pixel relative to the
  number of output buckets**, not the choice of backend.
- The GPU path has **feature limitations** relative to the CPU (integrator,
  materials, textures, samplers, memory). None of them are triggered by the
  scenes open-cam currently generates.

---

## GPU limitations relative to CPU

These are inherent to PBRT v4, verified against the source.

| Area | Limitation | Source |
|------|-----------|--------|
| Hardware | NVIDIA only; requires CUDA 11.0+ and OptiX 7.1+; GPU must support unified addressing | `README.md:204`, `gpu/util.cpp:91` |
| Memory | Scene must fit in GPU-addressable memory — no out-of-core streaming | (unified-memory design) |
| Integrator | `--gpu` **always** uses the wavefront path integrator; the scene's `Integrator` directive is ignored. `bdpt`, `mlt`, `sppm`, and the non-wavefront `path`/`volpath` variants are CPU-only | `cmd/pbrt.cpp:285` |
| Sampler | Every sampler works on GPU **except** `MLTSampler`/`DebugMLTSampler` | `wavefront/samples.cpp:22`, `samplers.cpp:352` |
| Materials | `MixMaterial` `amount` accepts only *basic* textures on GPU (mixing is resolved in the closest-hit shader) | `materials.cpp:116` |
| Textures | Ptex textures are unsupported on GPU | `textures.cpp:700` |
| Options | `--pixelmaterial` forces CPU (silently disables `--gpu`) | `cmd/pbrt.cpp:245` |

> **The integrator override is the one behavioral difference worth
> remembering.** Under `--gpu`, a scene that declares `Integrator "path"` is
> actually rendered by the wavefront volumetric path integrator (honoring
> `maxdepth`). For non-volumetric diffuse scenes — the ColorChecker and Munsell
> charts open-cam renders — the wavefront `volpath` and the CPU `path`
> estimators produce the same result, so this is benign. It only matters if the
> scene set grows to include participating media, or if a bidirectional method
> (`bdpt`/`mlt`) is needed for a reference image, in which case those must run
> on the CPU.

---

## Spectral accuracy: why CPU and GPU agree

PBRT v4 is a spectral renderer on **both** backends. The spectral machinery is
not reimplemented for the GPU — it is one implementation compiled for both, and
the GPU path stores the same spectral types the CPU uses.

- **Sample count.** `NSpectrumSamples = 4` is a global `constexpr`
  (`util/spectrum.h:36`), not a per-backend setting. Every path carries 4
  wavelengths (one hero + three secondary).
- **GPU carries the real types.** The wavefront work items store
  `SampledSpectrum` and `SampledWavelengths` directly
  (`wavefront/workitems.h:110-156`) — the same classes the CPU uses.
- **Same wavelength sampling.** GPU camera rays call `film.SampleWavelengths()`
  (`wavefront/camera.cpp:57`), which resolves to the shared visible-importance
  sampler `SampleVisible` (`util/spectrum.h:331`). All marked `PBRT_CPU_GPU`.
- **Same dispersion handling.** Chromatic dispersion collapses to the hero
  wavelength via `TerminateSecondary`, which is called from inside the shared
  material code (`materials.h:187,232`, `materials.cpp:274,361`) — e.g. a
  dielectric with a wavelength-dependent IOR. The wavefront path respects the
  flag (`wavefront/surfscatter.cpp:140`). So dispersive materials render the
  same on both backends.

### The only difference: floating point

The GPU accumulates results with atomics in a nondeterministic order and uses
different FMA/intrinsics, so per-pixel spectral values differ from a CPU render
at roughly the ULP/noise level. This is **stochastic, not a systematic color
bias** — both paths converge to the same image as spp increases.

**Implication for validation:** when comparing a CPU render against a GPU render
of the same scene, compare at matched (and reasonably high) spp and use a
tolerance rather than exact equality. Small non-zero per-pixel differences that
shrink with more samples are expected, not a bug.

---

## The real accuracy lever: spp vs. buckets

`SpectralFilm::AddSample` (`film.h:449-454`) splats each of the 4 hero
wavelengths into whichever output bucket it falls in, then averages:

```cpp
for (int i = 0; i < NSpectrumSamples; ++i) {   // NSpectrumSamples == 4
    int b = LambdaToBucket(lambda[i]);
    pixel.bucketSums[b]  += L[i];
    pixel.weightSums[b]  += weight;            // bucket = sum / weight
}
```

Each sample therefore fills only **4 of N** output buckets. A bucket sees
roughly `spp × 4 / N` contributions:

| spp | 32 buckets | 64 buckets |
|-----|-----------|-----------|
| 128 (`--smoke`) | ~16 / bucket | ~8 / bucket (noisy) |
| 1024 (dataset default) | ~128 / bucket | ~64 / bucket |
| 4096 (dual-munsell) | ~512 / bucket | ~256 / bucket |

This per-bucket variance is identical on CPU and GPU (shared `film.h` code). If
individual wavelength channels look noisy, the fix is **more spp or fewer
buckets** — not switching backends. For photon-transfer-curve validation, keep
render noise below sensor noise (256+ spp recommended).

---

## How the open-cam pipeline invokes PBRT

The dataset generators invoke PBRT on the GPU
(`scripts/generate_dataset.py:141`, `scripts/generate_dual_munsell.py:142`):

```
pbrt --gpu <scene>.pbrt
```

Sample count is **not** passed on the command line; it is baked into the scene
via the sampler directive (`tools/build_munsell_scenes.py:317`), which is
equivalent to `--spp`. The `run_pipeline.py` orchestrator adds `--gpu`
automatically when `render.gpu_enabled: true` in `config/pipeline.yaml`.

### Backend-relevant scene settings

Generated by `tools/build_munsell_scenes.py`:

| Directive | Value | GPU status |
|-----------|-------|-----------|
| `Sampler "zsobol"` | `pixelsamples = spp` | ✅ Supported (only MLT samplers are excluded) |
| `Integrator "path" maxdepth 6` | — | ⚠️ Ignored under `--gpu`; wavefront runs instead (benign for these scenes) |
| `Film "spectral"` | 64 buckets, 360–830 nm, `savefp16 false` | ✅ Shared `SpectralFilm`, float32 output |
| Materials | `spectrum reflectance` (SPD files), `spectrum L` illuminants | ✅ Basic spectra — no Ptex, no procedural `MixMaterial` amount |

None of the GPU feature limitations are triggered: the scenes use a
GPU-supported sampler, spectral SPD-based materials, and the spectral film.

### Downstream is backend-agnostic

`tools/pbrt_spectral_exr_to_electrons.py` reads the `S0.<lambda>` spectral
channels and integrates them against `QE(λ)`. It does not depend on whether the
EXR was produced on the CPU or GPU, so nothing in the wrapper reintroduces a
CPU/GPU difference.

---

## Practical guidance

- **Prefer GPU for throughput.** Spectral accuracy is not a reason to prefer the
  CPU for open-cam's chart scenes.
- **Validate with a tolerance**, not exact equality, when diffing CPU vs GPU
  renders.
- **Tune spp against bucket count** if spectral channels are noisy.
- **Fall back to CPU** if you ever add participating media, need a
  `bdpt`/`mlt`/`sppm` reference, or introduce Ptex / procedural `MixMaterial`
  amount textures — those are CPU-only.

---

## Reference

- [Building PBRT v4](building_pbrt.md) — build and toolchain details
- [PBRT v4 GitHub](https://github.com/mmp/pbrt-v4)
- Pharr, Jakob & Humphreys, *Physically Based Rendering* (4th ed.), Ch. 15
  "Wavefront Rendering on GPUs"
