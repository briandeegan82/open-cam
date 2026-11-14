# Opt-in Monte Carlo denoiser (display / demo renders only)

`tools/denoise_spectral_exr.py` runs [Intel Open Image Denoise](https://www.openimagedenoise.org) (OIDN,
Apache-2.0) on the pbrt spectral EXR, between pbrt and the sensor model. It removes the *renderer's* Monte Carlo
noise and fireflies; the sensor model then adds physical shot, read and dark noise as usual.

**It is off by default and must stay off for IQ measurements.** OIDN is a learned filter. It keeps the mean
level and hard edges, but it cannot tell fine stochastic texture from render noise. On the IQ-lab dead-leaves card
it removes about half of the 1–3 px detail, so texture MTF, SNR and colour-noise numbers from a denoised render
are wrong. Use it for previews, figures and demos, and say in the caption that the render was denoised.

## Why

In the IQ-lab diorama (window and 2 cm practical lamp on, `wide_22mm` realistic lens, 256 spp) the pbrt render
noise on the flat wall and floor is about 10 % pixel to pixel, while the sensor shot noise there is about 5 %.
Most of the render noise is fireflies: the worst 1 % of pixels carry 60–99 % of the variance. They come from
specular paths (glass/metal spheres) that reach the small lamp. pbrt's `maxcomponentvalue` clamp removes them but
darkens the image by 14–67 %, so it is not used. Doubling the spp only reduces the noise by √2.

## Install and run

```bash
venv/bin/python tools/fetch_oidn.py            # Linux x86-64: pinned, SHA-256-checked release -> third_party/oidn/
```

Other platforms: install OIDN 2.x and set `$OPENCAM_OIDN` to the `oidnDenoise` binary (or pass `--oidn`).

Pipeline (`config/pipeline.yaml`):

```yaml
render:
  denoise:
    enabled: true
    guides: albedo_normal  # albedo_normal | albedo | none
    guide_spp: 16
    quality: high          # high | balanced | fast
    device: cpu
    bypass: 2.0            # lamps above 2x the 99th percentile skip the denoiser (0 = off)
```

The stage denoises `exr_out` in place, keeps the noisy render as `<stem>_noisy.exr` and writes
`<exr>.denoise.json` (`"denoised": true`, energy ratio, settings). It runs before post-PSF, traced ghosts and the
sensor model, so those stay physical. Standalone:

```bash
venv/bin/python tools/denoise_spectral_exr.py --exr-in out/x.exr --scene scenes/generated/x.pbrt  # -> out/x_denoised.exr
```

## How it works

- **Guides.** The scene is re-rendered at `guide_spp` with pbrt's `gbuffer` film (the dispersion split is turned
  off for this pass) to get albedo and camera-space normals. This lets OIDN keep texture and geometry edges.
  `guides: none` skips that render and needs no scene.
- **Spectral buckets.** The `S0.<λ>nm` buckets are denoised three at a time, sorted by wavelength. OIDN sees
  each triplet as one colour image. `R,G,B` are denoised as one image; other channels are copied unchanged.
  Negative samples are clipped to zero first (the fraction is in the report).
- **Lamp bypass.** Light above `bypass` x the 99th percentile of the bucket sum (default 2) skips the denoiser and
  is added back unchanged, the same fraction in every channel of the pixel. Lamps of 1-3 px are converged, and
  a learned denoiser treats them as fireflies (see "Bright lamps" below). `bypass: 0` hands everything to OIDN.
- **Device.** `oidnDenoise -d cpu --affinity 0`. Thread pinning raised SIGFPE at device init on a cloud Xeon VM,
  so it is turned off.

## Measured (IQ-lab diorama, 960×640, 256 spp, 32 buckets, all optional lens effects on)

Stage defaults (16 spp guides, `quality: high`); flat regions are the 20th–80th percentile of the local median.

| | pbrt render | denoised |
|---|---|---|
| luminance noise, flat regions (robust σ/μ) | 7.9 % | 0.56 % |
| chroma noise, flat regions | 5.5 % | 0.31 % |
| mean level, flat regions | 1 | 1.013 |
| total spectral energy | 1 | 0.999 |
| dead-leaves 1–3 px detail (DoG σ) | 0.163 | 0.082 (−50 %) |

The +1.3 % in flat regions is firefly energy that OIDN spreads into the surroundings instead of leaving in
isolated pixels; the image total is unchanged. The whole stage (guide render + 32 buckets) took 39 s on 8 CPU
cores.

## Speckle: merge independent renders first (`tools/merge_seeds.py`)

OIDN turns an isolated bright pixel into a blob, and the stage's clamp of single-pixel spikes
(`clamp_fireflies`, 20x the brightest neighbour) discards the light of the largest ones. On a wet road at night the speckle is also real light: rare samples of a headlamp's highlight
on a car, seen in the road (docs/HIGHWAY_SCENES.md, "Lamp regularization"). One render cannot tell such a
sample from a small lamp. Several renders with different seeds can, because the lamp is in all of them.

```bash
for s in 1 2 3 4; do pbrt --seed $s --outfile out/x_s$s.exr x.pbrt; done          # each at spp / 4
venv/bin/python tools/merge_seeds.py --exr-in out/x_s?.exr --exr-out out/x.exr    # then denoise out/x.exr
```

Per pixel and per channel (every spectral bucket, and R, G, B), a render's value above
`ratio` x the median over the renders (default 3, plus a floor of half the local level) is capped, and the
part removed is averaged over the renders and added back through a Gaussian of `sigma` pixels (default 2) in
the same channel. The total of every channel equals that of the plain average. At least three renders are
needed. `<out>.merge.json` records how much light was moved.

Night wet highway, 480x270, four renders of 256 spp, against a 3072 spp reference (error = mean squared
relative error over four colour bands, lamp pixels excluded):

| renders | plain average | merged | image total | road behind the lead car |
|---|---:|---:|---:|---:|
| `--lamp-regularize 0.1` | 0.045 | 0.035 | 1.00000 | 48.05 → 48.06 |
| `--lamp-regularize 0` | 0.263 | 0.090 | 1.00000 | 50.55 → 50.55 |

A stricter cap (`--ratio 2 --sigma 3`) looks cleaner but starts to move ordinary noise and the edges of
lamps (error 0.185 on the first row). Like the denoiser, this is for figures: the merged image's noise is
correlated between neighbouring pixels, so it is not a render to measure noise or texture on.
`paper/ei2027/scripts/render_driving.sh` renders the night panel this way (`NIGHT_SEEDS`, default 4).

## Bright lamps: the bypass

At night the lamps carry most of the light in very few pixels: in the 480x270 wet highway, 458 pixels (0.35 %)
hold 83 % of the image energy, and a lamp is 1-3 px wide. OIDN spread them into their surroundings (frame total
+0.5 %). OptiX (below) keeps 2-7 % of a 1 px spot and dims 2-3 px spots by 10-25 % at lamp brightness, so it lost
44 % of the frame. With the bypass the frame total is 0.999 for both, and the lamp-free error does not change
(table below). Isolated fireflies are clamped before the bypass (`firefly_ratio`), so it does not let them through.

## OptiX (opt-in, NVIDIA GPU)

`tools/denoise_optix.py` runs the same stage (guides, spike clamp, lamp bypass, `<exr>.denoise.json`) with the
NVIDIA OptiX denoiser, through a small CUDA program, `tools/optix_denoise/optix_denoise.cpp`, built against the
OptiX 7.7 SDK (`$OPTIX_SDK`, default `~/NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64`; needs `nvcc` and an NVIDIA driver).
It is not wired into the pipeline config.

```bash
venv/bin/python tools/denoise_optix.py --build
venv/bin/python tools/denoise_optix.py --exr-in out/x.exr --scene x.pbrt     # -> out/x_optix.exr (--mode hdr)
```

`--mode hdr` (default) denoises R,G,B and the buckets three at a time, like the OIDN stage. `--mode aov` predicts
one filter from R,G,B and the guides and applies it to every bucket (OptiX 7.7 takes 3-channel AOV layers only, so
the buckets are packed in threes); it was expected to keep the spectrum better and did not.

Night wet highway, 480x270: four renders of 256 spp merged (`merge_seeds.py`), against the mean of 124 other
renders (seeds 5-128, 31744 spp). Error = mean squared relative error over four colour bands on the road rows,
lamps excluded; spectral shape = median L1 distance between a pixel's normalised bucket spectrum and the reference's.

| | error | spectral shape | frame total | road behind the lead car | near road |
|---|---:|---:|---:|---:|---:|
| merged input | 0.0065 | 0.059 | 0.999 | 1.001 | 1.000 |
| OIDN, no bypass | 0.0028 | 0.028 | 1.005 | 0.995 | 0.997 |
| OIDN | 0.0028 | 0.029 | 0.9985 | 0.996 | 0.997 |
| OptiX hdr, no bypass | 0.0024 | 0.019 | 0.557 | 0.995 | 0.998 |
| OptiX hdr | 0.0023 | 0.019 | 0.9987 | 0.995 | 0.999 |
| OptiX aov | 0.0031 | 0.021 | 0.9990 | 0.997 | 0.998 |

IQ-lab diorama (`tools/denoise_sweep.py` scores, R,G,B, against its 2048 spp reference). Detail and edge gains are
regression slopes of the 1-3 px band on the reference's; the reference's own noise holds the noisy render below 1,
so the column "of input" divides by the noisy render's value.

| | dead-leaves detail | of input | slanted edge | flat-wall noise | texture error | 99.9 % error |
|---|---:|---:|---:|---:|---:|---:|
| 64 spp, OIDN | 0.796 | 0.86 | 0.977 | 0.0070 | 0.088 | 0.65 |
| 64 spp, OptiX hdr | 0.799 | 0.87 | 0.998 | 0.0040 | 0.099 | 0.66 |
| 256 spp, OIDN | 0.833 | 0.93 | 0.979 | 0.0054 | 0.085 | 0.52 |
| 256 spp, OptiX hdr | 0.861 | 0.96 | 0.995 | 0.0035 | 0.093 | 0.53 |

OptiX keeps edges intact, leaves less noise on flat areas and the spectrum closer to the reference; it removes
about as much fine texture as OIDN and leaves a few more isolated bright pixels (hence the higher texture
error). The same rule holds: never compute IQ metrics on its output. Both take about 7 s on the night frame, most
of it the guide render.

## Limits

- OIDN is trained on RGB beauty renders, not on spectral buckets. The per-bucket noise is therefore removed
  well, but adjacent triplets are denoised independently, so the result is a smooth preview spectrum rather than a
  converged one.
- It cannot add light transport pbrt did not sample. Caustics (e.g. under the diorama's glass sphere) come out
  smooth but biased where pbrt found too few paths.
- With a realistic lens the albedo guide is itself blurred by defocus and lens aberrations, which is consistent
  with the beauty render.
