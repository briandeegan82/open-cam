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

## Limits

- OIDN is trained on RGB beauty renders, not on spectral buckets. The per-bucket noise is therefore removed
  well, but adjacent triplets are denoised independently, so the result is a smooth preview spectrum rather than a
  converged one.
- It cannot add light transport pbrt did not sample. Caustics (e.g. under the diorama's glass sphere) come out
  smooth but biased where pbrt found too few paths.
- With a realistic lens the albedo guide is itself blurred by defocus and lens aberrations, which is consistent
  with the beauty render.
