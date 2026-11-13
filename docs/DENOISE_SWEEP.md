# OIDN denoise strength: test plan and sweep

Goal: find settings of the opt-in denoiser (docs/DENOISING.md) that blur less while still removing enough render
noise, and quantify the trade-off. Display/demo renders only; none of this is for IQ metrics.

## Parameters under test

| Axis | Values | Expected effect |
|---|---|---|
| `blend` (new) | 0, 0.1, 0.2, 0.35, 0.5, 0.75, 1 | linear mix of the noisy render back in: the strength knob. Costs nothing per value. |
| input spp | 64, 256 | cleaner input means OIDN removes less real detail |
| `guide_spp` | 1, 4, 16, 64 | cleaner albedo/normal guides keep edges and texture the guides can see |
| `guides` | none, albedo, albedo_normal | without guides OIDN cannot separate texture from noise |
| `quality` | fast, balanced, high | fidelity vs speed; not expected to change blur much |
| `firefly_ratio` (new key) | 0, 5, 20, 100 | pre-OIDN spike clamp; 0 = off. Affects fireflies, not blur |

One axis varies at a time around the base (256 spp, `albedo_normal`, `guide_spp` 16, `high`, `firefly_ratio` 20,
`blend` 0). `blend` is also run on the 64 spp input. Interactions between axes are not covered; follow up with a
grid only on the axes that matter.

## Scene and reference

IQ-lab diorama (`tools/build_iq_diorama.py --camera perspective --xres 720 --yres 480 --caustics path`), window and
lamp on. Inputs: seed 1 at 64 and 256 spp. Reference: seed 7 at 2048 spp. ROIs come from `diorama.json`:
dead-leaves card (texture), slanted-edge card (edge), `shadow_reference` (flat back wall).

## Metrics (luminance, vs reference)

- `flat_err`: std(denoised - reference) / mean(reference) on the flat wall. Noise left plus bias. Lower is better.
- `texture_gain`, `edge_gain`: regression slope of the 1-3 px band (DoG 1 px minus 2 px) of the result on the
  reference's band. 1 means detail kept, below 1 means blurred. Noise does not raise it.
- `global_err`: median relative error over lit pixels.

Pass criterion: a setting is better than the base when `texture_gain` rises without `flat_err` more than doubling.
The reference still carries a little noise (2048 spp), so `texture_gain` can sit slightly below 1 even for a perfect
filter; compare settings with each other, not with 1.0.

## Run

```bash
venv/bin/python tools/build_iq_diorama.py --out-dir out/denoise_sweep/scene --camera perspective \
    --xres 720 --yres 480 --pixelsamples 64 --caustics path --integrator volpath
out/denoise_sweep/render_all.sh          # 64/256 spp noisy + 2048 spp reference (about 10 min on 12 cores)
venv/bin/python tools/denoise_sweep.py --work out/denoise_sweep
```

Outputs in `out/denoise_sweep/`: `results.json` and `figs/` (`overview.png`, `crops_texture.png`,
`crops_edge.png`, `tradeoff.png`, `axes.png`).

## Limits

Only R,G,B are denoised in the sweep (one OIDN call per variant); the pipeline also denoises the spectral buckets
with the same filter. One scene, one seed per input. Compare the final choice by eye on the crops as well.

## Results (256 spp input, `guide_spp` 16 unless noted; one seed per input, so differences under ~0.01 are noise)

`texture_gain` = detail kept (1 = reference); `texture_rmse` = rms error vs reference in the texture ROI (noise plus
lost detail, lower is better); `tail_err` = 99.9th percentile relative error over the frame.

| Scene (texture ROI) | Setting | `texture_gain` | `texture_rmse` | `tail_err` |
|---|---|---|---|---|
| IQ-lab diorama (dead-leaves card) | `albedo`, blend 0 | 0.873 | n/a | 0.525 |
| | `albedo_normal`, blend 0 | 0.833 | n/a | 0.522 |
| Day highway (asphalt) | `albedo`, blend 0 / 0.5 | 0.814 / 0.899 | 0.0414 / 0.0319 | 0.619 / 0.418 |
| | `albedo_normal`, blend 0 / 0.5 | 0.818 / 0.901 | 0.0411 / 0.0318 | 0.379 / 0.347 |
| Night wet highway (asphalt) | `albedo`, blend 0 / 0.5 | 0.457 / 0.584 | 0.0363 / 0.0589 | 2.449 / 1.881 |
| | `albedo_normal`, blend 0 / 0.5 | 0.328 / 0.520 | 0.0373 / 0.0585 | 1.444 / 1.670 |

- Day highway: the 256 spp input is already clean (sky noise 0.002), so blend 0.5 lowers the error (0.041 to 0.032) instead of raising it.
  A tree-against-sky ROI scored about 1.0 for every setting (hard edges dominate), so use a low-contrast texture ROI.
- Night wet highway: blend 0 gives the lowest `texture_rmse`; blend 0.5 raises it from 0.037 to 0.059. `albedo_normal`
  has a much lower `tail_err` than `albedo` (1.44 against 2.45) on the lamp-lit wet road. `texture_gain` is low for
  every setting because the dim asphalt texture is below the render noise.
- `firefly_ratio` 0 / 5 / 20 / 100 changed `tail_err` by under 0.04 at night (1.41 to 1.45) and nothing on the other scenes.
- The night reference (4 x 512 spp merged) still has some speckle in the headlamp reflection on the road.

Run on a highway scene: `tools/denoise_sweep.py --work <dir> --scene <dir>/highway.pbrt --rois '{"flat":[...],"texture":[...],"edge":[...]}'
--noisy "out/noisy_spp{spp}.exr" --reference out/reference.exr --tag _name [--grid]`.
