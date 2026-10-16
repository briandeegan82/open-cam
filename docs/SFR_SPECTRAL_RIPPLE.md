# Per-bucket MTF50 ripple in the EI 2027 D-Gauss renders

**Verdict: the ripple comes from pbrt Monte Carlo sampling noise. The lens does not cause it.** Its period comes from the way pbrt-v4 assigns its four hero wavelengths to film buckets. In these renders the pattern stayed the same when only `pbrt --seed` was changed, because the scene builders wrote `Option "seed" 0`, which overrides `--seed` (item 4). Both are fixed: see *Fix* below.

Setup: the `f2_focused` edge from `paper/ei2027/scripts/render_scenes.sh` (`dgauss.50mm.dat`, 25 mm aperture, 960x640, zsobol 512 spp). Spectral film has 32 buckets over 400-700 nm. MTF50 comes from `sfr_analysis.slanted_edge_sfr(..., flatfield=True)` on the `fig_mtf.edge_roi` crop. Variants were rendered with a `cropwindow` around the ROI.

| render | per-bucket MTF50 std / range |
|---|---|
| zsobol 512 spp (paper) | 5.17 % / 15.9 % |
| zsobol 2048 spp | 2.48 % / 7.5 % |
| independent 512 spp | 7.49 % / 22.4 % |
| independent 2048 spp | 2.76 % / 9.5 % |
| zsobol 512 spp, in-scene `"integer seed" [7]` | 4.59 % / 16.7 % (different pattern) |
| 10 nm narrow band at 450 / 550 / 650 nm, 512 spp | 0.2096 / 0.2096 / 0.2096 cy/px |

What the evidence shows:

1. **The lens has no wavelength dependence** (at the time of this analysis; lenses with a glass sidecar now have dispersion, see [`LENS_DISPERSION.md`](LENS_DISPERSION.md)). `dgauss.50mm.dat` gives one refractive index per element. pbrt-v4's `RealisticCamera` uses that scalar `eta`, so the traced lens has no dispersion. Narrow-band renders at 450, 550 and 650 nm give the same MTF50, 0.2096 cy/px. The broadband green-weighted value is 0.2093 cy/px.
2. **The period is set by how wavelengths are assigned to buckets.** `SampledWavelengths::SampleUniform` takes `NSpectrumSamples = 4` wavelengths per camera path, spaced (700-400)/4 = 75 nm apart. With 32 buckets that is 8 buckets. So buckets k, k+8, k+16 and k+24 receive the same paths. Their images are almost identical: the RMS difference of the normalised ROIs is 0.0015 at a lag of 8 buckets, against 0.28-0.48 at every other lag. With 30 buckets the identical lag becomes 15 (150 nm).
3. **The amplitude behaves like Monte Carlo noise.** Each bucket sees only about 1/8 of the samples. Going from 512 to 2048 spp (4x) halves the ripple, for both the zsobol and the independent sampler.
4. **The pattern stayed the same when only `--seed` changed. Cause (found later): every `tools/build_*.py` scene began with `Option "seed" 0`; pbrt-v4 applies in-scene options after the command line (`scene.cpp`), so the scene reset the seed to 0.** Original notes: In my renders, `--seed 1` and `--seed 2` produced EXRs with bit-identical pixel data, for both samplers. Setting the in-scene `"integer seed" [7]` changed the pattern. However, pbrt-v4's `samplers.cpp` defaults each sampler's seed to `Options->seed` (`parameters.GetOneInt("seed", Options->seed)`), so `--seed` should take effect. I have not found out why it did not here, so treat this as an open question, not as known pbrt behaviour. It is consistent with the paper seeing the same ripple in all three renders. The conclusions in items 1-3 do not depend on it.

## Fix

- **Seed:** the builders no longer write `Option "seed"`, so `pbrt --seed N` takes effect (default seed is still 0, so renders without `--seed` are unchanged). Scripts that already passed `--seed 1`, e.g. `paper/ei2027/scripts/render_scenes.sh` and the pbrt end-to-end tests, now really render with seed 1, so their noise pattern differs from earlier runs (same expectation). The in-scene sampler `"integer seed"` still works and still overrides `--seed`.
- **Ripple:** patch `0006-spectral-film-jittered-wavelengths.patch` makes the spectral film draw one wavelength per 75 nm stratum, each with its own jitter (`SampledWavelengths::SampleUniformJittered`), instead of one offset shared by all four. Each wavelength is still uniform, so the expected image is unchanged and the cost is the same; buckets one stratum apart no longer receive identical paths. `Film "spectral" "bool jitterwavelengths" false` restores stock pbrt. `tests/test_pbrt_seed_ripple.py` checks both: the bucket k / k+4 noise correlation (16 buckets) drops from 0.996 to 0.005.
- Per-bucket noise itself remains (each bucket still sees about 4/nbuckets of the paths); it is now uncorrelated across buckets instead of periodic, and still falls as 1/sqrt(spp).

Consequences: the per-bucket MTF of this prescription should be read as flat. To report a per-wavelength MTF, either render narrow-band images, which avoids sharing samples between buckets, or use many more spp. A dispersive prescription with per-element Abbe numbers would be needed to model real longitudinal chromatic aberration.

The bias in the old edge estimator is a separate problem. On the green-integrated render, the old `|derivative|` centroid (`row_edge_positions_legacy`) reports -2.31 deg. The ISO 12233 estimator (`fit_edge`) reports -4.957 deg, and the paper's workaround reported -4.962 deg. MTF50 is 0.2094 cy/px with `flatfield=True` and 0.1997 cy/px without it; the paper workaround gave 0.2093 cy/px.

## Sensor-response-weighted wavelengths (opt-in)

Patch `0012-spectral-film-bucket-weights.patch` adds `Film "spectral" "float bucketweights" [w_1 ... w_n]`
(one weight per bucket, all > 0). Wavelengths are then drawn per bucket in proportion to `w_b` and
uniformly inside each bucket, with the same per-stratum jitter as above, so each bucket's average
radiance stays unbiased and only its sample count changes. `tools/wavelength_importance.py` writes
the weights from a camera model's QE × IRCF (photon weighted, mixed with a 25 % uniform share):

    venv/bin/python tools/wavelength_importance.py scene/colorchecker.pbrt --camera-model config/camera_models/sony_alpha_6100.yaml

Measured on the ColorChecker (pinhole, 240×160, 32 buckets over 360–830 nm, Sony a6100 QE): bucket
and channel means match uniform sampling within 0.06 % at 1024 spp. At 64 spp the R/G/B electron
noise variance falls to 0.57 / 0.42 / 0.29 of uniform. The cost is in the buckets the sensor does not
see: above 780 nm they get about 2 samples per pixel at 64 spp, and their error variance rises ~39×
(often no sample at all). Use it only when the spectral EXR feeds the sensor the weights were built
for, not for NIR or RGB-IR sensors or spectral analysis.
