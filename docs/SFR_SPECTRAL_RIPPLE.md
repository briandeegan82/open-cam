# Per-bucket MTF50 ripple in the EI 2027 D-Gauss renders

**Verdict: the ripple comes from pbrt Monte Carlo sampling noise. The lens does not cause it.** Its period comes from the way pbrt-v4 assigns its four hero wavelengths to film buckets. The pattern is the same in every render because the `pbrt --seed` flag does not change the output.

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

1. **The lens has no wavelength dependence.** `dgauss.50mm.dat` gives one refractive index per element. pbrt-v4's `RealisticCamera` uses that scalar `eta`, so the traced lens has no dispersion. Narrow-band renders at 450, 550 and 650 nm give the same MTF50, 0.2096 cy/px. The broadband green-weighted value is 0.2093 cy/px.
2. **The period is set by how wavelengths are assigned to buckets.** `SampledWavelengths::SampleUniform` takes `NSpectrumSamples = 4` wavelengths per camera path, spaced (700-400)/4 = 75 nm apart. With 32 buckets that is 8 buckets. So buckets k, k+8, k+16 and k+24 receive the same paths. Their images are almost identical: the RMS difference of the normalised ROIs is 0.0015 at a lag of 8 buckets, against 0.28-0.48 at every other lag. With 30 buckets the identical lag becomes 15 (150 nm).
3. **The amplitude behaves like Monte Carlo noise.** Each bucket sees only about 1/8 of the samples. Going from 512 to 2048 spp (4x) halves the ripple, for both the zsobol and the independent sampler.
4. **The pattern is fixed across renders because `--seed` has no effect.** EXRs rendered with `--seed 1` and `--seed 2` have bit-identical pixel data, for both samplers. Only the in-scene `"integer seed"` parameter changes the pattern. This explains why the paper saw the same ripple "in all three renders".

Consequences: the per-bucket MTF of this prescription should be read as flat. To report a per-wavelength MTF, either render narrow-band images, which avoids sharing samples between buckets, or use many more spp. Varying the seed requires the in-scene `"integer seed"` parameter. A dispersive prescription with per-element Abbe numbers would be needed to model real longitudinal chromatic aberration.

The bias in the old edge estimator is a separate problem. On the green-integrated render, the old `|derivative|` centroid (`row_edge_positions_legacy`) reports -2.31 deg. The ISO 12233 estimator (`fit_edge`) reports -4.957 deg, and the paper's workaround reported -4.962 deg. MTF50 is 0.2094 cy/px with `flatfield=True` and 0.1997 cy/px without it; the paper workaround gave 0.2093 cy/px.
