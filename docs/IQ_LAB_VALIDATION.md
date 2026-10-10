# IQ lab ground-truth validation

`tools/validate_iqlab_metrics.py` checks the `tools/iqlab` and `tools/sfr_analysis.py` metrics against
synthetic inputs with known answers. Each suite runs the metric over several noise realisations and
reports the bias and spread against the analytic truth. `tests/test_iqlab_ground_truth.py` runs every
suite and fails if any case leaves its tolerance.

```bash
venv/bin/python tools/validate_iqlab_metrics.py --out-dir out/iq_lab/validation --figure
venv/bin/python -m pytest tests/test_iqlab_ground_truth.py
```

The command writes `iqlab_validation.json` (truth, mean, std and limits for every case), `iqlab_validation.md`
and, with `--figure`, `iqlab_validation.png`. It covers 75 cases and takes about 10 s.

## Suites and ground truth

| Suite | Synthetic input | Truth | Metric under test |
|---|---|---|---|
| `edge` | Slanted edge (3, 5 and 8 deg) convolved with a Gaussian PSF (sigma 0.5–2 px) on an 8x grid, then box-averaged into pixels. Edge SNR of infinity, 100 or 30. | \( \exp(-2\pi^2\sigma^2 f^2)\,\lvert\mathrm{sinc}\,f\rvert \); MTF50 by root finding; CPIQ acutance of the true MTF | `slanted_edge_sfr`, MTF50, `cpiq.acutance` |
| `texture` | Dead-leaves target with Gaussian blur and white noise, with noise PSD taken from a separate flat patch | Gaussian MTF | `dead_leaves.texture_mtf`, texture acutance |
| `snr` | Poisson flats with read noise from 5 to 5000 e-, plus PRNU in the total-noise case | \( S/\sqrt{S + r^2 + (pS)^2} \) | `snr.patch_stats`, `snr.temporal_patch_stats` |
| `dr` | A ladder of clipped patches with read noise of 1.5, 3 and 10 e- | SNR = 1 and SNR = 10 thresholds from \( S^2 = T^2(S + r^2) \) | `snr.dynamic_range` |
| `cdp` | Pairs of Gaussian dark and bright patches | CDP from 1-D quadrature of the pair-contrast distribution | `p2020.contrast_detection_probability` |
| `colour` | Chart patches with a known L\*C\*h shift, rendered as linear sRGB at an arbitrary exposure, with or without 2 % noise | The injected ΔL\*, ΔC\* and Δh, and the resulting CIEDE2000 | `run_skin_tone_test.score` (normalised to the white patch) |

## Tolerances

Limits are defined in `TOLERANCES` and apply per noise tier:
- "clean" means noise-free, or an edge SNR of at least 100.
- "noisy" means the noisiest case in each suite.

Bias limits apply to the mean over trials. RMS and `*_abs` limits apply to \( \sqrt{\mathrm{bias}^2 + \mathrm{std}^2} \).

| Suite | Clean | Noisy |
|---|---|---|
| edge | MTF50 bias ≤ 3.5 %, RMS ≤ 4 %; MTF RMS ≤ 0.02; acutance ≤ 0.015 | MTF50 bias ≤ 7 %, RMS ≤ 8 %; MTF RMS ≤ 0.05; acutance ≤ 0.02 |
| texture | MTF RMS ≤ 0.005; acutance ≤ 0.005 | MTF RMS ≤ 0.06; acutance ≤ 0.03 |
| snr | bias ≤ 1 %, RMS ≤ 3 % | (same limits) |
| dr | ≤ 0.4 dB | (same limits) |
| cdp | ≤ 0.006 | (same limits) |
| colour | ≤ 1e-6 (exact) | ΔE00 ≤ 0.3, ΔL\* ≤ 0.15, ΔC\* ≤ 0.35, Δh ≤ 1 deg |

## Results (seed 0)

All 75 cases pass, and they also pass with seeds 1–3. The largest errors were:

- **Edge MTF50:** sharp edges read slightly low. At sigma = 0.5 px, where MTF50 is about 0.32 cy/px, the noise-free MTF50 is 2.5–3 % low. At sigma = 0.8 px the bias is about 1 %. At sigma ≥ 1.2 px it is below 0.6 %.
  - The cause is the 4x ESF binning, which is the ISO 12233 default. It is not the Hamming window or the derivative correction.
  - With `oversampling=8` the bias drops to 0.4 % (pinned by `test_sharp_edge_bias_is_from_4x_binning`).
  - For very sharp optics, use 8x oversampling, or quote MTF50 as about 3 % conservative.
- **Edge with noise:** at an edge SNR of 30, the MTF RMS error reaches 0.04 and the MTF50 spread reaches 5–6.5 % with 10 trials.
- **Texture MTF:** noise-free texture MTF is within 0.003 RMS. With noise of σ = 0.03, noise-PSD subtraction leaves an RMS error of 0.04 at sigma = 1.5 px.
- **SNR:** single-frame SNR, temporal SNR and total SNR with PRNU have a bias of ≤ 0.5 % from 5 e- to 5000 e-.
- **Dynamic range:** within 0.25 dB of the analytic SNR = 1 and SNR = 10 thresholds.
- **CDP:** within 0.003 of the quadrature value.
- **Colour:** noise-free scoring is exact to 1e-12. With 2 % noise on 24 px patches, the worst-patch spread is about 0.19 ΔE00 and 0.5 deg of hue.
