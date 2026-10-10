# Image-quality lab: metrics (`tools/iqlab`)

Image-quality metrics that run on plain NumPy arrays, for scoring open-cam renders (pbrt -> sensor ->
ISP) and real captures in the same way. Scene targets and runners (HDR chart, flare, skin tones,
diorama) build on these modules. Tests: `tests/test_iqlab.py`. Each test checks a metric against a
known analytic answer.

| Module | Metric | Inputs |
|---|---|---|
| `iqlab.snr` | Patch mean / std / SNR (plane-detrended, saturation fraction); temporal SNR from a frame stack; SNR-threshold dynamic range (SNR = 1 EMVA-style, SNR = 10 ISO 15739-style) in dB / stops | linear, black-subtracted patches |
| `iqlab.p2020` | Contrast detection probability (CDP); colour separation (Euclidean, Mahalanobis, Φ(D_M/2) separation probability, empirical classification accuracy) | linear patch pixels; `(N, C)` colour pixels |
| `iqlab.cpiq` | CPIQ CSF, viewing conditions, acutance (MTF·CSF), visual noise, chroma level, colour uniformity (Δu'v', ΔE00 vs centre) | MTF from `sfr_analysis`; sRGB-encoded patches |
| `iqlab.dead_leaves` | Dead-leaves target generator; texture MTF from the ideal-vs-captured PSD with the noise PSD subtracted; texture acutance | registered captured and ideal target |
| `iqlab.geometry` | Dot-grid centroids; local geometric distortion (radial %, ideal grid fitted from the central dots); lateral chromatic displacement (R−G, B−G centroid offsets) | dot-grid chart image |

## Definitions and sources

- **CDP** (Geese et al., IS&T EI 2018; the IEEE P2020 KPI). Sample random dark/bright pixel pairs and
  compute the Michelson contrast `C_k = (b − d)/(b + d)` for each pair. Then
  `CDP = P(|C_k − C_nom| ≤ ε·C_nom)`, with `ε = 0.5` by default. Wrong-sign pairs and pairs with
  `b + d ≤ 0` count as failures. `C_nom` defaults to the contrast of the patch means. Pass the scene
  contrast instead to also penalise tone-curve compression and flare.
- **Colour separation**: Mahalanobis distance between two patch distributions using the pooled
  covariance. `Φ(D_M/2)` is the optimal linear classifier's accuracy under equal-covariance Gaussians.
- **Acutance**: `∫ MTF(ν)·CSF(ν) dν / ∫ CSF(ν) dν` over 0 to 0.5 cy/px, with
  `CSF(ν) = 75 ν^0.8 e^(−0.2 ν) / 34.05` (ν in cycles/degree, as given by Baxter et al., SPIE 8293, 2012).
  The viewing geometry maps cy/px to cy/deg. The presets are `monitor_100pct`, `uhd_24in_0p6m`
  and `print_4x6_0p4m`.
- **Visual noise**: follows the ISO 15739 structure. Luminance is filtered by the CSF and chroma by a
  Gaussian low-pass in the viewing geometry, then the patch is converted to CIELUV and
  `VN = log10(1 + w1·σ²L* + w2·σ²u* + w3·σ²v*)`. The defaults are `w = (1, 0.852, 0.323)` and a
  chroma low-pass σ of 4 cy/deg.
- **Chroma level**: `100 × mean C*ab(measured) / mean C*ab(reference)` over the chosen patches.
- **Texture MTF**: `sqrt((PSD_capture − PSD_noise) / PSD_ideal)`, radially averaged. In simulation the
  ideal target is known exactly, so no analytic dead-leaves spectrum model is needed.

## Not standards-certified

IEEE P2020 and IEEE 1858 (CPIQ) are not open documents, so these implementations come from the
published papers and the ISO structure. Constants are exposed as arguments: CSF coefficients,
visual-noise weights, the chroma filter and CDP ε. Check them against the standard text before
quoting absolute values. Quality-loss/JND mapping and the exact CPIQ chroma filters are not
implemented. Relative comparisons between cameras, recipes and ISP settings are what this module is
meant for.
