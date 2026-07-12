# Colour & CFA — Bayer Sampling, Demosaic, and Colour Science

This chapter covers how colour is captured and processed in a digital camera: the colour filter array, spatial crosstalk, demosaicing, channel crosstalk, and the sRGB encoding applied to preview images.

---

## 1. Colour Filter Arrays

Most digital camera sensors are monochromatic photodiode arrays. Colour information is captured by placing a **colour filter array (CFA)** — a mosaic of dye filters — directly on the sensor surface. Each pixel sees only one colour channel.

### Bayer Pattern

The most common CFA is the **Bayer pattern**, invented by Bryce Bayer (Kodak, 1976). It tiles 2×2 cells with:

```
G R       R G
B G  or   G B
```

The extra green (G) samples match the peak of human luminance sensitivity ($V(\lambda)$). Four Bayer sub-patterns are supported by the pipeline:

| Pattern | Top-left 2×2 |
|---------|-------------|
| `RGGB`  | R Gr / Gb B |
| `BGGR`  | B Gb / Gr R |
| `GRBG`  | Gr R / B Gb |
| `GBRG`  | Gb B / R Gr |

Set in `cfa.pattern` in the camera model.

### Alternative CFA Patterns

Some cameras use non-Bayer CFAs. The pipeline supports several via the `config/camera_recipes/default_*.yaml` variants:

| Recipe | Pattern | Description |
|--------|---------|-------------|
| `default_rccb.yaml` | RCCB | Red, two Cyan, Blue — near-IR capable |
| `default_rggcy.yaml` | RGGCY | Red, two Green, Cyan, Yellow |
| `default_cmy.yaml` | CMY | Cyan, Magenta, Yellow (complementary) |
| `default_ryycy.yaml` | RYYCY | Custom |

---

## 2. Quantum Efficiency and Spectral Sensitivity

Each CFA channel has its own spectral sensitivity curve (QE × CFA dye transmittance × IRCF). These are stored in:

```
spectra/QE/interpolated/
    QE_red.csv
    QE_green.csv
    QE_blue.csv
    QE_IRCF.csv
```

The combined effective sensitivity for channel $c$ is:

$$S_c(\lambda) = \text{QE}_c(\lambda) \cdot T_\text{IRCF}(\lambda)$$

Typical peak sensitivities:
- Red: ~680 nm
- Green: ~540 nm
- Blue: ~450 nm

---

## 3. CFA Spatial Crosstalk

In physical sensors, photons do not always stop in the pixel they enter. Longer-wavelength photons (red, near-IR) penetrate deeper into silicon before being absorbed. If they diffuse laterally before being collected, they are captured by a neighbouring pixel — creating **optical/diffusion crosstalk**.

This effect is wavelength-dependent:
- Blue (400–480 nm): absorbed in the first 1–2 µm — almost no crosstalk
- Green (500–600 nm): absorbed at 2–5 µm depth — moderate crosstalk
- Red (620–700 nm): absorbed at 4–10 µm depth — significant crosstalk in small pixels

The pipeline models this as a per-channel Gaussian blur on the CFA mosaic before noise:

```python
sigma_r = sigma_r / 2.0   # half-sigma on subsampled Bayer phase
sigma_g = sigma_g / 2.0
sigma_b = sigma_b / 2.0
```

The factor $/2$ converts from full-resolution sigma to the subsampled Bayer phase grid.

### Configuration

```yaml
cfa:
  spatial_crosstalk:
    enabled: true
    sigma_pixels: 0.5         # scalar: same for all channels
    # Per-channel overrides (optional):
    sigma_pixels_r: 0.6       # red diffuses more
    sigma_pixels_g: 0.4
    sigma_pixels_b: 0.3       # blue diffuses least
```

---

## 4. Channel Crosstalk Matrix

Beyond spatial diffusion, even a photon stopped exactly within a pixel may excite the wrong channel due to spectral overlap between CFA dyes. This is modelled as a **channel crosstalk matrix** $M$:

$$\begin{pmatrix} e_R' \\ e_G' \\ e_B' \end{pmatrix} = M \begin{pmatrix} e_R \\ e_G \\ e_B \end{pmatrix}$$

The off-diagonal elements of $M$ represent spectral leakage. For example, a typical red dye may transmit 5% at green wavelengths, so $M_{G \leftarrow R} = 0.05$. The channel crosstalk matrix is configured in `sensor.channel_crosstalk` (identity matrix by default, i.e. no crosstalk).

---

## 5. Demosaicing

After noise is applied to the RAW Bayer mosaic, demosaicing reconstructs a full-colour image by interpolating the missing colour values at each pixel location.

### Bilinear Demosaic

The default `demosaic: bilinear` uses bilinear interpolation independently for each channel:

- Green at red/blue positions: average of 4 neighbouring green pixels
- Red at green positions: average of 2 horizontal or 2 vertical red pixels
- Red at blue positions: average of 4 diagonal red pixels
- (similarly for blue)

Bilinear demosaic is computationally simple but introduces **zipper artefacts** (false colour at sharp edges) and **colour moire** in high-spatial-frequency regions.

More sophisticated algorithms (adaptive, gradient-based, or frequency-domain) produce better results at edges but are not implemented in the current pipeline. The pipeline's intent is physical accuracy of the noise chain, not computational photography processing.

### Demosaic Evaluation

The `validate_demosaic_linear.py` tool computes PSNR and per-channel MAE between the demosaiced output and the ground-truth full-colour image (before CFA sampling), using a crop margin to avoid boundary effects.

---

## 6. White Balance

Before sRGB conversion, pixel values are scaled by per-channel white balance gains $(g_R, g_G, g_B)$ to normalise for the illuminant colour. White balance gains are typically computed as:

$$g_c = \frac{1}{S_c(\lambda_\text{illuminant})} \cdot \frac{1}{\langle S_G \rangle}$$

so that a white surface under the chosen illuminant maps to equal R, G, B values. In the pipeline preview output, white balance is disabled by default (`preview_white_balance_enabled: false`).

---

## 7. Colour Correction Matrix

The CFA spectral sensitivities do not match the CIE XYZ colour matching functions. A **colour correction matrix (CCM)** transforms the camera-native RGB to a device-independent colour space:

$$\begin{pmatrix} X \\ Y \\ Z \end{pmatrix} = M_\text{CCM} \begin{pmatrix} R_\text{camera} \\ G_\text{camera} \\ B_\text{camera} \end{pmatrix}$$

The CCM minimises the colour error over a reference target (typically a ColorChecker). In the pipeline this is set via `preview_color_correction_enabled` and the camera model's `color_correction_matrix` field.

---

## 8. sRGB OETF

The final preview images are encoded in sRGB. The **opto-electronic transfer function (OETF)** — often informally called "gamma" — converts linear light values to display-encoded values:

$$V_\text{sRGB}(x) = \begin{cases} 12.92 \, x & x \leq 0.0031308 \\ 1.055 \, x^{1/2.4} - 0.055 & x > 0.0031308 \end{cases}$$

This is the exact IEC 61966-2-1 formula. The linearisation threshold 0.0031308 and the coefficients 12.92 and 1.055 are defined in the standard and must not be approximated.

The two-piece form avoids a discontinuity in slope that would arise from a pure power law at $x = 0$: the linear segment near black provides a smooth transition and avoids noise amplification in the dark region.

The sRGB encoding is applied in `apply_emva_noise.py` when generating preview PNGs (`demosaic_srgb: true`).

---

## 9. Colour Spaces Reference

| Space | Primaries | White point | Gamma | Usage |
|-------|-----------|-------------|-------|-------|
| sRGB | ITU-R BT.709 | D65 | IEC 61966-2-1 | Consumer display |
| Adobe RGB | Wide gamut | D65 | $\gamma = 2.2$ | Photography, printing |
| CIE XYZ | Imaginary primaries | D65 / E | Linear | Colour science reference |
| ProPhoto RGB | Wide gamut | D50 | $\gamma \approx 1.8$ | RAW processing |
| Linear sRGB | ITU-R BT.709 | D65 | 1.0 | HDR, tone mapping |

The pipeline stores intermediate electron arrays in linear units. sRGB encoding is only applied to 8-bit preview PNGs.

---

## See Also

- [Sensor Physics](02_sensor_physics.md) — QE integration per channel
- [Noise Model](03_noise_model.md) — per-CFA-channel PRNU and DSNU
- [Optics & PSF](04_optics.md) — diffusion PSF and spatial crosstalk relation
- [Validation](../guides/validation.md) — demosaic quality metrics
