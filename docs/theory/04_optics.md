# Optics & PSF — Diffraction, Aberration, and Stray Light

This chapter covers how the optical system blurs and scatters light before it reaches the sensor. The pipeline models optics at two levels: physically in PBRT (for realistic lens models) and analytically as a post-processing PSF blur (`tools/apply_spectral_psf.py`).

---

## 1. The Airy Disk and Diffraction Limit

Even a geometrically perfect lens produces a finite-sized spot at focus due to wave diffraction at the circular aperture. The intensity distribution of the focused point-spread function (PSF) is an **Airy pattern**:

$$I(r) = I_0 \left[\frac{2 J_1(x)}{x}\right]^2, \qquad x = \frac{\pi D r}{\lambda z}$$

where $D$ is the aperture diameter, $z$ is the image distance, $r$ is the radial distance from the optical axis, and $J_1$ is the first-order Bessel function of the first kind.

The radius of the first dark ring (Airy disk radius) is:

$$r_\text{Airy} = 1.22 \frac{\lambda}{D} z = 1.22 \lambda N$$

where $N = z/D \approx f/D$ is the f-number (for distant objects). The **FWHM** of the Airy pattern is approximately:

$$\text{FWHM}_\text{Airy} = 1.028 \lambda N$$

---

## 2. Gaussian Approximation

For the purposes of an efficient post-PSF blur, the Airy pattern is approximated by a Gaussian with the same FWHM. Because $\text{FWHM}_\text{Gaussian} = 2\sqrt{2\ln 2}\,\sigma \approx 2.355\,\sigma$:

$$\sigma = \frac{\text{FWHM}_\text{Airy}}{2.355} = \frac{1.028}{2.355} \lambda N = 0.437\, \lambda N$$

Converting to pixel units (pixel pitch $p$ in µm, $\lambda$ in nm):

$$\boxed{\sigma_\text{pixels} = \frac{0.437 \cdot \lambda_\text{nm} \cdot N}{1000 \cdot p_\text{µm}}}$$

The factor 1000 converts nm·µm to consistent units (both are $10^{-9}$·$10^{-6}$ = $10^{-15}$ m, so the ratio $\lambda[\text{nm}]/p[\text{µm}]$ is dimensionless when divided by 1000).

This is implemented in `apply_spectral_psf.py`:

```python
sigma_diff = 0.437 * f_number * wavelength_nm / (1000.0 * pixel_pitch_um)
```

The coefficient 0.437 = 1.028 / 2.355 is exact (Airy FWHM matched to Gaussian sigma).

---

## 3. Chromatic Aberration

### Longitudinal Chromatic Aberration (LCA)

LCA (also called axial chromatic aberration) occurs because the refractive index of glass depends on wavelength. Shorter wavelengths (blue) focus at a slightly shorter distance than longer wavelengths (red). The result is coloured halos around sharp edges in out-of-focus areas.

In the analytic PSF model, LCA is added on top of diffraction by specifying a wavelength-dependent additional sigma:

```yaml
lens:
  post_psf:
    enabled: true
    mode: gaussian
    sigma_pixels: 0.75        # geometric (non-diffraction) aberration base
    lca_enabled: true         # adds chromatic sigma per wavelength
    lca_strength: 0.003       # sigma_lca = lca_strength * |lambda - lambda0| / lambda0
```

When LCA is enabled, the total sigma per wavelength is:

$$\sigma_\text{total}(\lambda) = \sqrt{\sigma_\text{diff}^2(\lambda) + \sigma_\text{geom}^2 + \sigma_\text{LCA}^2(\lambda)}$$

### Transverse Chromatic Aberration (TCA)

TCA is a lateral shift of the PSF centre as a function of wavelength, causing colour fringing at high-contrast edges. It is not explicitly modelled in the current analytic post-PSF but is handled physically by the PBRT realistic lens tracer.

---

## 4. Geometric Aberrations

Real lenses have additional aberrations beyond diffraction:

| Aberration | Description | Effect |
|-----------|-------------|--------|
| Spherical | On-axis blur | Uniform softness |
| Coma | Off-axis asymmetric blur | Comet-shaped PSF |
| Field curvature | Focus shifts with field angle | Soft corners |
| Astigmatism | Different focus planes for tangential/sagittal rays | Directional blur |
| Distortion | Magnification varies with field height | Barrel/pincushion shape |

In PBRT, all geometric aberrations are implicitly simulated by ray-tracing through the lens prescription file (`.dat`). In the analytic Gaussian PSF the geometric residual is captured as the `sigma_pixels` base blur, which is independent of wavelength.

---

## 5. PSF Convolution Implementation

The post-PSF tool (`apply_spectral_psf.py`) applies the wavelength-dependent Gaussian blur to each spectral channel of the PBRT output EXR before the radiometric integration step:

1. Load multispectral EXR channels (`S0.360nm` … `S0.830nm`)
2. For each wavelength band, compute $\sigma_\text{total}(\lambda)$
3. Apply 2D Gaussian convolution (`scipy.ndimage.gaussian_filter`)
4. Save blurred EXR; the subsequent `pbrt_spectral_exr_to_electrons.py` integrates over the blurred channels

When the realistic lens model is used in PBRT, the physical PSF (diffraction + aberrations) is already baked into the render. In that case the post-PSF tool should be disabled or its `sigma_pixels` set to near zero.

---

## 6. Stray Light

Stray light is unwanted light that reaches the sensor via unintended paths: reflections off lens barrel surfaces, scattering from dust, and ghost images from bright sources. It manifests as:

- **Veiling glare**: a uniform additive fog that reduces contrast globally
- **Halos**: diffuse rings around bright point sources (lens flare)
- **Ghost images**: faint reflections of bright scene elements

The stray light model in the pipeline adds:

$$L_\text{total} = L_\text{direct} + f_\text{veil} \cdot \langle L_\text{direct}\rangle + f_\text{halo} \cdot (L_\text{direct} * G_\text{halo})$$

where $\langle \cdot \rangle$ is the spatial mean and $G_\text{halo}$ is a large Gaussian kernel.

Configuration:

```yaml
lens:
  post_psf:
    stray_light:
      enabled: true
      veiling_glare_fraction: 0.01      # fraction of mean scene luminance
      halo_sigma_pixels: 8.0            # halo kernel radius
      halo_strength: 0.04               # halo contribution at peak
```

---

## 7. PBRT Realistic Lens Model

The `realistic` lens type in PBRT ray-traces a multi-element lens prescription. The lens files (`.dat`) in `config/lenses/` describe each element as:

```
# radius  thickness  ior  aperture_diameter
-35.0     5.5        1.65    32.0
0.0       6.2        0.0     30.0    # aperture stop
...
```

Fields:
- **radius**: spherical surface radius of curvature (mm); 0 = flat (aperture stop)
- **thickness**: axial separation to next surface (mm)
- **ior**: index of refraction (0 = air gap)
- **aperture_diameter**: clear aperture diameter (mm)

PBRT traces each ray through all surfaces using Snell's law, handles total internal reflection, and computes the exact intersection geometry. This captures:
- Geometric aberrations (field curvature, astigmatism, distortion, coma)
- Vignetting (partial blocking of oblique bundles by barrel elements)
- Ghost images (not by default — requires separate reflection path)

The realistic lens model does not require the post-PSF tool for aberrations; diffraction is not modelled by ray tracing and must still be added post-render if desired.

---

## 8. Diffraction Limit vs Pixel Pitch

A useful rule of thumb: the diffraction spot is smaller than a pixel when:

$$\sigma_\text{pixels} = \frac{0.437 \lambda N}{1000 p} < 1$$

For $\lambda = 550$ nm (green light):

$$N < \frac{1000 p}{0.437 \cdot 550} = \frac{1000 p}{240}$$

| Pixel pitch $p$ | Diffraction-limited at $N <$ |
|----------------|------------------------------|
| 1.4 µm | 5.8 |
| 3.8 µm | 15.8 |
| 5.9 µm | 24.5 |

Phone cameras with 1.4 µm pixels become diffraction-limited at around $f$/5.6. DSLR sensors with 5.9 µm pixels remain aberration-limited up to $f$/22.

---

## See Also

- [Radiometry](01_radiometry.md) — the thin-lens irradiance formula
- [Colour & CFA](05_color_cfa.md) — CFA spatial crosstalk (related to diffusion PSF)
- `tools/apply_spectral_psf.py` — post-PSF implementation
- `docs/LENS_MODELS.txt` → [Lens Models Guide](../guides/lens_models.md)
