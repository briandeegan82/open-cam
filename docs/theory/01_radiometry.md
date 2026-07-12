# Radiometry — From Scene to Pixel Irradiance

This chapter covers the physical quantities and equations that describe how light travels from a scene surface to the focal plane of a camera sensor. These are the foundations of the `pbrt_spectral_exr_to_electrons.py` conversion step.

---

## 1. Radiometric Quantities

| Symbol | Name | SI unit | Definition |
|--------|------|---------|-----------|
| $Q$ | Radiant energy | J | Energy carried by photons |
| $\Phi$ | Radiant flux | W | Power: $\Phi = dQ/dt$ |
| $I$ | Radiant intensity | W/sr | Flux per solid angle: $I = d\Phi/d\omega$ |
| $L$ | Radiance | W/(m²·sr) | Flux per projected area per solid angle |
| $E$ | Irradiance | W/m² | Incident flux per unit area: $E = d\Phi/dA$ |

In a spectral simulation all quantities are additionally per-nanometre: $L_\lambda$ [W/(m²·sr·nm)], $E_\lambda$ [W/(m²·nm)].

### Radiance

Radiance $L$ is the fundamental quantity preserved along a ray in a lossless medium:

$$L(\mathbf{x}, \hat{\omega}) = \frac{d^2\Phi}{dA\cos\theta \, d\omega}$$

PBRT's `Film "spectral"` records spectral radiance at each pixel: channels named `S0.360nm` through `S0.830nm`, units W/(m²·sr·nm) per steradian of sensor solid angle (after sensor-response normalisation by the film integrator).

---

## 2. Irradiance at the Focal Plane

### 2.1 The Thin-Lens Irradiance Formula

For a camera with a circular aperture of diameter $D$ and focal length $f$, imaging a scene at distance $d_o \gg f$, the irradiance at the focal plane from a Lambertian scene element of radiance $L$ is:

$$E = \frac{\pi}{4} \left(\frac{D}{f}\right)^2 \tau_\text{optics} \cdot L \cos^4\!\theta$$

where $\theta$ is the off-axis angle. Because $N = f/D$ is the f-number:

$$\boxed{E = \frac{\pi}{4 N^2} \, \tau_\text{optics} \cdot L \cos^4\!\theta}$$

This formula is implemented in `sensor_radiometry.py` and `pbrt_spectral_exr_to_electrons.py` with the key expression:

```python
E = L * math.pi / (4.0 * f_number**2) * tau_optics * irr_scale
```

### 2.2 Magnification Correction

The thin-lens formula above assumes $d_o \gg f$ (i.e. magnification $m = f/d_o \approx 0$). For close-focus (macro) work the effective f-number increases:

$$N_\text{eff} = N (1 + m), \qquad m = \frac{f}{d_o - f}$$

and the irradiance formula becomes:

$$E = \frac{\pi}{4 N_\text{eff}^2} \, \tau_\text{optics} \cdot L$$

For a 1:1 macro shot $m = 1$, so $N_\text{eff} = 2N$: the sensor receives four times less light than the marked f-number suggests. The `research_thinlens_macro.yaml` recipe uses this geometry.

### 2.3 f-Number and Exposure

The exposure $H$ (J/m²) accumulated during integration time $t_\text{int}$ is:

$$H = E \cdot t_\text{int} = \frac{\pi \, \tau_\text{optics} \cdot L \cdot t_\text{int}}{4 N^2}$$

Halving the f-number quadruples $E$; doubling $t_\text{int}$ doubles $H$. These combine as the classic "exposure value" (EV) scale.

---

## 3. Photon Flux Density

Irradiance $E_\lambda$ [W/(m²·nm)] is power. Each photon at wavelength $\lambda$ carries energy $hc/\lambda$, so the photon flux density (photons per second per m² per nm) is:

$$\Phi_\lambda = \frac{E_\lambda}{\,hc/\lambda\,} = \frac{E_\lambda \cdot \lambda}{h c}$$

with constants:
- $h = 6.62607015 \times 10^{-34}$ J·s (Planck constant)
- $c = 2.99792458 \times 10^8$ m/s (speed of light)

Implemented in `sensor_radiometry.py`:

```python
def photon_flux_density_from_irradiance(E_lambda, lambda_nm):
    """E_lambda: W/(m^2 nm), lambda_nm: nm -> photons/(s m^2 nm)"""
    h = 6.62607015e-34
    c = 2.99792458e8
    lambda_m = lambda_nm * 1e-9
    return E_lambda * lambda_m / (h * c)
```

---

## 4. Vignetting

### 4.1 Natural Vignetting (cos⁴ Law)

Even a perfect lens produces a radial fall-off of irradiance known as natural or radiometric vignetting. For a thin lens imaging a distant scene the irradiance at field angle $\theta$ relative to the on-axis irradiance $E_0$ is:

$$\frac{E(\theta)}{E_0} = \cos^4\!\theta$$

The four powers arise from:

1. $\cos\theta$ — projected aperture area as seen from the off-axis image point
2. $\cos^2\theta$ — reduced solid angle subtended by the aperture ($1/d^2$ factor, $d \propto 1/\cos\theta$)
3. $\cos\theta$ — oblique incidence on the focal plane

Enabled in the camera model with `vignetting_cos4: true`.

### 4.2 Optical / Mechanical Vignetting

Real lenses add further fall-off from barrel and aperture mechanical vignetting (partial blocking of oblique bundles). This is modelled phenomenologically by the `optics_transmittance_spatial` block in the camera model:

```yaml
optics_transmittance_spatial:
  enabled: true
  mode: radial_power
  edge_factor: 0.85    # relative transmission at image corner
  exponent: 2.0        # roll-off shape (2.0 ≈ cos^4 profile)
```

The transmission map is:

$$T(r) = 1 - (1 - \alpha) r^\beta$$

where $r \in [0,1]$ is the normalised radial distance from image centre, $\alpha$ = `edge_factor`, $\beta$ = `exponent`.

PBRT realistic lenses model vignetting physically; enabling the `optics_transmittance_spatial` block is only relevant for the analytic forward model (`spectral_sensor_forward.py`).

---

## 5. Photometric Units

While radiometry uses watts, photometry weights by the CIE photopic luminosity function $V(\lambda)$. The pipeline uses photometric quantities for scene calibration (target illuminance in lux).

| Photometric | Radiometric equivalent | Conversion |
|------------|----------------------|------------|
| Luminous flux [lm] | Weighted radiant flux | $\Phi_v = 683 \int V(\lambda) \Phi_\lambda \, d\lambda$ |
| Illuminance [lux = lm/m²] | Weighted irradiance | $E_v = 683 \int V(\lambda) E_\lambda \, d\lambda$ |
| Luminance [cd/m² = lm/(m²·sr)] | Weighted radiance | $L_v = 683 \int V(\lambda) L_\lambda \, d\lambda$ |

The factor 683 lm/W is the maximum spectral luminous efficacy at $\lambda = 555$ nm. The pipeline's `radiometric_autocalibration: mean_photopic_lux` mode uses CIE 1924 $V(\lambda)$ at 380–780 nm to convert from the simulated spectral radiance to lux, then scales the exposure accordingly.

---

## 6. Lambert's Cosine Law and Cosine Shading

A Lambertian surface reflects diffusely: the radiance $L$ is the same in all viewing directions. However, the irradiance it receives from a directional source depends on the angle of incidence $\alpha$ between the surface normal $\hat{n}$ and the illumination direction $\hat{s}$:

$$E_\text{surface} = E_\text{source} \cos\alpha = E_\text{source} (\hat{n} \cdot \hat{s})$$

The `cosine_shading` option in `sensor_forward_model` accounts for this: when a chart surface is tilted relative to the illumination, the reflected radiance is reduced by $\cos\alpha$.

---

## See Also

- [Sensor Physics](02_sensor_physics.md) — converting irradiance to electrons
- [Optics & PSF](04_optics.md) — aberrations and diffraction
