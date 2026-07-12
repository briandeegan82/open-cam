# Sensor Physics — From Photons to Electrons

This chapter explains how the pipeline converts per-pixel spectral irradiance into a count of photo-generated electrons. These are the computations performed by `tools/pbrt_spectral_exr_to_electrons.py` (PBRT path) and `tools/spectral_sensor_forward.py` (analytic forward-model path).

---

## 1. The Photoelectric Effect

When a photon with energy $E_\text{photon} = hc/\lambda$ is absorbed in a silicon photodetector, it may generate a free electron–hole pair. The probability that an absorbed photon produces a collected electron is the **internal quantum efficiency** (IQE). Accounting for reflective and absorption losses at the silicon surface, the observable quantity is the **external quantum efficiency** (QE):

$$\text{QE}(\lambda) = \frac{\text{electrons collected per second}}{\text{incident photons per second}}$$

QE is dimensionless, in the range $[0, 1]$. In practice it is wavelength-dependent because:
- Blue photons ($\lambda \lesssim 450$ nm) are absorbed near the surface where recombination is high
- Near-infrared photons ($\lambda \gtrsim 800$ nm) are not absorbed in the thin silicon layer
- The channel spectral sensitivity also includes the colour filter array (CFA) dye transmission

The QE curves are stored as CSV files: `spectra/QE/interpolated/QE_red.csv`, `QE_green.csv`, `QE_blue.csv`. The IRCF (infrared cut filter) transmission is in `QE_IRCF.csv` and is multiplied into the effective QE before integration.

---

## 2. Fill Factor

Not all of the pixel area is sensitive to light. The **fill factor** (FF) is the fraction of pixel area that is photosensitive:

$$\text{FF} = \frac{A_\text{photodiode}}{A_\text{pixel}}$$

Typical values:
| Sensor type | Typical FF |
|------------|-----------|
| Front-side illuminated (FSI) CMOS | 0.40–0.60 |
| Back-side illuminated (BSI) CMOS | 0.65–0.80 |
| Micro-lens BSI | 0.90–0.98 |
| Phone BSI (stacked) | 0.75–0.80 |

Fill factor is set per camera model in `sensor.fill_factor`. Modern phone sensors with micro-lenses can reach ~0.76; the default model uses 0.95 for a BSI sensor with micro-lens.

---

## 3. Electron Count Integral

The number of photo-electrons collected by a pixel of pitch $p$ (µm) during integration time $t_\text{int}$ (s) is obtained by integrating over the spectral range:

$$\boxed{e_c = \int_{\lambda_\min}^{\lambda_\max} \Phi_\lambda(\lambda) \cdot \text{QE}_c(\lambda) \cdot \Delta\lambda \cdot A_\text{pixel} \cdot t_\text{int} \cdot \text{FF}}$$

where:
- $\Phi_\lambda(\lambda)$ — photon flux density [photons/(s·m²·nm)] at wavelength $\lambda$
- $\text{QE}_c(\lambda)$ — quantum efficiency for colour channel $c$ (R, G, or B)
- $A_\text{pixel} = (p \times 10^{-6})^2$ — pixel area [m²]
- $t_\text{int}$ — integration time [s]
- FF — fill factor [dimensionless]

In code (`pbrt_spectral_exr_to_electrons.py`):

```python
# photon_flux shape: (H, W, N_bands)
# qe_interp shape:   (N_bands,) for one channel
electrons = np.trapz(photon_flux * qe_interp, x=lambda_nm, axis=-1)
electrons *= pixel_area_m2 * t_int * fill_factor
```

The integral is evaluated by the trapezoidal rule over the 32 spectral buckets produced by `Film "spectral"` in the range 360–830 nm.

### Spectral Band Centres

PBRT's spectral film divides [360, 830] nm into `spectral_nbuckets` equally-spaced bins. With 32 buckets, the bin width is $(830-360)/32 = 14.7$ nm and the band-centre wavelengths are at 367.3, 382.0, … nm. The QE curves are interpolated to these bin centres before integration.

---

## 4. ISO Gain and Exposure Scale

Camera ISO is implemented as a post-ADC metadata tag in digital cameras, but its effect on noise is equivalent to amplifying the signal before digitisation. In the pipeline, ISO is modelled via the `processing.exposure_scale_e_per_unit` parameter, which scales the effective electron count:

$$e_\text{effective} = e_\text{raw} \times \text{ISO\_scale}$$

When `sensor_forward.mode = pbrt_exr`, the `exposure_time_override_s` in `pipeline.yaml` sets the integration time globally, overriding the camera model's `sensor.integration_time_s`.

---

## 5. The Analytic Forward Model

The analytic forward model (`spectral_sensor_forward.py`) bypasses PBRT and computes electron maps directly from a spectral scene description. It is used for rapid validation and for generating `sensor_forward_electrons.npz`.

The calibration chain is:

1. **Target illuminance** (`target_illuminance_lux`): the scene is assumed to be a Lambertian reflector under a standard illuminant at the specified lux level.
2. **Irradiance from illuminance**: $E_\lambda = L_\text{illuminant}(\lambda) \cdot \rho(\lambda) / \pi$ (irradiance at Lambertian reflectance $\rho$).
3. **Aperture factor**: $E_\text{focal} = E_\text{scene} \cdot \pi/(4N^2) \cdot \tau_\text{optics}$ when `use_aperture_factor: true`.
4. **Photon flux → electrons**: same integral as Section 3.
5. **Cosine shading** and **cos⁴ vignetting** applied as spatial multiplier maps.

The calibration mode `photon_counting` uses the full spectral integral; `photopic_lux` uses the $V(\lambda)$ photometric calibration.

---

## 6. Spectral QE Data Format

QE CSV files have two columns: `wavelength_nm`, `qe`. Example excerpt:

```
wavelength_nm,qe
380,0.12
390,0.18
400,0.25
...
700,0.55
710,0.52
```

Files are expected to span at least 360–830 nm after interpolation. The pipeline linearly interpolates to the PBRT band centres. Points outside the range are extrapolated as zero.

The IRCF CSV (`QE_IRCF.csv`) follows the same format and represents the transmission of the infrared cut filter; values near 0 above ~700 nm. The combined effective QE is:

$$\text{QE}_\text{eff}(\lambda) = \text{QE}_\text{channel}(\lambda) \cdot T_\text{IRCF}(\lambda)$$

---

## 7. Full Well Capacity

Each pixel can hold a maximum charge $Q_\text{FW}$ (electrons). When the accumulated charge reaches `full_well_e`, additional photons contribute no further electrons — the pixel saturates. This is modelled in the ADC stage as hard clipping:

$$\text{signal}_\text{clipped} = \min(e_c, Q_\text{FW})$$

Typical values range from ~2000 e⁻ for 1.22 µm phone pixels to 80 000 e⁻ for medium-format sensors with 5.3 µm pitch.

Full well capacity scales approximately as pixel area:

$$Q_\text{FW} \approx \rho_\text{FW} \cdot p^2$$

where $\rho_\text{FW} \approx 1000\text{–}3000$ e⁻/µm² depending on technology generation and fill factor.

---

## 8. Dark Current

Even in complete darkness, thermal energy generates electron–hole pairs in the silicon depletion region. The **dark current** $I_\text{dark}$ [e⁻/s] adds electrons to each pixel proportionally to integration time:

$$e_\text{dark} = I_\text{dark}(T) \cdot t_\text{int}$$

Dark current follows the **Arrhenius model** with silicon trap-generation activation energy $E_a \approx 0.63$ eV:

$$\frac{I_\text{dark}(T)}{I_\text{dark}(T_0)} = \exp\!\left[\frac{E_a}{k_B}\!\left(\frac{1}{T_0} - \frac{1}{T}\right)\right]$$

where $T$ is sensor temperature (K), $T_0$ is the reference temperature, and $k_B = 8.617 \times 10^{-5}$ eV/K.

Dark current roughly doubles every 6–8 °C — which is the legacy `dark_current_doubling_per_c` approximation. The Arrhenius model is more accurate and is the default when `dark_activation_energy_eV: 0.63` is set (non-zero).

See [Noise Model](03_noise_model.md) for how dark current contributes to shot noise.

---

## See Also

- [Radiometry](01_radiometry.md) — how irradiance is derived from PBRT radiance
- [Noise Model](03_noise_model.md) — statistical noise on top of the electron signal
- [Optics & PSF](04_optics.md) — spectral blur before electron integration
