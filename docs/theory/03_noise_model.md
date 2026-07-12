# Noise Model — EMVA 1288 Noise Chain

This chapter describes every noise source in the pipeline, their statistical models, and how they combine in the EMVA 1288 framework. The implementation is in `tools/apply_emva_noise.py`, with analytic variance predictions in `tools/emva_theory.py`.

---

## 1. EMVA 1288 Overview

The [EMVA Standard 1288](https://www.emva.org/standards-technology/emva-1288/) defines a common methodology for characterising digital camera noise. It separates noise into temporal (shot-to-shot) and spatial (pixel-to-pixel fixed pattern) components, and provides a photon transfer curve (PTC) formalism that links all parameters.

**Key quantities:**

| Symbol | Name | Units |
|--------|------|-------|
| $\mu_e$ | Mean signal electrons | e⁻ |
| $\sigma_d$ | Temporal dark noise (read noise) | e⁻ |
| $K$ | Overall system gain | e⁻/DN |
| $\sigma_q$ | Quantisation noise | DN |
| $\sigma_\text{DN}$ | Total temporal noise in DN | DN |
| $\text{PRNU}$ | Photo-response non-uniformity | fraction (e.g. 0.01 = 1%) |
| $\text{DSNU}$ | Dark signal non-uniformity | e⁻ |

---

## 2. Photon Shot Noise

Shot noise arises from the discrete, Poisson-distributed arrival of photons. The variance in collected electrons equals the mean:

$$\sigma_\text{shot}^2 = \mu_e + \mu_\text{dark}$$

where $\mu_\text{dark} = I_\text{dark} \cdot t_\text{int}$ is the mean dark charge. Both photo-electrons and dark electrons arrive independently from Poisson processes; their variances add, so the total signal (including dark) is drawn from a Poisson distribution with mean $\mu_e + \mu_\text{dark}$.

In the pipeline (`apply_emva_noise.py`):

```python
# Combined Poisson draw for photo + dark electrons
total_mean = signal_e + dark_mean_e
noisy_e = rng.poisson(total_mean).astype(np.float32)
```

This correctly models the fact that dark shot noise scales with $\sqrt{\mu_\text{dark}}$, which is non-negligible at high temperatures or long exposures.

---

## 3. Temporal Dark Noise (Read Noise)

Read noise $\sigma_d$ represents all noise sources that do not scale with signal: amplifier thermal noise, reset noise after correlated double sampling (CDS), and quantisation. It is modelled as a zero-mean Gaussian:

$$n_d \sim \mathcal{N}(0, \sigma_d^2) \quad [\text{e}^-]$$

Typical values:
- 1.5–2.5 e⁻ for modern BSI CMOS (phone/mirrorless)
- 2.5–4.0 e⁻ for larger-pixel DSLR sensors
- >5 e⁻ for old CCD or high-gain sensors

---

## 4. kTC Reset Noise

When a capacitor is reset (disconnected from a voltage reference), thermal agitation leaves a residual charge with variance:

$$\sigma_\text{kTC}^2 = \frac{k_B T C}{q^2} = k_B T / (K \cdot q) \quad [\text{e}^-]^2$$

Modern CMOS sensors use **correlated double sampling (CDS)**: the reset level is sampled and subtracted from the signal level, cancelling kTC noise. CDS is the default for rolling-shutter BSI CMOS — hence `ktc_noise.enabled: false` in the default model.

Global-shutter sensors and some older CCDs may not have CDS, in which case kTC noise must be included. Enable via:

```yaml
noise:
  emva:
    ktc_noise:
      enabled: true
      node_capacitance_fF: 5.0   # optional; estimated from K if absent
```

---

## 5. Dark Current and Arrhenius Model

Dark current $I_\text{dark}(T)$ [e⁻/s] is the mean rate of thermally-generated electrons per pixel. The reference value at temperature $T_0$ is `dark_current_e_per_s`. At other temperatures it follows the Arrhenius model:

$$\frac{I(T)}{I(T_0)} = \exp\!\left[\frac{E_a}{k_B}\!\left(\frac{1}{T_0} - \frac{1}{T}\right)\right]$$

- $E_a = 0.63$ eV — silicon trap-generation activation energy
- $k_B = 8.617333262 \times 10^{-5}$ eV/K — Boltzmann constant
- Temperatures in Kelvin: $T[\text{K}] = T[\text{°C}] + 273.15$

The dark current doubles roughly every 6–8 °C. At $T_0 = 20\,\text{°C}$ and $T = 40\,\text{°C}$:

$$I(40°C)/I(20°C) = \exp\!\left(0.63/8.617\!\times\!10^{-5} \cdot (1/293.15 - 1/313.15)\right) \approx 4.3\times$$

**Legacy fallback**: setting `dark_activation_energy_eV: 0.0` uses the simpler doubling rule:

$$I(T) = I(T_0) \cdot 2^{(T - T_0)/\Delta T_\text{double}}$$

where `dark_current_doubling_per_c` defaults to 6.0 °C. The Arrhenius model is preferred.

---

## 6. PRNU — Photo-Response Non-Uniformity

PRNU is a spatially fixed, multiplicative non-uniformity arising from pixel-to-pixel variation in photodiode area, fill factor, micro-lens alignment, and CFA dye thickness. It is characterised by its standard deviation as a fraction of the mean signal:

$$\sigma_\text{PRNU} = \text{PRNU} \cdot \mu_e$$

where PRNU is the fractional standard deviation (e.g. 0.01 for 1%). PRNU is modelled as a zero-mean Gaussian gain map:

$$g_\text{PRNU}(x,y) = 1 + \mathcal{N}(0, \text{PRNU}^2)$$

clipped to $\geq 0$ (gain cannot be negative). The pixel signal becomes $e_\text{effective} = g_\text{PRNU} \cdot e_c$.

### Per-Channel CFA PRNU

Different CFA dyes have different uniformity. The pipeline supports per-channel PRNU standard deviations via:

```yaml
noise:
  emva:
    prnu_std_fraction: 0.01       # global fallback
    prnu_std_fraction_r: 0.012    # red channel
    prnu_std_fraction_g: 0.009    # green channel
    prnu_std_fraction_b: 0.013    # blue channel
```

Four independent PRNU maps are generated, one per Bayer phase (R, Gr, Gb, B).

---

## 7. DSNU — Dark Signal Non-Uniformity

DSNU is the pixel-to-pixel variation in dark current (due to variation in defect density). Unlike PRNU it is an additive fixed offset, not multiplicative.

The distribution of dark current across pixels is **log-normal** (not Gaussian). Dark defect sites are rare and generate large positive offsets (hot pixels follow a power-law tail). The log-normal distribution captures this asymmetry:

$$d \sim \text{LogNormal}(\mu_\text{ln}, \sigma_\text{ln})$$

where the log-normal parameters are derived from the desired mean $\mu_\text{DSNU}$ and standard deviation $\sigma_\text{DSNU}$:

$$\sigma_\text{ln} = \sqrt{\ln\!\left(1 + \left(\frac{\sigma_\text{DSNU}}{\mu_\text{DSNU}}\right)^2\right)}$$

$$\mu_\text{ln} = \ln(\mu_\text{DSNU}) - \frac{\sigma_\text{ln}^2}{2}$$

The zero-mean DSNU offset applied to each pixel is $d - \mu_\text{DSNU}$, so the mean offset over the array is zero.

Parameters in camera model YAML:
```yaml
noise:
  emva:
    dsnu_std_e: 0.3       # DSNU standard deviation [e-]
    dsnu_mean_e: 0.3      # DSNU mean [e-]; defaults to dsnu_std_e if absent
```

---

## 8. Hot Pixels and Stuck Pixels

**Hot pixels** have abnormally high dark current (often $10\times$–$100\times$ the typical pixel). They arise from crystal defects and increase with radiation damage (e.g. in space or astronomy applications).

The pipeline models hot pixel dark current excess with an **exponential distribution**:

$$e_\text{hot,excess} \sim \text{Exp}\!\left(\frac{e_\text{hot,max} - e_\text{hot,min}}{3}\right)$$

clipped to $[e_\text{hot,min},\, e_\text{hot,max}]$. The exponential distribution correctly models the heavy right tail of the defect population.

**Stuck pixels** are permanently saturated (output always at maximum DN) or permanently dark (output always zero). Their fraction is set by `hot_pixel_fraction` and `stuck_pixel_fraction` in the camera model.

---

## 9. Fixed Pattern Noise — Row and Column

Row and column FPN arises from variations in the row/column amplifier circuits and bias lines. It is modelled as:

$$n_\text{row}(y) \sim \mathcal{N}(0, \sigma_\text{row}^2)$$
$$n_\text{col}(x) \sim \mathcal{N}(0, \sigma_\text{col}^2)$$

These are added as constant offsets across each row and column respectively, and are the same for every frame (unlike read noise which varies frame-to-frame).

---

## 10. ADC — Analogue-to-Digital Conversion

The analogue electron signal is converted to a digital number (DN) by dividing by the system gain $K$:

$$\text{DN}_\text{ideal} = \frac{e_\text{total}}{K} + \text{black\_level}$$

### Quantisation Noise

The rounding to integer DN introduces uniform quantisation noise with variance:

$$\sigma_q^2 = \frac{1}{12} \quad [\text{DN}^2]$$

### Integral Non-Linearity (INL)

INL is a smooth, systematic deviation of the ADC transfer curve from its ideal straight line. It is modelled as a quadratic bow:

$$\Delta_\text{INL}(\text{DN}) = \text{INL\_fraction} \cdot Q_\text{FW} \cdot \frac{4 \cdot \text{DN}(Q_\text{FW} - \text{DN})}{Q_\text{FW}^2}$$

This peaks at mid-scale and is zero at black and white. `inl_quadratic_fraction` is the peak INL as a fraction of full scale.

### Differential Non-Linearity (DNL)

DNL is the variation in step size between adjacent ADC codes. It is modelled as independent Gaussian offsets per code:

$$\text{DNL}(i) \sim \mathcal{N}(0, \sigma_\text{DNL}^2) \quad [\text{LSB}]$$

---

## 11. Total Temporal Variance

Combining all uncorrelated temporal noise sources in quadrature (EMVA 1288 §4):

$$\sigma_\text{total,DN}^2 = \underbrace{\frac{\mu_e + \mu_\text{dark}}{K^2}}_\text{shot} + \underbrace{\frac{\sigma_d^2}{K^2}}_\text{read} + \underbrace{\frac{\sigma_\text{kTC}^2}{K^2}}_\text{kTC} + \underbrace{\sigma_q^2}_\text{quant}$$

This is the variance predicted by `emva_theory.temporal_variance_dn_squared()`.

The **photon transfer curve (PTC)** plots $\sigma_\text{DN}^2$ vs $\mu_\text{DN}$ (mean signal level). At low signal the y-intercept gives $\sigma_d^2/K^2 + \sigma_q^2$ (noise floor); the slope of the linear region gives $1/K$ (inverse system gain).

---

## 12. Noise Budget Example

For the default camera model at half-saturation ($\mu_e = 2400$ e⁻, $K = 4.77$ e⁻/DN, $\sigma_d = 2.6$ e⁻, $\text{bit\_depth} = 10$, $Q_\text{FW} = 4800$ e⁻):

| Source | Contribution [DN rms] |
|--------|----------------------|
| Shot noise | $\sqrt{2400}/4.77 = 10.2$ |
| Read noise | $2.6/4.77 = 0.55$ |
| Quantisation | $1/\sqrt{12} = 0.29$ |
| **Total** | **10.2** (shot dominated) |

Dynamic range = $Q_\text{FW} / (K \cdot \sigma_\text{total,e}) \approx 4800 / (4.77 \cdot 2.6) \approx 387:1 \approx 51$ dB.

---

## See Also

- [Validation](../guides/validation.md) — how to interpret PTC plots and EMVA reports
- [Sensor Physics](02_sensor_physics.md) — electron count before noise
- `tools/emva_theory.py` — analytic variance functions
- `tools/apply_emva_noise.py` — noise application implementation
