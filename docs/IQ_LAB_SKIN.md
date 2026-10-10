# IQ lab: skin-tone tests

- `tools/iqlab/skin.py` provides:
  - Skin spectra: a melanin series synthesised on the measured ColorChecker light-skin reflectance, `R = R_base · exp(−τ (λ/500 nm)^−3.33)`. The exponent is the melanosome absorption power law (Jacques 2013), applied twice because light crosses the epidermis twice. τ = −0.25 … 3.4 gives ITA ≈ +61° (very light) to ≈ −33° (dark) under D65. The haemoglobin bands and dermal scattering come from the measured base spectrum.
  - Metrics: CIEDE2000 and ΔL*/ΔC*/Δh, plus the individual typology angle `ITA = atan2(L*−50, b*)` (Chardon et al. 1991).
- `tools/build_skin_tone_chart.py` builds a 12-patch pbrt chart with one scene per illuminant (default D65, A, F11, LED_B4). The patches are the 8 synthetic tones, the measured ColorChecker light and dark skin, 90 % white and 18 % grey. `chart.json` holds the ROIs and the spectral reference L*a*b* for each illuminant.
- `tools/run_skin_tone_test.py` scores a linear-sRGB render or capture. It measures patch means → XYZ → L*a*b*, using the chart's own white patch as reference white, so exposure and white balance drop out. It reports per-tone ΔE00, ΔL*, ΔC*, Δhue and ITA shift.

**Provenance.** The synthetic tones are physically motivated but are not measured population data. Swap in measured spectra (for example NIST skin reflectance) by editing `skin_tone_set`. Only the two ColorChecker patches are measured.

**Known offset.** pbrt's RGB film matches the 5 nm CIE reference to within 1 ΔE00 under D65. Under illuminant A it reads up to ~2 ΔE00 high in chroma on the darkest tones. This sets the noise floor of the chart when it is rendered through the RGB film.
