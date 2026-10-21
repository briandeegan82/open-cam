# Generic (non-Bayer) CFA and NIR / no-IRCF modelling

All of this is **opt-in**. Without `cfa.layout` the legacy 2x2 Bayer path of
`tools/apply_emva_noise.py` (`cfa.pattern` RGGB/BGGR/GRBG/GBRG, `bayer_sample_rgb`,
bilinear/Malvar demosaic) runs unchanged; `tests/test_cfa_bayer_regression.py` pins its
decoded outputs (SHA-256 goldens in `tests/data/bayer_path_golden.json`). Without the NIR pbrt
build (`PBRT_LAMBDA_MAX_NM`) pbrt keeps its stock 360-830 nm range.

## Generic CFA (`tools/cfa_mosaic.py`)

* **Layout**: any periodic NxM tile of channel names, or a preset
  (`RGGB`, `BGGR`, `GRBG`, `GBRG`, `RGBW`, `RGBW_4X4`, `RCCB`, `RCCC`, `RCCG`, `RYYCY`,
  `RGGCY`, `CMY`, `QUAD_BAYER`). Set in the sensor model under `cfa.layout`.
* **Per-channel spectral QE**: `cfa.channels.<name>.qe_csv` (wavelength_nm,qe CSV in
  `spectra/`), default mapping `cfa_mosaic.DEFAULT_CHANNEL_QE` (`W`/`C` = unfiltered
  `QE_mono.csv`, `Ye`/`Cy`/`Mg` = `QE_yellow/cyan/magenta.csv`). Electrons are integrated **per
  channel directly from the spectral EXR planes** (`processing.linear_exr_mode: integrate_qe`;
  `channel_qe` hook of `pbrt_spectral_exr_to_electrons.spectral_radiance_to_electrons`), then each
  pixel keeps the channel of its tile site -- no RGB proxy.
* **Noise per site**: shot noise acts on each site's own electrons; PRNU/DSNU/dark current per
  pixel; `site_parameter_map` allows per-channel `*_<CH>` overrides; full well and ADC are per pixel.
* **Crosstalk**: `cfa.spatial_crosstalk` (charge-conserving Gaussian diffusion with a per-source
  channel width `sigma_pixels_<CH>`), applied on the mosaic so charge moves between sites of
  different colours. The legacy `sensor.crosstalk.matrix_3x3` is RGB-only and is rejected in
  generic mode.
* **Quad Bayer**: `cfa.binning: charge` (2x2 same-colour charge sum before read noise: one read,
  4x full well) or `digital` (sum after readout: 4 reads, 2x read noise); `remosaic_to_bayer`
  re-arranges full-resolution Quad Bayer into an RGGB mosaic (nearest same-colour site).
* **Demosaic**: `cfa.demosaic: bilinear` (normalised convolution per channel; measured samples
  are preserved exactly) or `gradient` (Adams-Hamilton / Cok-style: interpolate the
  densest "guide" channel, then colour differences to it). References: B. E. Bayer, US 3,971,065
  (1976); D. R. Cok, US 4,642,678 (1987); J. E. Adams & J. F. Hamilton, US 5,629,734 (1997);
  B. K. Gunturk et al., "Demosaicking: color filter array interpolation", IEEE Signal Process.
  Mag. 22(1), 44-54 (2005).
* **Colour**: a C x 3 channel -> linear sRGB matrix (`cfa.ccm`): `colorchecker_spectral`
  (default: least squares on the 24 ColorChecker reflectances x illuminant x channel QE, white
  preserved; reuses `ccm_fit` / `color_science` targets) or `exr_reference` (fit on the rendered
  chart against the pbrt reference). XYZ via the sRGB (IEC 61966-2-1) matrix. CIEDE2000 per
  Sharma, Wu & Dalal, Color Res. Appl. 30(1), 21-30 (2005) (existing `color_science.delta_e2000`).

Recipes using it: `default_rgbw`, `default_rgbw_4x4`, `default_rccb`, `default_rccg`,
`default_ryycy`, `default_rggcy`, `default_cmy`, `default_quad_bayer` (charge binning),
`default_rgb_noircf`, `default_rccc_noircf` (`config/camera_recipes/`). **Change**: the former
RGB/QE-proxy versions of `default_rccb/ryycy/rccg/rggcy/cmy` are now true CFAs; the old proxy
behaviour is available by setting `cfa.layout: null` and `cfa.demosaic: malvar` in the sensor
model. Example: [`config/examples/cfa_custom_layout.yaml`](../config/examples/cfa_custom_layout.yaml).

## NIR and no-IRCF

* **What is required**: pbrt-v4 hard-codes `Lambda_max = 830` nm, and photometric light
  normalisation divides by the photopic integral (zero for an 850/940 nm source).
  `third_party/patches/0002-nir-lambda-max-radiometric-lights.patch` makes `Lambda_max` a
  compile-time macro `PBRT_LAMBDA_MAX_NM` (default 830, so the stock build is unchanged) and lets
  lights take `"bool photometric" false` (absolute radiometric `scale`). Build:
  `PBRT_LAMBDA_MAX_NM=1100 PBRT_BUILD_DIR=third_party/pbrt-v4/build-nir tools/build_pbrt.sh`.
  Scene: `tools/build_colorchecker_scene.py --film spectral --illuminant spectra/illuminant/nir/LED_850nm.csv --radiometric-light --lambda-max 1100 --spectral-lambda-max 1100 --pbrt-lambda-max 1100`.
  The manifest records `"photometric": false`, so no lux scale is applied; use
  `calibration.irradiance_scale_W_m2nm_per_unit`.
* **No IRCF**: `cfa.ircf: false` drops `sensor.quantum_efficiency.ircf_csv`.
* **NIR QE** (`tools/nir_spectra.py` -> `spectra/QE/nir/*_noircf.csv`, 300-1100 nm):
  `QE = (1 - R)(1 - exp(-alpha d)) eta_c` with intrinsic c-Si `n, k` from M. A. Green,
  Sol. Energ. Mat. Sol. Cells 92, 1305-1310 (2008) (`spectra/silicon/si_green2008_nk.csv`, via
  refractiveindex.info), d = 3 um, eta_c = 1, joined continuously to the measured visible curves
  between 650 and 800 nm. Colour filters are **assumed** IR-transparent above 800 nm (typical of
  organic pigment CFAs, not a measurement) -- so all channels respond about equally at 850/940 nm.
* **NIR LEDs**: `spectra/illuminant/nir/LED_850nm.csv`, `LED_940nm.csv` are Gaussian SPDs
  (peak / FWHM 850/30 nm and 940/40 nm, illustrative datasheet-typical values, not a specific
  product).

### Not modelled / approximations

* HDR pixel modes (DCG / split pixel / LOFIC / multi-exposure) run only on the legacy Bayer/RGB path; combining them with a generic `cfa.layout` raises an error.
* The X-Rite ColorChecker (`spectra/xrite`, 380-730 nm) and the skin chart built from its
  light-skin patch have no public measured NIR data; pbrt holds each patch at its 730 nm value, so
  their no-IRCF, clear-channel and 850/940 nm results are extrapolations. Use the measured
  **USGS NIR material chart** (below) when an NIR result has to rest on data.
* Si QE ignores the pixel stack (microlens, anti-reflection, deep-trench isolation), so NIR QE is an
  upper-bound-style model; real sensors with DTI/NIR enhancement differ.
* Lens transmission and the camera-specific IRCF are not extended beyond their CSV ranges.

## Validation

* `tests/test_cfa_mosaic.py`: noise-free exact mosaic sampling for every layout against a direct
  per-channel spectral integral; RGGB generic == legacy `bayer_sample_rgb`; demosaic preserves
  samples; crosstalk conserves charge; Quad Bayer charge vs digital binning SNR; CCM/CIEDE2000.
* `tests/test_nir_spectra.py`: Si absorption vs Green (2008) table values, Beer-Lambert limits,
  continuity, LED FWHM, committed curves reproduce, patch is opt-in.
* `tools/validate_cfa.py`: full pipeline on a real pbrt ColorChecker render (D65, default recipe,
  t = 0.12 s, patch 22) vs the `paper/ei2027/scripts/fig_cfa.py` analytic method on the same render:

| layout | mean e- @500 lux pipeline / analytic | luma/G pipeline (cfa_summary.json) | SNR 10 lux shot+read / full / analytic [dB] | CIEDE2000 mean / max |
|---|---|---|---|---|
| Bayer RGGB | 1424.7 / 1421.9 | 1.002 (1.000) | 13.05 / 13.45 / 13.61 | 4.10 / 8.46 |
| RGBW | 3737.8 / 3728.9 | 2.624 (2.622) | 18.48 / 18.69 / 18.35 | 4.06 / 8.70 |
| RCCB | 3713.1 / 3728.9 | 2.606 (2.622) | 18.41 / 18.13 / 18.35 | 4.23 / 8.65 |
| RYYCy | 2211.6 / 2205.7 | 1.552 (1.551) | 15.88 / 15.72 / 15.83 | 4.22 / 8.17 |
| Quad Bayer | 1425.2 / 1421.9 | 1.000 (1.000) | 12.85 / 13.13 / 13.61 | 4.05 / 8.44 |
| Quad Bayer 2x2 charge bin | 5686.0 / 5687.8 | 3.991 (-) | 21.29 / 20.02 / 20.31 | 4.35 / 8.31 |

The relative luma-channel gains agree with `cfa_summary.json` to < 0.7 %. Absolute electrons in
`cfa_summary.json` are 5.0x lower than the current calibration chain gives for the same
recipe/lux/t (the JSON predates later radiometry changes on main), so its absolute SNRs are not
reproduced; the pipeline is compared with the same analytic formula on the current chain instead.
At 500 lux and 0.12 s the white patch exceeds the 4800 e- full well, so colour is evaluated at
100 lux (25 lux binned). SNR is from ~225 sites per patch (sampling uncertainty about +-0.4 dB).

## Measured NIR material chart (`spectra/usgs_nir`)

24 materials from the USGS Spectral Library v7 (Kokaly et al. 2017, USGS Data Series 1035,
doi:10.5066/F7RR1WDJ; public domain), ASD FieldSpec absolute reflectance, resampled to 1 nm over
350-2500 nm (the SWIR range is used by the InGaAs sensors in [SWIR.md](SWIR.md)). It uses the ColorChecker's 6 x 4 layout:

```bash
python tools/build_colorchecker_scene.py --patch-dir spectra/usgs_nir \
  --illuminant spectra/illuminant/nir/LED_850nm.csv --radiometric-light \
  --lambda-max 1100 --pbrt-lambda-max 1100 --spectral-lambda-max 1100
```

Rows: leaves, pine needles, dry grass, plywood, burlap; red, green, blue and black nylon, white
cotton; green tarp, red vinyl, yellow HDPE, blue pipe, black HDPE, white PVC; painted aluminium,
red and grey brick, road concrete, road asphalt, black tar paper, Spectralon. Their red edges differ
strongly: the fresh oak leaf goes from 0.11 at 650 nm to 0.85 at 850 nm, and the green tarp from
0.12 at 730 nm to 0.61 at 850 nm, while black HDPE, grey brick and Spectralon stay flat.
`tools/nir_materials.py` regenerates the CSVs from the sha256-checked archive (cached, gitignored);
`tests/test_nir_materials.py` checks coverage, the red edges and the `--patch-dir` scene build.

Limits: single field samples (not population means); the curves are hemispherical-directional
reflectance used as Lambertian albedo; none of the materials is a colour reference like the
ColorChecker, so ΔE-type colour metrics still use `spectra/xrite`.


## In-cabin driver monitoring at 940 nm (`tools/build_cabin_scene.py`)

A night-time DMS scene: camera on the steering column ~0.65 m from the driver, looking up at the
face, lit only by a 940 nm (or `--led-nm 850`) LED next to the lens. It needs the NIR pbrt build
(`PBRT_LAMBDA_MAX_NM=1100`); the film is spectral, 400–1100 nm.

```bash
python tools/build_cabin_scene.py --out-dir scenes/cabin_dms --skin light --gaze-yaw-deg 10 --glasses
cd scenes/cabin_dms && ../../third_party/pbrt-v4/build-nir/pbrt cabin.pbrt
```

- **LED:** Lambertian disc (`--led-radius-mm`, default 2.5) with on-axis radiant intensity
  `--led-w-sr` (default 0.5 W/sr, nominal) in absolute units. Patch 0015 lets `diffuse` area
  lights take `"bool photometric" false`; without it pbrt divides by the ~0 photometric integral.
  A disc rather than a point light is needed so the cornea can mirror it.
- **Skin:** measured ColorChecker skin (`--skin light|dark`) up to 730 nm, then the measurement
  times the relative shape of `tools/nir_skin.py` (two-layer melanin/blood/water model after
  Jacques, OMLC; water: Hale & Querry 1973; haemoglobin: Prahl). The model's own absolute level is
  too dark against the measured patches in the red (≈0.48 vs 0.73 at 730 nm), so only its shape is
  used. Result: R(940) ≈ 0.66 (light), ≈ 0.24 (dark), with the water dip at 970 nm.
- **Eyes:** sclera sphere, iris and pupil discs (`--pupil-mm`, default 6 mm at night) and a
  dielectric cornea cap (η = 1.376, radius 7.8 mm). `--gaze-yaw-deg/--gaze-pitch-deg` rotate the
  eyes; `--glasses` adds flat η = 1.5 spectacle lenses 20 mm in front of the eyes. By default
  (`--glasses-model thin`) each lens is one `thindielectric` surface with `"bool passthrough"
  true` (pbrt patch 0017, volpath integrator only). Shadow rays pass it with the two-face
  transmittance 1 − R_slab(θ) (0.923 at normal incidence, incoherent multiple reflections). A
  path that hits it is reflected with probability R_slab, and otherwise continues as through a
  null interface, so the light-sampling and BSDF-sampling estimates stay consistent. The eye
  behind the lens is then lit by the LED through the lens, as in a real car. Not modelled:
  lens power (prescription), AR coatings and NIR-absorbing tints, and the lateral offset of a
  2 mm slab. `--glasses-model slab` keeps the old 2 mm `dielectric` box. Shadow rays can't pass
  that, so the eyes are lit only by BSDF paths that happen to hit the small LED, and render
  dark and noisy. The manifest lists eye, cornea and pupil
  centres (world space) as gaze ground truth.
- **Corneal glint:** the virtual image of a 2.5 mm LED in the cornea is ~0.015 mm, far below a
  pixel (~0.7 mm on the face), so a smooth cornea is almost never hit by a camera sample (a few
  scattered bright samples at 128–1024 spp, or none). `--cornea-roughness` (GGX α, default 0.02)
  spreads the reflection over ~1–3 px, where pbrt samples it with next-event estimation. The total
  glint power stays the mirror-sphere value J = ρ E R²/4 (pbrt within 3 % for α = 0.02–0.05 on an
  isolated cornea, `tests/test_cabin_dms.py`; in the full scene with black eye materials the glint
  centroid is within 0.25 px of `glint.pixel_xy` and its power 1.01–1.03× the analytic value), so a sensor/lens model downstream sees the right
  energy at the right place. The blob is wider than the true sub-pixel glint; `--cornea-roughness 0`
  restores the smooth cornea. The manifest gives each eye's analytic glint (`glint.pixel_xy`,
  `glint.camera_irradiance_w_m2`, Fresnel ρ, `on_cornea`) and `pupil_pixel_xy` for pupil-centre /
  corneal-reflection gaze ground truth. Only the first Purkinje image is modelled; there is no lens
  inside the eye. The rough cornea also lets next-event estimation light the iris through it, so
  the iris shows bright render-noise specks at 512 spp that converge with more samples (with a
  smooth cornea the iris is lit only by rare paths that hit the LED and stays under-converged).
- **Retina / bright pupil** (`--retina on`, default; `off` writes the old scene byte for byte):
  the pupil disc is a `retroreflective` material (patch 0001) with a 2 % diffuse base. The LED is
  imaged, defocused, on the fundus; the fundus reflects it diffusely and the part leaving through
  the pupil, (r_p / 20 mm)² of it, returns towards the LED after two passes through 22 mm of
  water-equivalent ocular media (Hale & Querry; at 940 nm the round trip transmits 0.31, at 850 nm
  0.83). The returning beam spreads over observation angles up to
  α_e = (r_p + r_LED)/d + r_p |1/d − 1/f_eye| (pupil footprint, LED size, defocus;
  `--eye-focus-m`, default infinity, i.e. looking at the road). pbrt's lobe is exponential,
  R_A = ra exp(−α/w) √(cos θi cos θo), with w = α_e/3 (same mean angle as a uniform disc of radius
  α_e) and ra = P/(2π w²), so it integrates to the returned fraction P. For the default 6 mm pupil
  at 0.65 m, α_e ≈ 0.81° and w ≈ 0.27°. With the LED 40 mm from the lens (3.8° observation angle,
  the default) the pupil stays dark (retroreflection 1e-5 of a white diffuser); with
  `--led-offset-m 0.008` (0.77°) it is ~0.9× a white diffuser, the bright pupil. In a 256 spp render
  the bright pupil is 1.24× the cheek (R ≈ 0.66), i.e. ~0.8× white. The manifest gives
  `driver.retina` and each eye's `bright_pupil` (observation angle, R_A, pupil radiance).
  So that light reaches the pupil, the cornea gets `"bool passthrough" true` (patch 0020):
  shadow rays pass it with the single-interface transmittance 1 − Fr(θ), and paths either reflect
  (glint) or continue unrefracted. This also lights the iris properly through the cornea. The cost
  is that the cornea no longer refracts: the pupil disc stands for the entrance pupil (the
  apparent pupil, ~13 % larger than the anatomical one). Fundus reflectance (~10 % in the NIR,
  `--fundus-scale`) and the eye distances are nominal, not measured; the fundus is spectrally flat
  above 700 nm, with no pigmentation differences between drivers.
- **Sun** (`--sun-elevation-deg`, `--sun-azimuth-deg`; default none, night): an ASTM G173 direct
  normal spectrum (AM1.5) distant light, radiometric, plus a roof panel above the head; the
  windscreen and side windows are open (no glass, no NIR-absorbing windscreen) and there is no sky
  light. At 940 nm the sun is in a water band but still lights the face ~275× more than the LED in
  the sample (cheek radiance), so a DMS needs a short pulsed exposure or a narrow band-pass.
- **Nominal / not modelled:** head and eye geometry, sclera/iris/pupil reflectances, eye optics and
  LED output are nominal. No eyelids/brows/hair, no subsurface blur in skin.
