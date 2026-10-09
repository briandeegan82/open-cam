# Pixel angular response: CRA, microlens shift, lens and colour shading

Opt-in model in `tools/pixel_angular_response.py`. Default **off**: when the block is absent or
`enabled: false`, electrons are bit-identical to the plain QE integration
(`tests/test_pixel_angular_response.py::TestPipelineHook::test_disabled_is_bit_identical_to_main`).

## What it does

For every pixel and every wavelength bucket it computes a relative response `R(x, y, λ)` and
multiplies it into the spectral electron weights inside
`pbrt_spectral_exr_to_electrons.spectral_radiance_to_electrons`:

```
e_c(x, y) = Σ_λ L_λ(x, y) · w_c(λ) · R(x, y, λ)
```

That function is shared by `pbrt_spectral_exr_to_electrons.py` and by `apply_emva_noise.py`
(`integrate_qe`), so both paths behave the same. Because `R` depends on wavelength it produces
lens shading (G versus field) and colour shading (R/G, B/G versus field).

1. **Incidence-angle distribution** for each film point, weighted by projected solid angle:
   * `traced`: real skew rays traced from the film point through the pbrt-v4 lens prescription,
     clipped by every element aperture and the stop (pbrt `RealisticCamera` convention, film
     extent from the film `diagonal`). This gives the real chief ray, the real cone and pupil
     vignetting.
   * `table`: `lens.chief_ray_angle_table: [[image_height_norm, cra_deg], ...]` with a circular
     cone of NA `1/(2N)`.
   * `pinhole`: the CRA is the field angle from `fov_deg`, which pbrt applies to the shorter axis.
   * `auto` (default): `traced` if the lens model is `realistic` with a lensfile, otherwise
     `table` if a table is present, otherwise `pinhole`.
2. **Microlens and stack** (geometric optics). An ideal lenslet of aperture `D` and focal length
   `f` sits at height `d` above the silicon, in a stack of index `n_s`. Snell refraction into
   the stack moves the defocused spot (width `D·|1−d/f|`) by `d·tan θ_s`. The spot is blurred
   by a Gaussian approximation of diffraction, σ = 0.21 λ/NA [4], and integrated over a square
   collection window. Light is absorbed at depths following Beer–Lambert with α(λ) of
   crystalline silicon [5] (`spectra/silicon/si_green2008_nk.csv`). Below the surface it keeps
   walking off at the silicon refraction angle, so red light walks off more than blue: this is
   the colour-shading mechanism discussed by Agranov et al. [2] and Catrysse & Wandell [1].
   The fraction landing in each of the 8 neighbouring windows is also computed.
3. **Microlens shift** `c = −d·tan θ_s,design(h)` toward the optical centre:
   * `none`
   * `matched`: designed for this lens, i.e. the energy-centroid CRA of its traced incidence
     distribution
   * `linear`: CRA rises linearly to `max_cra_deg` at the frame corner (the usual datasheet
     specification)
   * `table`: `[[image_height_norm, cra_deg], ...]`
   Any design that differs from the lens leaves CRA-mismatch shading.
4. **Optical crosstalk** (optional, `crosstalk.enabled`). Light filtered by one pixel's CFA that
   lands in a neighbour's window is added to the neighbour's own CFA channel. Set
   `crosstalk.cfa_pattern` to the same value as `cfa.pattern`. This term is separate from, and
   additive to, carrier-diffusion crosstalk (`cfa.spatial_crosstalk`).

`normalize: normal_incidence` (default) divides by the response of an unshifted pixel to
collimated normal light, which is the condition the QE curve represents. `normalize: center`
instead makes the centre pixel 1.

## Enable it

Example recipe: `config/camera_recipes/research_realistic_wide22_cra.yaml`. Its sensor model is
`config/sensor_models/research_realistic_wide22_cra.yaml`, with a linear 20° microlens design on
the 22 mm wide lens. To add the block to any sensor model:

```yaml
sensor_forward:
  model:
    pixel_angular_response:
      enabled: true
      incidence: auto
      microlens_shift: {mode: linear, max_cra_deg: 20.0}
      stack: {stack_height_um: 2.0, stack_index: 1.55, window_fraction: 0.8, collection_depth_um: 3.0}
      crosstalk: {enabled: false, cfa_pattern: RGGB}
```

Scene manifests from `build_*_scene.py` record `lensfile`, `aperture_diameter_mm`,
`film_diagonal_mm` and `fov_deg`, and these are forwarded automatically (`camera_geometry`).
If a manifest has no film diagonal, pbrt's default of 35 mm is used, or set `film_diagonal_mm`
in the block. Example run:

```
venv/bin/python tools/apply_emva_noise.py --camera-model-config config/camera_recipes/research_realistic_wide22_cra.yaml ...
```

Validation figure and numbers:
`venv/bin/python tools/validate_pixel_angular_response.py --out out/cra_validation`

## Validation (tests/test_pixel_angular_response.py)

* The closed-form box⊛Gaussian window fraction matches adaptive quadrature to 1e-9 and matches
  the box-overlap limit.
* For a focused lenslet with no diffraction, the cutoff angle equals the analytic
  θc = asin(n_s·sin(atan(w/2d))).
* A matched shift restores the normal-incidence response exactly (no penetration). With gaps
  removed, own + neighbour fractions sum to 1.
* The closed form matches an independent Monte-Carlo ray model (exact Snell in stack and
  silicon, ideal thin lenslet, sampled diffraction and absorption depth) within 0.02, for own
  and crosstalk fractions at 0–30° and 450–650 nm.
* α(500 nm) = 1.11e4 cm⁻¹ and α(800 nm) = 850 cm⁻¹ (Green 2008 table).
* Traced lens: the on-axis projected solid angle equals π/(4N²) with `traced_f_number` within
  1%, and the real chief ray matches the paraxial exit-pupil CRA within 0.01° near the axis
  (wide_22mm and fisheye.10mm).

## Not modelled / approximations

* The microlens is ideal and thin (no aberrations, no Fresnel or AR-coating angular loss). A
  planar stack of one index is assumed.
* Diffraction is a Gaussian approximation. Below about 2 µm pitch, wave-optical effects
  (FDTD) dominate [3], so treat results there as qualitative.
* The colour filter is lumped at the microlens. Metal shadowing is lumped into
  `window_fraction`. There is no polarisation and no carrier diffusion (see
  `cfa.spatial_crosstalk`).
* `traced` mode traces at infinity focus. The CRA table and pinhole modes assume a circular
  cone, so they have no pupil vignetting; pbrt's realistic camera already renders natural
  vignetting into the EXR.
* The default stack values are illustrative, not measured; set them from the sensor's stack
  data. With default values the result is a physically consistent trend model, not a
  prediction for a specific product.

## References

1. P. B. Catrysse, B. A. Wandell, "Optical efficiency of image sensor pixels," J. Opt. Soc. Am. A 19(8), 1610–1620 (2002). doi:10.1364/JOSAA.19.001610
2. G. Agranov, V. Berezin, R. H. Tsai, "Crosstalk and microlens study in a color CMOS image sensor," IEEE Trans. Electron Devices 50(1), 4–11 (2003). doi:10.1109/TED.2002.806473
3. Y. Huo, C. C. Fesenmaier, P. B. Catrysse, "Microlens performance limits in sub-2µm pixel CMOS image sensors," Opt. Express 18(6), 5861–5872 (2010). doi:10.1364/OE.18.005861
4. B. Zhang, J. Zerubia, J.-C. Olivo-Marin, "Gaussian approximations of fluorescence microscope point-spread function models," Appl. Opt. 46(10), 1819–1829 (2007). doi:10.1364/AO.46.001819
5. M. A. Green, "Self-consistent optical parameters of intrinsic silicon at 300 K including temperature coefficients," Sol. Energy Mater. Sol. Cells 92, 1305–1310 (2008). doi:10.1016/j.solmat.2008.06.009 (data: refractiveindex.info, CC0)
