# Tutorial 01 -- Optics / Point-Spread Function

**Demo:** `opencam-gui demo optics`
**Audience:** graduate / advanced undergraduate, camera or imaging background helpful but not required
**Goal:** connect lens f-number and pixel pitch to the actual post-render PSF stage in
`tools/apply_spectral_psf.py` -- diffraction, geometric aberration, and lateral
chromatic aberration.

Every number and image in this demo comes from importing and calling the real
functions in `tools/apply_spectral_psf.py` (`psf_sigma_chromatic`,
`airy_disk_convolve`, `separable_gaussian_blur_2d`, `apply_lateral_ca`). The
GUI does not re-derive the optics -- it only wires sliders to those functions
and plots the results.

---

## 1. Principles

### 1.1 Two blur sources, added in quadrature

Open Cam's post-PSF stage models the lens as two independent blur
contributions:

- **Diffraction**, from the finite aperture. For an f-number *N* at
  wavelength lambda (nm) and pixel pitch *p* (um), the equivalent Gaussian sigma is

  sigma_diff(lambda) = 0.437 x N x lambda / (1000 x p)

  (0.437 = 1.028 / 2.355 converts the Airy-disk FWHM to an equivalent Gaussian sigma.)

- **Geometric aberration** (defocus, coma, field curvature, ...), modelled as
  a single constant `sigma_geometric_pixels`, independent of wavelength.

Combined in quadrature: `sigma_total(lambda) = sqrt(sigma_diff(lambda)^2 + sigma_geometric^2)`.

### 1.2 Two ways to render the diffraction term

- `chromatic_gaussian` -- collapses diffraction to the Gaussian sigma above.
  Fast, but loses the Airy disk's ring structure.
- `airy_disk` -- convolves with the physically correct 2-D Airy pattern
  (first-zero radius `rho0 = 1.22 x N x lambda / (1000 x p)`), combined with the
  geometric Gaussian in the Fourier domain. Slower, keeps the central lobe and
  first rings.

The demo's 2-D kernel panel and radial-profile plot are the literal output of
whichever mode you select, run through a single-pixel delta image -- convolving
a delta with a kernel returns the kernel itself.

### 1.3 Lateral chromatic aberration

An uncorrected lens magnifies different wavelengths by different amounts:

M(lambda) = 1 + lca_coefficient x (lambda - lambda_ref) / lambda_ref

Long wavelengths (red) magnify outward relative to `lambda_ref` (550 nm
default); short wavelengths (blue) pull inward. Applied per-channel to the
same scene, this produces the colour fringing visible on high-contrast edges
in real uncorrected lenses -- worst toward the image periphery.

### 1.4 Diffraction-limited vs aberration-limited

A lens is **diffraction-limited** when `sigma_diff >= sigma_geometric` at the
shortest wavelength you care about (blue diffracts least, so it is the
hardest channel to blur away aberration under) -- past that point, stopping
down the aperture further or improving the lens design no longer buys
sharpness; only a larger aperture (smaller *N*) or bigger pixels would.

---

## 2. UI map

| Panel / control | What it shows |
| --- | --- |
| Per-channel table | lambda, sigma_diff, sigma_geom, sigma_total, Airy rho0 for R/G/B |
| Radial PSF profile | Azimuthally-averaged intensity vs radius, one line per R/G/B |
| 2-D PSF kernel | The actual convolution kernel (G channel) as a heatmap |
| Lateral-CA test chart | A ring/spoke chart before and after per-channel `apply_lateral_ca` |
| Status line | f-number, pixel pitch, mode, diffraction-limited verdict |

**Core sliders:** f-number, pixel pitch, PSF mode.
**Advanced:** geometric aberration sigma, lateral CA coefficient.
**Camera recipe dropdown:** loads f-number / pixel pitch / `post_psf` settings straight
from a real `config/camera_recipes/*.yaml` via `tools/camera_model.load_camera_model`.

---

## 3. Guided walkthrough (lecture path)

### Experiment A -- Diffraction limit on a compact sensor (~8 min)

1. Load **Diffraction-limited compact sensor**.
2. Read the per-channel table: `sigma_diff` for blue should already exceed
   `sigma_geometric` (0.2 px) -- the lens is diffraction-limited out of the box.
3. Drop the pixel pitch slider toward 0.7 um and watch `sigma_total` grow in
   *pixel* units even though nothing about the optics changed.

**Ask the room:** "Did the lens get worse, or did the ruler change?"

### Experiment B -- Aberration vs diffraction trade-off (~8 min)

1. Load **Fast wide-aperture phone lens** (f/1.8, larger geometric term).
2. Raise `sigma_geometric_pixels` from 0.2 to 1.0 and watch which term
   dominates `sigma_total` in the table.
3. Switch PSF mode to `airy_disk` -- at f/1.8 the Airy disk is small, so the
   2-D kernel should look almost purely Gaussian (aberration-dominated).

### Experiment C -- Landscape diffraction limit (~8 min)

1. Load **Stopped-down DSLR (landscape diffraction limit)** (f/16, `airy_disk`).
2. Note the visible rings in the 2-D kernel panel.
3. Sweep f-number from f/4 to f/16: the central lobe visibly widens and rings
   spread -- the classic "diffraction softening" landscape photographers hit
   past f/11-f/16 on full-frame.

### Experiment D -- Lateral chromatic aberration (~8 min)

1. Load **Uncorrected achromat: visible lateral CA**.
2. Compare the greyscale input chart against the post-CA preview: look for a
   colour tint creeping in near the outer rings.
3. Push the lateral CA slider to its maximum and note the fringing intensify
   outward from the centre (`lambda_ref` = 550 nm sits at zero shift).

### Experiment E -- Airy vs Gaussian approximation (~8 min)

1. Load **Airy rings vs Gaussian approximation** (`airy_disk` mode, f/5.6).
2. Toggle PSF mode to `chromatic_gaussian` with the same sliders unchanged.
3. Compare the radial profile: `chromatic_gaussian` is a smooth monotonic
   falloff; `airy_disk` has a narrower central lobe plus small side lobes
   (rings) that the Gaussian approximation cannot represent.

---

## 4. Self-paced lab sheet

| Task | Measurement | Your value |
| --- | --- | --- |
| f/2.8, 1.0 um pitch, `airy_disk`: blue Airy `rho0` | pixels | |
| Same, red Airy `rho0` | pixels | |
| f/1.8 vs f/16, same pitch: `sigma_diff` ratio (blue channel) | dimensionless | |
| `sigma_geometric` = 0.5 px: f-number at which blue becomes diffraction-limited | f/N | |
| lateral CA = 0.04: at what radius does fringing become clearly visible? | qualitative | |

---

## 5. Check your understanding

1. Why is the blue channel used as the "hardest case" when calling a lens
   diffraction-limited?
2. `chromatic_gaussian` and `airy_disk` agree on `sigma_diff`. What do they
   disagree about, and why does it matter for MTF measurements?
3. Why does lateral CA fringing grow toward the image edges rather than
   appearing uniformly?
4. A phone lens (small pitch) and a full-frame DSLR (large pitch) share the
   same f-number. Which is more likely diffraction-limited, and why?

---

## 6. Bridge to the next tutorial

The PSF you just explored blurs light *before* it reaches the sensor. The
next tutorial picks up from there: how that same photon flux gets converted
to electrons and then to noisy digital numbers via the EMVA1288 noise model.
