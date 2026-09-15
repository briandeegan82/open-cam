# Tutorial 02 -- PSF, Diffraction and Aberrations

**Demo:** `opencam-gui demo optics`
**Audience:** graduate / advanced undergraduate; Tutorial 01 assumed
**Goal:** connect lens f-number and pixel pitch to the actual post-render PSF stage in
`tools/apply_spectral_psf.py` -- diffraction, geometric aberration, lateral
chromatic aberration, and the stray light that destroys contrast without
blurring anything.

Tutorial 01 treated the lens as perfect: one object point, one image point.
This one takes that point and spreads it.

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

### 1.5 Stray light: contrast loss without blur

Everything above is convolution: energy stays local, and the PSF describes
exactly where it went. Stray light is the part that does not stay local --
scattered off glass surfaces, off the barrel, off the sensor cover. It barely
changes the PSF, and it destroys contrast anyway. `apply_stray_light` in
`tools/apply_spectral_psf.py` models four mechanisms, and the demo exposes them
separately:

- **Veiling glare.** A uniform pedestal proportional to total scene flux, added
  everywhere. It does not blur an edge at all -- it lifts the black level, so
  local contrast falls while the *shape* of the edge is untouched. A 5% veil on
  a 1000:1 scene leaves you with about 20:1.
- **Halo.** A broad, low-amplitude Gaussian skirt around bright sources, from
  wide-angle scatter. Same energy path as the PSF but with a sigma tens of times
  larger.
- **Ghost reflection.** Light bouncing between two element surfaces arrives back
  at the sensor inverted through the optical axis, so a bright source in one
  corner puts a faint copy of itself in the opposite one.
- **Aperture diffraction spikes.** An *n*-blade iris diffracts a point source
  into a starburst: `n` spikes for even *n*, `2n` for odd *n*, because an even
  polygon's opposite edges are parallel and their spikes coincide. A 7-blade iris
  gives 14 spikes; a 9-blade one gives 18.

Contrast is measured here as Michelson contrast, `(max - min)/(max + min)`, on a
profile across a high-contrast edge. It is the right metric because it is exactly
what stray light attacks: the denominator survives, the numerator does not.

---

## 2. UI map

| Panel / control | What it shows |
| --- | --- |
| Per-channel table | lambda, sigma_diff, sigma_geom, sigma_total, Airy rho0 for R/G/B |
| Radial PSF profile | Azimuthally-averaged intensity vs radius, one line per R/G/B |
| 2-D PSF kernel | The actual convolution kernel (G channel) as a heatmap |
| Lateral-CA test chart | A ring/spoke chart before and after per-channel `apply_lateral_ca` |
| Stray light tab: before / after | The high-contrast test source with and without stray light |
| Stray light tab: starburst kernel | The `n`-blade aperture diffraction PSF on its own |
| Stray light tab: edge profile | A cut across the high-contrast edge, clean vs strayed |
| Stray light tab: contrast readout | Michelson contrast before and after, and the loss in stops |
| Status line | f-number, pixel pitch, mode, diffraction-limited verdict |

**Core sliders:** f-number, pixel pitch, PSF mode, stray light on/off.
**Advanced:** geometric aberration sigma, lateral CA coefficient, veiling glare fraction,
halo strength and sigma, ghost reflection on/off and strength, aperture blade diffraction
on/off, iris blade count, starburst strength, blade rotation.
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

### Experiment F -- Veiling glare destroys contrast without blurring (~10 min)

1. Switch to the **Stray light** tab and load **Uncoated lens: veiling glare
   kills contrast**.
2. Read the Michelson contrast before and after. Then look at the edge profile:
   the transition is just as *steep* as before -- the black level has been lifted,
   not the edge softened.
3. Sweep the veiling glare fraction from 0 to 0.1 and watch contrast collapse
   while the edge shape stays put.

**Ask the room:** "If the PSF is unchanged, will an MTF measurement catch this?"
(It will, as a scale factor on the whole curve -- which is why MTF is normally
quoted after black-level correction, and why that convention hides veiling glare.)

### Experiment G -- Sunstars and the blade count (~8 min)

1. Load **Sunstars: blade count sets the spike count** and count the spikes on
   the starburst kernel.
2. Step the blade count from 6 to 9 and count again. Even counts give *n* spikes,
   odd counts give 2*n*, because an even polygon's opposite edges are parallel and
   their spikes land on top of each other.
3. Rotate the blades and watch the whole pattern rotate with them.

### Experiment H -- Ghosts are geometric, not random (~6 min)

1. Load **Backlit shot: ghost reflection through the optical axis**.
2. Find the ghost. It sits diametrically opposite the source through the frame
   centre -- move the source and the ghost moves the other way.
3. That predictability is why ghosts can sometimes be removed in software and
   veiling glare essentially cannot.

---

## 4. Self-paced lab sheet

| Task | Measurement | Your value |
| --- | --- | --- |
| f/2.8, 1.0 um pitch, `airy_disk`: blue Airy `rho0` | pixels | |
| Same, red Airy `rho0` | pixels | |
| f/1.8 vs f/16, same pitch: `sigma_diff` ratio (blue channel) | dimensionless | |
| `sigma_geometric` = 0.5 px: f-number at which blue becomes diffraction-limited | f/N | |
| lateral CA = 0.04: at what radius does fringing become clearly visible? | qualitative | |
| Veiling glare 0.05: Michelson contrast before and after | two values | |
| Same, expressed as a loss | stops | |
| 7-blade vs 8-blade iris: spike count | two integers | |
| Halo sigma 4 px vs 40 px: which changes the edge profile more? | circle one | |

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
5. Veiling glare leaves the PSF essentially unchanged but ruins the picture.
   What quantity is it attacking, and why does the PSF not describe it?
6. An 8-blade iris gives 8 spikes and a 9-blade iris gives 18. Explain the
   factor of two.
7. Lens coatings cost money and add manufacturing steps. Which of the four
   stray-light mechanisms do they address, and which survive regardless?

---

## 6. Bridge to the next tutorial

You have now seen the blur as a picture -- a kernel, a radial profile, a fringed
test chart. What you cannot do yet is compare two lenses, or check either
against theory, because "looks blurrier" is not a number. The next tutorial
measures the same PSF the way the standards do: a slanted edge, an oversampled
edge profile, and an MTF curve you can put a diffraction prediction on top of.
