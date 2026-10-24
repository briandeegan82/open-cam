# Optical low-pass (anti-aliasing) filter

`tools/olpf.py` models the birefringent anti-aliasing filter mounted in front of many sensors. It
is opt-in under `sensor.olpf` and is off by default and in every shipped sensor model.

## Model

A quartz plate cut with its optic axis at θ to the plate normal splits each ray into an ordinary
and an extraordinary ray of equal power. The extraordinary ray walks off sideways by
`d = t·tan ρ` (Yariv & Yeh, *Optical Waves in Crystals*, sec. 4.3):

    tan ρ = (1 − n_o²/n_e²) tan θ / (1 + (n_o²/n_e²) tan² θ)

Two crossed plates (with a depolarising wave plate between them) make the usual four-spot filter.
Its point spread is 1/4 at (±dx/2, ±dy/2) and its transfer function is

    H(fx, fy) = cos(π fx dx) · cos(π fy dy)

The first zero sits at `f = 1/(2d)`, so a one-pixel split puts it at the pixel Nyquist frequency.
n_o(λ) and n_e(λ) of quartz come from Ghosh, *Opt. Commun.* 163, 95 (1999), the refractiveindex.info
SiO2 Ghosh-o/e data. These give n_o = 1.5442 and n_e = 1.5533 at 589 nm. A 45° cut walks off
5.9 µm per mm (6.03 µm at 450 nm, 5.84 µm at 650 nm).

## Configuration (`sensor.olpf`)

| key | meaning |
|---|---|
| `enabled` | default `false` |
| `split_pixels` | spot separation in pixel pitches on both axes (wavelength independent) |
| `split_x_um`, `split_y_um` | separation in µm (wavelength independent); 0 or absent = no split on that axis |
| `thickness_x_um`, `thickness_y_um` | quartz plate thickness; d(λ) = t·tan ρ(λ) |
| `axis_angle_deg` | optic-axis angle of the plates, default 45 |
| `exr_pixel_pitch_um` | EXR pixel pitch if it differs from `sensor.pixel_pitch_um` |
| `interpolation` | fractional-split weights: `box` (default, no ringing) or `fourier` (band-limited) |

Example for a 4 µm pixel with full Nyquist suppression on both axes: `{enabled: true, split_pixels: 1.0}`.
The same filter in quartz is `{enabled: true, thickness_x_um: 680, thickness_y_um: 680}`, giving a
1.00 px split at 550 nm.

## Where it runs

`pbrt_spectral_exr_to_electrons.spectral_radiance_to_electrons` applies the filter to the spectral
radiance cube before QE integration and the pixel angular response. That is the shared conversion
used by `run_pipeline.py` (`sensor_forward` `pbrt_exr`) and by `apply_emva_noise` (`integrate_qe`),
so the IQ-lab scorecard and the GUI pick it up from the sensor model. Each path applies it once.
The conversion metadata reports `olpf.split_px_range` and `olpf.clipped_energy_fraction`.
`tools/olpf.py --camera-model-config M --exr-in in.exr --exr-out out.exr` applies the filter to an
EXR on its own, for inspection only. Don't feed that output back through the conversion, or the
filter is applied twice.

## Limits

- pbrt has already integrated each EXR pixel over its area, so the filter acts as its transfer
  function on the sampled image. The spots sit at (0, 0), (d, 0), (0, d) and (d, d), as behind a
  real plate where the ordinary ray is undeviated, so the image moves by d/2 and integer splits are
  exact pixel shifts with no ringing. For fractional splits, `interpolation: box` (default) takes the
  irradiance as constant over each EXR pixel, so a spot d = n + f px away lands with weights (1 − f, f)
  on pixels n and n + 1. The kernel is non-negative, so it cannot ring, and it adds a little blur
  (MTF |½(1 + (1 − f)e^(−2πiνn) + f e^(−2πiν(n+1)))| instead of |cos πνd|). `interpolation: fourier` is the
  band-limited version with the exact cos OTF, but pbrt's box-filtered pixels are not band-limited:
  around a lamp or window hundreds of times brighter than the room, its ringing runs along the whole
  row and column. Its undershoot is clipped and reported. The
  filter removes detail below the pixel Nyquist frequency and the CFA colour aliasing it causes.
  It cannot remove luma aliasing that pbrt has already folded into the pixel grid above Nyquist.
  Doing that would need a supersampled render followed by pixel integration.
- Light spread past the image border is lost (at most d px at the right and bottom edges).
- Not modelled: the IR-cut coating and cover-glass of the filter stack (the IRCF is a QE curve).
  The spread of incidence angles inside the lens cone and the plates' focus shift are opt-in
  (`cone`, below).

## Angle of incidence (`angle_dependence`, opt-in)

Off axis, light reaches the plates at the chief-ray angle (CRA), and the walk-off changes. With
`sensor.olpf.angle_dependence: {enabled: true, cra: linear|table, max_cra_deg: 25, field_grid: [5, 7]}`,
each quartz plate is traced exactly (`olpf.plate_split_um`):
- the ordinary ray refracts with n_o;
- the extraordinary wave vector k satisfies `(k.c)^2/n_o^2 + (|k|^2 - (k.c)^2)/n_e^2 = 1` for the
  plate's optic axis c, and its ray runs along the gradient of that surface;
- the split is the difference between the two rays' exit points. At normal incidence it is
  exactly t tan(rho).

Off axis the split also gains a component across the plate's own axis, so the four spots become
a parallelogram. The image is filtered per field zone with that zone's two 2-D split vectors,
blended with bilinear weights over `field_grid`.

The CRA comes from `lens.chief_ray_angle_table` (`cra: table`) or rises linearly to
`max_cra_deg` at the frame corner (`cra: linear`). Only the `thickness_*` (quartz) filters have
an angle dependence; `split_*` values are taken as given.

Example: 680 µm plates at 550 nm, nominally a 1.00 px split on a 4 µm pixel. At 20° CRA the split
along the plate's axis is 0.88 px on the side tilted toward the optic axis and 1.12 px on the
other side. Tilting across the plate's axis adds a 0.35 px cross-axis component.

## Light cone and focus shift (`cone`, opt-in)

A pixel does not receive only the chief ray, but a cone of rays with image-space numerical aperture
1/(2N) at the working f-number N. The walk-off depends on the incidence angle, so each ray's
extraordinary spot lands slightly differently. With
`sensor.olpf.cone: {enabled: true, samples: 16, f_number: null, focus_compensated: true}`:
- the split is traced (`olpf.plate_split_um`) for `samples` rays spread evenly over the pupil:
  direction sines in a disk of radius 1/(2N) around the chief ray, sunflower pattern
  (`olpf.pupil_sines`);
- the three displaced spots become clouds of 3 × `samples` spots with box-pixel weights, so the
  filter stays non-negative (`interpolation: box` only);
- `f_number` defaults to the camera's working f-number, the same one the radiance → irradiance
  conversion uses (the traced lens for `realistic` cameras);
- with `angle_dependence` on, each field zone uses its own CRA as the cone axis.

A plane-parallel plate also moves the focus back by t(1 − 1/n_o) for the ordinary ray (paraxial;
480 µm for two 680 µm plates). `focus_compensated: true` (default) takes the sensor to sit at that
shifted focus, as in a camera designed with its filter stack: the ordinary spot is the focus
reference and only the extraordinary rays' extra spread remains. `focus_compensated: false` models a
plate added behind a lens focused without it: every ray also lands off by its own ordinary-ray
offset, a defocus disc of diameter about t(1 − 1/n_o)/N (43 px at f/2.8 for either example below).
That is only meant for "filter added to a lens focused without it" studies. The conversion
metadata reports `olpf.cone.focus_shift_um` and `defocus_disc_px`.

Size of the cone effect at 550 nm. MTF is along the plate's split axis, chief ray vs the cone, from
the spot pattern alone (no pixel aperture). The two plates are the 680 µm / 4 µm and 240 µm / 1.4 µm
examples, both about a 1 px split:

| N | CRA | e-spot spread x / y (px RMS) | MTF at 0.25 cy/px | MTF at 0.4 cy/px |
|---|---:|---:|---:|---:|
| 1.8 | 0° | 0.05 / 0.14 | 0.705 → 0.693 | 0.304 → 0.292 |
| 1.8 | 25° | 0.07 / 0.15 | 0.737 → 0.713 | 0.408 → 0.371 |
| 2.8 | 0° | 0.03 / 0.09 | 0.705 → 0.700 | 0.304 → 0.299 |
| 2.8 | 25° | 0.04 / 0.09 | 0.737 → 0.728 | 0.408 → 0.393 |
| 5.6 | 25° | 0.02 / 0.05 | 0.737 → 0.735 | 0.408 → 0.404 |

The spread is the same for both plate examples to within 0.01 px, because it scales with the split. It
matters mostly at fast apertures off axis, where it lowers the MTF by up to 0.04 near the filter's
zero. The mean split hardly moves (by at most 0.012 px).

Cost on a 960 × 540, 16-bucket cube: 0.3 s for the chief-ray filter, 2.4 s with `cone` (16 rays),
9 s with `angle_dependence` alone (5 × 7 zones), 77 s with both.
