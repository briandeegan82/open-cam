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
  exact pixel shifts with no ringing. Fractional splits are a band-limited interpolation, whose small undershoot at hard edges is clipped and reported. The
  filter removes detail below the pixel Nyquist frequency and the CFA colour aliasing it causes.
  It cannot remove luma aliasing that pbrt has already folded into the pixel grid above Nyquist.
  Doing that would need a supersampled render followed by pixel integration.
- Light spread past the image border is lost (at most d px at the right and bottom edges).
- Not modelled: the spread of incidence angles inside the lens cone (only the chief ray is
  used), the IR-cut coating and cover-glass of the filter stack (the IRCF is a QE curve), and the
  focus shift the plate causes.

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
