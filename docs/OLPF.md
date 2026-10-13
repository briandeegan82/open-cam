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
  function on the sampled image. This is exact for integer even splits; other splits are a
  band-limited interpolation, whose small undershoot at hard edges is clipped and reported. The
  filter removes detail below the pixel Nyquist frequency and the CFA colour aliasing it causes.
  It cannot remove luma aliasing that pbrt has already folded into the pixel grid above Nyquist.
  Doing that would need a supersampled render followed by pixel integration.
- Light spread past the image border is lost (at most d/2 px at each edge).
- Not modelled: the walk-off's dependence on angle of incidence (it varies with the chief-ray
  angle), the IR-cut coating and cover-glass of the filter stack (the IRCF is a QE curve), and the
  focus shift the plate causes.
