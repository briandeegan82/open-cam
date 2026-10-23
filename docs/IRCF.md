# IR-cut filter

## Research camera: hybrid glass + coating (default)

The research-camera models (`default*`, `research_*`, `config/sensor_models/*`) apply
`spectra/QE/ircf/QE_IRCF_hybrid.csv` to the bare silicon × CFA QE in `spectra/QE/nir/*_noircf.csv`
(300–1100 nm). The filter is modelled the way phone and automotive IR-cut filters are usually built:
an IR-absorbing blue glass carrying an interference long-wave-cut coating
(`tools/ircf_model.py`):

    T(λ) = Ti_BG60(λ)^(d / 1 mm) · T_coat(λ)
    T_coat(λ) = t_pass · U(λ) · (floor + (1 − floor) / (1 + exp((λ − λc) / w)))

| Parameter | Value | Source |
|---|---|---|
| Glass | SCHOTT BG60, internal transmittance | SCHOTT filter glass calculation program 2024 (`spectra/QE/ircf/schott_BG60_taui_1mm.csv`) |
| Glass thickness d | 0.21 mm | illustrative |
| Coating edge width w | 2.5 nm | illustrative |
| Coating pass level t_pass | 0.97 | illustrative |
| Coating blocking floor | 0.01 (OD 2), 680–1100 nm | illustrative |
| Half-power point | 650 nm (λc solved, 654.9 nm) | kept from the legacy curve |
| UV edge U | legacy curve below 550 nm | `spectra/QE/interpolated/QE_IRCF.csv` |

Resulting transmittance: 0.93 at 550 nm, 0.80 at 600 nm, 0.48 at 650 nm, 0.06 at 660 nm, and a
0.1–0.3 % leak from 680 to 1100 nm, where the glass and the coating floor both block. Replace the
illustrative values with a measured curve if you have one:
`python tools/ircf_model.py --glass-thickness-mm … --edge-width-nm … --blocking-floor …`.

The interference part alone is `QE_IRCF_hybrid_coating.csv`, set as
`sensor.quantum_efficiency.ircf_coating_csv`. It is the part whose edge moves to shorter wavelengths
with incidence angle, so `pixel_angular_response.ircf_angle_shift` shifts only it (the absorbing
glass doesn't shift), and it is the reflecting face of the `ircf_cover_glass` sensor stack in
`tools/lens_ghosts.py` (R = 1 − T_coat; the glass absorbs rather than reflects). The glass's longer
path at oblique incidence (≈6 % at 30° in air) is ignored.

### Effect against the legacy curve (spectral, white patch, 380–1100 nm)

| Illuminant | Red signal | Green | Blue | R/G | ColorChecker CCM ΔE00 mean / max |
|---|---:|---:|---:|---|---|
| D65 | 0.79× | 0.97× | 0.99× | 0.54 → 0.45 | 1.16 / 2.82 → 1.07 / 2.51 |
| A (tungsten) | 0.79× | 0.95× | 0.99× | 1.04 → 0.86 | 0.83 / 2.99 → 0.71 / 2.34 |
| LED_B3 | 0.80× | 0.96× | 0.99× | 0.74 → 0.61 | — |

Almost all of the red loss is the glass's gradual roll-off between 580 and 650 nm. The near-IR leak
above 700 nm is 0.3 % of the red signal under illuminant A and 0.1 % under D65 (the illuminant
tables stop at 780 nm, so lamps' longer NIR tails are not counted here).

### Legacy curve (opt-in)

`spectra/QE/interpolated/QE_IRCF.csv` with the `spectra/QE/interpolated/QE_*.csv` QE is the previous
model: a coating-only edge, 65 % at 650 nm and 0 from 662 nm, with no near-IR leak. Set them back in a
camera model to reproduce older renders (and drop `ircf_coating_csv`).

## Measured cameras: no generic IR-cut filter

The 60 measured camera models (`spectra/QE/cameras/*`, Image Engineering spectral sensitivities,
380–755 nm) are whole-camera responses that already include each camera's own IR-cut filter (their
red response above 700 nm is 0.2–2.7 % of peak, which bare silicon behind a red CFA would not give).
They now set `ircf_csv: null`. Multiplying the generic curve on top removed 9–25 % of their red
signal (median 15 %) and 5 % of green under D65. Their spectral CCM ΔE00 rises slightly (median
0.96 → 1.16) because the double filter trimmed the red tail toward a better colour fit; the new
value is the camera's own colorimetry. Their data stop at 755 nm, so they have no NIR leak beyond it.
