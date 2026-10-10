# Glass dispersion in traced lenses

pbrt lens files (`config/lenses/*.dat`) give one refractive index per glass, its n_d. Stock pbrt-v4
traces every wavelength through those same indices, so realistic-lens renders had no axial or
lateral chromatic aberration. open-cam now adds dispersion. Each glass row of a lens can get a
catalogue glass and a design Abbe number. The index at wavelength λ is then

    n(λ) = n_file + s · (n_g(λ) − n_g(λ_d)),   s = ((n_file − 1) / ν_d) / (n_g(F) − n_g(C))

Here n_g is the glass's Sellmeier curve (SCHOTT catalogue, 2018-06-01). This keeps n(587.6 nm)
equal to n_file, so focus, f-number and EFL don't change. It also gives the design's principal
dispersion, n_F − n_C = (n_file − 1)/ν_d. The catalogue glass sets only the *shape* of n(λ),
including its partial dispersion and NIR tail.

## Files

| File | Role |
|---|---|
| `config/lenses/<lens>.glass.yaml` | sidecar: lens-file row → `{glass, nu_d}` (no sidecar = no dispersion) |
| `tools/lens_glass.py` | Sellmeier table, n(λ), sidecar loader, pbrt `"float dispersion"` values |
| `third_party/patches/0003-realistic-camera-dispersion.patch` | pbrt `RealisticCamera` reads `"float dispersion"` (7 values per interface: s, B1–B3, C1–C3) and traces each camera ray at its hero wavelength; `SpectralFilm` skips terminated secondary wavelengths |
| `tools/lens_diffraction_psf.py` | `LensSystem.from_file(..., wavelength_nm=)` and `LensPsfModel.dispersive_lens`: pupils are traced at each bucket wavelength with the same n(λ) |
| `scripts/validate_lens_dispersion.py` | pbrt vs Python real-ray trace, per spectral bucket |

Sidecars shipped:

- `dgauss.50mm.glass.yaml`: ν_d is the Tronnier patent's (US 2,673,491). Glasses are the nearest
  SCHOTT glasses in (n_d, ν_d): N-BAF10, N-SF15, F5, N-LAK22, N-LAK10.
- `wide_22mm.glass.yaml`: **inferred**. The source gives n_d only, so each row uses the nearest
  SCHOTT glass in n_d, with a flint (F2) for the negative 1.617 elements. ν_d is the catalogue
  value.
- `fisheye.10mm.dat` and `telephoto.250mm.dat` have no sidecar, so they are still dispersion-free.

## Use

- On by default for any realistic lens with a sidecar. All scene builders write the `"float
  dispersion"` line.
- To turn it off, set `lens.dispersion: false` in the camera config, or pass
  `--no-lens-dispersion` to the builder.
- `lens.post_psf.lens_diffraction.dispersion: false` turns it off in the diffraction PSF only.
- The pbrt binary must be rebuilt with the patch: `tools/build_pbrt.sh`. An unpatched pbrt fails
  with an unused-parameter error on `"dispersion"`.

## Validation

`venv/bin/python scripts/validate_lens_dispersion.py` (1024 spp, about 1 min) renders a small
flat-spectrum emitter through the lens, with and without dispersion, into 12 × 25 nm buckets (2 µm
pixels). For each bucket it compares the PSF centroid (lateral colour) and the PSF RMS radius
(axial colour) with a Python real-ray trace at the bucket centre.

| Case | Quantity | pbrt | trace | pbrt, n_d only |
|---|---|---|---|---|
| wide 22 mm, 9 mm field | lateral colour 412.5 nm vs 587.5 nm | 58.4 µm | 61.2 µm | −0.6 µm |
| wide 22 mm, 9 mm field | lateral colour 687.5 nm vs 587.5 nm | −8.6 µm | −8.6 µm | −1.9 µm |
| D-Gauss f/2, on axis | RMS radius 412.5 / 587.5 / 687.5 nm | 37.8 / 13.7 / 22.1 µm | 33.6 / 13.6 / 22.8 µm | 13.3 / 14.2 / 13.0 µm |
| D-Gauss f/5.6, 9 mm field | RMS radius 437.5 / 512.5 / 662.5 nm | 2.7 / 6.0 / 2.0 µm | 2.4 / 6.0 / 2.1 µm | 4.0 / 4.0 / 4.0 µm |

Across the 12 buckets pbrt follows the trace to within about 2 µm (1 pixel) on lateral colour,
and within about 1 µm on RMS radius except in the deep blue. The D-Gauss is nearly symmetric, so
its lateral colour stays below 0.5 µm in both. The n_d-only column is flat, as it should be.

## Caveats

- **Hero wavelength.** A dispersive lens refracts the four wavelengths of a pbrt path
  differently, so the camera keeps only the first (`TerminateSecondary`). This adds about 2×
  colour noise per sample compared with dispersion-free renders. RGB film output stays unbiased.
- **Fixed focus and pupil.** Focus (`TraceLensesFromScene`) and the exit-pupil bounds are
  computed at n_d. The diffraction PSF uses the same focus, so both paths agree.
- **Bucket-centre wavelength.** The Python PSF uses each bucket's centre wavelength. pbrt samples
  wavelengths continuously within each bucket.
- **Ghosts.** `tools/traced_ghosts.py` still uses n_d (see `LENS_GHOSTS.md`).
- **Glass choices are assumptions** wherever the source gives no glass. They are labelled in each
  sidecar.
