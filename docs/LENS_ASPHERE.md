# Aspheric lens surfaces

Traced lenses can now include even-asphere surfaces. pbrt (`third_party/patches/0004-realistic-camera-asphere.patch`)
and the Python tracer in `tools/lens_diffraction_psf.py` (`tools/lens_asphere.py`) use the same surface
geometry, so pbrt renders, diffraction PSFs and traced ghosts all see the same lens.

## Surface model

Each surface has a sag measured from its vertex, with radial height \(r\) in mm and \(c = 1/R\):

\[ z(r) = \frac{c r^2}{1 + \sqrt{1 - (1+k) c^2 r^2}} + \sum_{i=4,6,\dots,16} A_i r^i \]

\(R\) is the curvature radius column of the `.dat` file, which stays unchanged. Paraxial focal length,
focusing and pupil helpers therefore stay correct, because the asphere terms start at \(r^4\). Rays are
intersected by Newton iteration starting from the vertex plane. The surface normal comes from the analytic
slope \(dz/dr\). Where the square root goes negative (when \(k > -1\)), the surface is undefined and the
ray is lost, which acts as vignetting.

## Sidecar

`<lens>.asphere.yaml` sits next to the `.dat` file. Rows are zero-based data rows of the `.dat` file, not
patent surface numbers. Air and stop rows can't be aspheric.

```yaml
rows:
  1: {k: -6.47244e-1, A: {4: 1.94523e-2, 6: -8.79893e-2, 8: 1.97305e-1, 10: -5.99721e-1}}
```

`tools/lens_glass.pbrt_camera_lines` writes `"float asphere"` into the realistic-camera block: 8 values
per lens row (`k, A4..A16`). Lenses without a sidecar render exactly as before.

pbrt change (on top of the dispersion patch 0003): `LensElementInterface` gets `k`/`A` plus the sag and
slope, `IntersectAsphericalElement` dispatches from both `TraceLensesFromFilm` and `TraceLensesFromScene`,
and `RealisticCamera::Create` checks the array length. Phone lenses put the stop in front of the first
vertex with a negative thickness (a virtual stop). pbrt now clips rays at that plane without moving the
ray origin, which matches the Python tracer.

## Phone lens: `phone_6p_2.97mm`

Source: US 8,390,940 B2 (Largan), Embodiment 1, Tables 1-1 and 1-2. It is a six-element plastic lens with
f = 2.97 mm, F/2.45 and HFOV 37.2°, and all 12 surfaces are aspheric.

Taken from the patent: radii, thicknesses, n_d, ν_d, k and A4–A16.

Not in the patent, so inferred:
- Clear apertures come from tracing every ray through the full stop from 0° to 37.2°, plus 5 %.
- Row 9 (k = 15.8) is undefined beyond r = 1.101 mm, so it vignettes the upper rim at full field.
- The IR-cut plate is left out because open-cam applies the IR cut spectrally. Its air-equivalent
  thickness is folded into the back focus, and pbrt refocuses anyway.
- Dispersion: the patent ν_d sets the principal dispersion. The curve shape comes from the nearest SCHOTT
  glass (`.glass.yaml`).

Checks (`tests/test_lens_asphere.py`):

| Quantity | Patent | open-cam |
|---|---|---|
| f (paraxial) | 2.97 mm | 2.966 mm |
| TTL (IR plate air-equivalent) | 3.73 mm | 3.734 mm |
| Distortion, 10°–37° | −2.0 … +1.5 % (Fig. 1D) | inside that band |
| On-axis RMS spot, 588 nm | — | < 2 µm (spherical surfaces only: > 5× larger) |

The test suite also checks:
- A zero-coefficient asphere traces identically to the sphere (wide 22 mm, 200 rays).
- The Newton intersection hits the surface.
- The analytic slope matches the numerical derivative of the sag.

### pbrt against the Python trace

`venv/bin/python scripts/validate_lens_asphere.py` takes about 1 min. It uses a 0.25 µm film pitch and a
full stop (1.212 mm), with the emitter at 3.2 m.

| Field | RMS radius, 588 nm (pbrt / trace) | Lateral colour 412 nm vs 588 nm (pbrt / trace) |
|---|---|---|
| On axis | 0.3–0.5 / 0.50 µm | 0 / 0 µm |
| 1.5 mm (0.67 field) | 3.3–3.9 / 4.04 µm | 3.8 / 2.9 µm |

The image height at 1.5 mm agrees to 2 µm. With the aspheres removed, pbrt passes no light at that field
and the Python trace passes 6 %, so the aspheric terms really are being traced. Per-bucket pbrt values
vary by about ±0.5 µm because of sample noise.

## Limits

- Even aspheres only: no odd terms, Zernike or freeform surfaces, and no decentre or tilt.
- Clear apertures are traced estimates, not the patent's mechanical apertures.
- The `iphone_8` recipe still uses its spherical stand-in. To use this lens, point a recipe's `lensfile`
  at `config/lenses/phone_6p_2.97mm.dat`. That changes its scorecard numbers.
