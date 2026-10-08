# Highway scenes (automotive / ADAS)

`tools/build_highway_scene.py` writes a spectral pbrt-v4 scene of a divided highway seen from
windscreen height (1.35 m, middle lane, looking down the road):

- 3 lanes per direction (3.66 m), US/MUTCD markings: yellow left edge, broken white lane lines
  (3.05 m dash / 9.14 m gap), solid white right edge; shoulders; concrete median barrier;
  W-beam guard rails on posts; a guide sign and a speed-limit sign.
- Traffic: CC0 car models that ship as pbrt-v4 scenes (BMW M6, Pontiac GTO 67, a vintage car).
  The original glass/chrome/rubber/light materials are kept; body paint becomes a spectral
  `coateddiffuse` clearcoat, glass a `dielectric` (eta 1.52), tyres spectral rubber.
- Ambient CG asphalt / grass colour maps (luminance only) modulate measured-shape analytic
  spectral reflectances (`tools/highway_spectra.py`); normal maps add surface relief.
- Rolling grass terrain with instanced Poly Haven trees and shrubs (leaves use
  `diffusetransmission` with spectral leaf reflectance/transmittance).
- Sun + sky with absolute photometry (below).

## Assets

Nothing third-party is committed. `config/highway_assets.yaml` lists every asset with its
source, author, licence (all CC0) and a pinned sha256 (or pinned git commit). Fetch and
prepare them (about 0.55 GB into the gitignored `scenes/assets/highway/`):

```bash
venv/bin/python tools/fetch_highway_assets.py --list
venv/bin/python tools/fetch_highway_assets.py          # all assets, or pass asset ids
```

Preparation removes the sun from each HDRI (measuring its direction and its illuminance
relative to the sky), converts the sky to pbrt's equal-area format, converts the glTF
vegetation to PLY, and indexes the car scenes' materials and meshes. The Bitterli car zips
also contain an `envmap.pfm` that is *not* CC0; it is never used.

## Build, render, convert

```bash
venv/bin/python tools/build_highway_scene.py                       # pinhole, clear day
venv/bin/python tools/build_highway_scene.py --camera realistic \
    --lensfile config/lenses/wide_22mm.dat --aperture-diameter-mm 4 --focus-distance 25
venv/bin/python tools/build_highway_scene.py --sky syferfontein_6d_clear --sun-azimuth 25   # low sun
third_party/pbrt-v4/build/pbrt scenes/generated/highway/highway.pbrt
venv/bin/python tools/pbrt_spectral_exr_to_electrons.py --exr out/highway_spectral.exr \
    --camera-model-config config/camera_recipes/research_realistic_wide22.yaml \
    --scene-manifest-json scenes/generated/highway/highway_manifest.json --integration-time-s 0.0002
```

Useful options: `--sky {kloofendal_43d_clear,kloofendal_48d_partly_cloudy,syferfontein_6d_clear,analytic}`,
`--sun-elevation` (analytic sky), `--sun-azimuth` (degrees from the road direction, + = right),
`--global-illuminance-lux`, `--road {asphalt031,asphalt026c,plain}`, `--vegetation`, `--traffic none`,
`--proxy-cars`, `--cam-height`, `--cam-lane`, `--cam-pitch`, and the usual film/camera options.
`--allow-missing-assets` builds without any downloads (proxy cars, analytic sky, untextured
surfaces); the tests use it.

## Radiometry

- Sun: `distant` light with a spectral direct-normal SPD (5778 K envelope x Rayleigh, aerosol,
  ozone and O2/H2O transmission along the Kasten-Young air mass), `illuminance` =
  128 klux x exp(-0.21 m) (clear sky).
- Sky: `infinite` light from the sun-free HDRI (or a CIE type-12 clear sky), `illuminance`
  set so that sky/sun horizontal illuminance matches the HDRI's measured ratio (IESNA clear-sky
  diffuse fit for the analytic sky). `--global-illuminance-lux` rescales both.
- The manifest records `lighting.reference_illuminance_lux` (sun + sky on a horizontal plane at
  the road) and `reference_illuminance_exr_lux`. `pbrt_spectral_exr_to_electrons.py` uses them
  like a chart illuminance, so electrons are absolute and the recipe's
  `target_illuminance_lux` is ignored (pass `--target-illuminance-lux` to override).
  Realistic-camera EXRs are film irradiance, pinhole/thin-lens EXRs radiance, as for the
  other builders. Daylight is bright: pick a short `--integration-time-s`.

## Spectral sky

pbrt-v4's image `infinite` light only takes RGB maps, which pbrt turns into
`RGBIlluminantSpectrum`s (smooth sigmoid x colour-space illuminant), so with the default sky
every sky pixel has a D65-shaped, smooth spectrum regardless of the real sky. Two opt-in
modes make the sky spectral, so different CFAs (RGGB, RCCB, RYYCy, ...) and sensor QE curves
see daylight spectra:

```bash
# Hosek-Wilkie spectral clear sky (no HDRI needed); turbidity 1-10, default 3
venv/bin/python tools/build_highway_scene.py --sky hosek --sun-elevation 43 --turbidity 3
# keep the Poly Haven HDRI (or --sky analytic) but render it with daylight spectra
venv/bin/python tools/build_highway_scene.py --sky kloofendal_43d_clear --sky-spectrum daylight
```

Both need pbrt built with `tools/build_pbrt.sh`, which applies
`third_party/patches/pbrt-v4-spectral-basis-infinite-light.patch` (see `docs/BUILD_PBRT.txt`).
The patch lets an `infinite` light take a `filename` *and* K `"spectrum L"` SPDs; the
equal-area EXR then has channels `B0..B{K-1}` and
`Le(w, lambda) = scale * sum_k B_k(w) L_k(lambda) / Y(L_k)`, sampled by luminance as before;
`"float illuminance"` keeps setting the horizontal illuminance. RGB maps are unchanged, so the
default scene renders identically.

**Basis.** K = 3 spectra from the CIE daylight model S0 + M1 S1 + M2 S2 (Judd, MacAdam &
Wyszecki 1964, JOSA 54:1031; CIE 015:2018 sec. 4.1.2), whose components were derived from
measured daylight and skylight spectra. The CIE table (`spectra/illuminant/original/
CIE_illum_Dxx_comp.csv`, doi:10.25039/CIE.DS.w7zunnny, CC BY-SA 4.0) is in the repo. The
three basis SPDs are vertices of the region where the daylight model is non-negative over
360-830 nm (so each basis SPD is physical); their triangle contains the daylight locus from
~3300 K to infinity. Each pixel is given the daylight-model spectrum with the **same CIE XYZ**
(exact for colours inside the triangle; outside it the negative weight is clamped and the
luminance kept; the manifest reports `out_of_gamut_luminance_fraction`).

**Sources.**
- `--sky hosek`: Hosek & Wilkie (2012), "An Analytic Model for Full Spectral Sky-Dome
  Radiance", ACM TOG 31(4):95. Their BSD-3 reference code/data ship with pbrt-v4
  (`src/ext/skymodel`); the coefficient tables are copied to
  `spectra/sky/hosek_wilkie_2012_spectral.npz` (licence in `spectra/sky/README.md`) and
  evaluated by a NumPy port (`HosekWilkieSky`, matches the C code to 1e-15). Ground albedo
  0.1. Spectra are 320-720 nm; the daylight-basis metamer reproduces them within ~4-10 %
  (luminance-weighted max deviation, 400-720 nm, sun 20-70 deg) and extends them to 830 nm.
- `--sky-spectrum daylight`: the RGB map (HDRI with the sun removed, or the CIE type-12
  map) read as linear sRGB. For `kloofendal_43d_clear` all sky pixels are in gamut;
  `syferfontein_6d_clear` (sun at 6 deg, orange horizon) has ~30 % of its luminance clamped.

**Radiometry is unchanged.** The sun (spectral distant light) and the absolute horizontal
sky illuminance are exactly those of the RGB path: HDRI ratio for HDRIs, the IESNA
clear-sky fit for `hosek`/`analytic`. Hosek's own absolute diffuse illuminance is recorded as
`lighting.sky.spectral.model_illuminance_horizontal_lux` for comparison.

**Validation** (`tests/test_highway_spectral_sky.py`, turbidity 3):

| sun elevation | Hosek E_diffuse / IESNA fit | sky CCT (horizontal) | sun + sky CCT |
|---|---|---|---|
| 10 deg | 1.24 | 8.8 kK | 5.3 kK |
| 30 deg | 1.21 | 12.5 kK | 5.5 kK |
| 45 deg | 1.13 | 15.5 kK | 5.6 kK |
| 60 deg | 1.07 | 19 kK | 5.7 kK |

- Luminance: Hosek's absolute diffuse horizontal illuminance is within 30 % of the IESNA
  clear-sky fit used by the builder, and its relative luminance distribution correlates with
  the CIE S 011 / ISO 15469 standard clear sky type 12 (log-luminance r > 0.85 at 20-45 deg,
  outside 10 deg of the sun).
- Colour: horizontal skylight is 8-25 kK, slightly above the Planckian locus like the CIE
  daylight locus (Duv ~ +0.006 vs +0.003), with a bluer zenith; sun + sky on a horizontal
  plane is 5.0-6.6 kK, i.e. typical daylight between D50 and D65 (CIE 015:2018). The
  `kloofendal_43d_clear` HDRI converted with `--sky-spectrum daylight` gives 11.3 kK
  horizontal / 17 kK zenith, consistent with Hosek at the same sun elevation (14.6 kK).
- The manifest records `horizontal_cct_k`, `zenith_cct_k` and Duv for every spectral sky.

Limitations: the basis has three degrees of freedom, so spectral detail beyond the daylight
model (e.g. the ozone Chappuis band shape, O2/H2O absorption lines) is smoothed; the Hosek
model has no spectral data above 720 nm (the basis extrapolates as daylight); turbidity is
not tied to the sun model's Angstrom aerosol coefficient.

## Known approximations

- No retroreflective BSDF in pbrt: markings use a glass-bead-like rough clear coat over a
  spectral binder; sign sheeting is diffuse. Night/headlight retroreflection is not modelled.
- By default the sky radiance is RGB, upsampled by pbrt; only the sun is spectral (see
  [Spectral sky](#spectral-sky) for the opt-in spectral sky).
- No participating medium (haze), wet road, or road curvature; texture colour maps only modulate
  luminance.
- Asset car materials other than paint/glass/tyres are the authors' RGB values.
