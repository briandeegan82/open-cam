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

## Night and dusk (`--time-of-day`)

Code: `tools/highway_night.py` (hooks in `build_highway_scene.py`); tests:
`tests/test_highway_night.py`. Requires the patched pbrt (`tools/build_pbrt.sh`,
docs/BUILD_PBRT.txt) because of the retroreflective material.

```bash
venv/bin/python tools/build_highway_scene.py --time-of-day night          # moonless, LED 4000 K
venv/bin/python tools/build_highway_scene.py --time-of-day night --moon-phase-deg 0 --streetlights hps
venv/bin/python tools/build_highway_scene.py --time-of-day dusk --twilight-sun-elevation -4
```

| flag | default | effect |
| --- | --- | --- |
| `--time-of-day day\|dusk\|night` | `day` | `day` output is unchanged |
| `--twilight-sun-elevation` | -4 | dusk sun elevation [deg] |
| `--night-sun-elevation` | -30 | night sun elevation (below -18: no twilight) |
| `--moon-phase-deg`, `--moon-elevation`, `--moon-azimuth` | off, 35, -40 | optional moon (phase angle 0 = full) |
| `--headlamps auto\|mixed\|led\|halogen\|off` | auto (mixed when dark) | ego + traffic low beams |
| `--brake-lights none\|lead\|all` | lead | which cars brake |
| `--streetlights auto\|none\|led_4000k\|hps` | auto (LED when dark) | median lamp posts |
| `--retroreflective auto\|on\|off` | auto (on when dark) | marking/sign material |

**Natural light (absolute).** The horizontal illuminance of the sky is log-interpolated in sun
elevation through clear-sky anchors: 400 lx at sunset (-0.833 deg), 3.4 lx at the end of civil
twilight (-6), 0.008 lx nautical (-12), 0.002 lx moonless night (<= -18) (IESNA Lighting
Handbook; Bond & Henderson 1963, US Naval Observatory). Dusk uses an analytic twilight sky map
(blue zenith, warm horizon glow towards the sun); night a uniform sky with an airglow +
starlight spectrum (Leinert et al. 1998). The moon (Krisciunas & Schaefer 1991, eq. 8–9:
full moon ~0.27 lx at the zenith) is a `distant` light with a solar-like spectrum reddened by
the lunar albedo and air mass, plus a moonlit sky with the clear-sky diffuse/direct ratio.

**Vehicle lamps.** Low beams are `goniometric` lights whose equal-area maps hold candelas: an
analytic ECE R112/R149 class-B (right-hand traffic) passing beam with a 15 deg kink cutoff,
checked against the test points B50L, 75R, 75L, 50L, 50R, 50V, 25L, 25R and zone III
(halogen peak ~23 kcd / ~310 lm, LED ~32 kcd / ~430 lm per lamp). Spectra: halogen = 3200 K
Planck (H7); LED = 5700 K phosphor-converted LED (blue InGaN + YAG). 5 % of the flux is moved
into a visible Lambertian lens quad. Tail lamps (6 cd) and brake lamps (86 cd, ECE R7 ranges)
are `diffuse` area lights with a red LED spectrum (626 nm, 17 nm FWHM; inside the ECE red
chromaticity box). Traffic lamp positions come from each car's bounding box.

**Street lights.** 12 m twin-arm poles in the median every 48 m (2.5 m outreach), ANSI/IES
RP-8 / EN 13201 freeway geometry. Each luminaire is a goniometric light with an analytic
full-cutoff IES Type III medium distribution (16 klm LED 4000 K with a red phosphor;
22 klm 250 W HPS with Na D self-reversal, x,y ~ 0.52,0.41, de Groot & van Vliet 1986) plus an
8 % flux lens quad. The design gives E_avg ~14 lx (LED) / ~20 lx (HPS), U0 ~ 0.58 on the
carriageway (manifest `lighting.artificial.streetlights.carriageway_illuminance`).

**Retroreflection.** pbrt-v4 has no retroreflective BSDF, so
`third_party/patches/0001-retroreflective-material.patch` adds `Material "retroreflective"`:
a Lambertian base plus a lobe around the incident direction with
`f = R_A(alpha, b_i, b_o) / (cos b_i cos b_o)` and
`R_A = ra exp(-(alpha - alpha0)/w) (cos b_i cos b_o)^(q/2)` (alpha = observation angle).
The lobe is importance sampled (alpha ~ Gamma(2, w)) and clamped to conserve energy.
- Sign sheeting: ASTM D4956 Type III (EN 12899-1 class RA2), alpha0 = 0.2 deg,
  w = 0.34 deg, q = 2.7, `ra` set to 1.2x the D4956 minimum at (0.2 deg, -4 deg) (white 300,
  green 54 cd lx^-1 m^-2); the model meets the minima at (0.2, -4), (0.2, 30), (0.5, -4),
  (0.5, 30) for white, yellow, green and red.
- Markings: EN 1436 30 m geometry (viewing 2.29 deg, illumination 1.24 deg elevation),
  R_L = 300 (white, class R5) and 200 mcd m^-2 lx^-1 (yellow, R4), alpha0 = 1.05 deg,
  w = 1.6 deg, q = 1.
- `ra` is spectral (colour of the surface, photometric value under CIE illuminant A as in
  ASTM E810). `TestPbrtRetroreflection` renders the material next to a white Lambertian patch
  and recovers R_A and R_L from the image within 4–5 %.

**Radiometry.** At dusk/night `lighting.reference_illuminance_lux` is the *natural* horizontal
illuminance (sky + moon, e.g. 0.002 lx moonless) and `reference_illuminance_exr_lux` the
matching EXR value, so `pbrt_spectral_exr_to_electrons.py` stays absolute. Lamps are absolute
photometric sources (cd, cd/m^2 via pbrt's luminance-normalised spectra), so headlamp- and
streetlight-lit pixels come out in the same units; their budgets are recorded under
`lighting.artificial`. Use long integration times at night.

**Approximations.** The twilight sky map is RGB and the night sky uniform (no stars, no
light-pollution dome); lamp lenses are Lambertian and carry a fixed flux fraction, so their
apparent luminance is lower than real optics (no glare/flare); the asphalt normal map is
dropped when dark (normal mapping has no masking/shadowing and speckles under grazing
headlamp light); sign/marking retroreflection is isotropic in azimuth and has no wet-road
or dew behaviour.

## Known approximations

- Daytime markings use a glass-bead-like rough clear coat over a spectral binder and sign
  sheeting is diffuse; with `--time-of-day dusk|night` (or `--retroreflective on`) both switch
  to the patched pbrt `retroreflective` material (see "Night and dusk" below).
- The sky radiance is RGB, upsampled by pbrt; only the sun is spectral.
- No participating medium (haze), wet road, or road curvature; texture colour maps only modulate
  luminance.
- Asset car materials other than paint/glass/tyres are the authors' RGB values.
