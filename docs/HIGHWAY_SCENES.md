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
source, author, licence (CC0, except the heavy vehicles: CC-BY-4.0 with attribution) and a pinned sha256 (or pinned git commit). Fetch and
prepare them (about 1 GB into the gitignored `scenes/assets/highway/`):

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

## Road surface wear

`--road-wear none|light|moderate|heavy` (default `moderate`) ages the road with seeded,
reproducible distress (`--road-wear-seed`, default `--seed`). The code lives in
`tools/highway_road_wear.py`. The default `asphalt` material is replaced, and so are the clean
`paint_*` quads. Only material fields change, so the manifest's absolute illuminance is not
affected. The resulting area-weighted luminous reflectance is recorded in
`road.wear.luminous_reflectance`. For the default level it is about 0.12 (p5-p95 0.11-0.13),
which sits inside the measured aged-asphalt range: new asphalt is about 0.05 and aged asphalt
0.10-0.18 (Pomerantz, Akbari et al., LBNL cool-pavement reports; Herold et al. 2004).

- **Anti-tiling**: the `--road` ambientCG asphalt and Asphalt033 are each re-synthesised into a
  6 m periodic tile. The method is variance-preserving stochastic tiling: random offset/rotated
  copies blended on a cos² partition of unity (Heitz & Neyret 2018, without the histogram
  transform). This covers luminance, roughness and normals, with normals rotated alongside the
  image. The two asphalts are mixed with a 2.5 m noise mask.
- **Wear maps** (one 97.5 m period, 2 cm texels, mapped through the road uv so they follow `--curve-radius`/`--grade`; `--road-wear-texel-m`):
  - Tyre wheel paths at ±0.88 m from each lane centre, with wander σ 0.32 m (MEPDG default
    10 in wander SD convolved with the tyre width). They are weighted towards the slow lanes,
    about 20% darker, and polished (roughness 0.35 → 0.12).
  - Oil/drip stains along lane centres.
  - Transverse and longitudinal (joint/wheel-path) cracks: unsealed 6-15 mm, or sealed with
    60-120 mm rubberised sealant overbands.
  - Pothole, wheel-path and full-lane patches with younger binder (~0.07) and sealed edges.
  - Distress names and severities follow the LTPP Distress Identification Manual
    (FHWA-RD-03-031).
- **Markings**: the spectral mix of traffic paint and asphalt is area-weighted by a seeded
  paint-loss map that models flaking, edge chipping and dash-to-dash variation. Intact paint
  keeps a smoother coat (roughness 0.15, giving a glass-bead sheen in daylight). Worn areas take
  the asphalt roughness.
  With `--retroreflective on` (default at dusk/night) the worn paint is a pbrt `mix` of the
  retroreflective `paint_*` material and bare asphalt, driven by the same paint-loss map, so
  R_L scales with the remaining paint. At dusk/night the normal maps of the wear asphalts are
  dropped as well (see Night and dusk).
- **Raised pavement markers**: 100 x 100 x 18 mm geometry instances (ASTM D4280 height limit
  ~20 mm). They sit in every other broken-line gap (2N = 24.4 m, MUTCD 3B.11-3B.14), with
  clear/red lenses on lane lines and amber on the yellow edge, and about 10% are missing. Lenses
  are smooth coated diffuse (daytime look only; their retroreflection is not modelled, even at night).
- **Normal maps**: pbrt-v4 decodes 8-bit JPEGs as sRGB even when a linear encoding is requested,
  which tilted the asphalt normals and rendered the road ~7x too dark. The asphalt and grass
  normal maps (with or without wear) are therefore converted to PNG
  (`highway_road_wear.linear_normal_map`).
- **Solid lines** are split into one-dash-cycle (12.19 m) segments in every scene, with or
  without wear: pbrt renders the 1.5 km x 0.2 m sliver triangles of an unsplit edge line
  far too dark.

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
~3800 K to infinity. Each pixel is given the daylight-model spectrum with the **same CIE XYZ**
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
| 10 deg | 1.24 | 9.3 kK | 5.4 kK |
| 30 deg | 1.21 | 12.9 kK | 5.5 kK |
| 45 deg | 1.13 | 15.5 kK | 5.6 kK |
| 60 deg | 1.07 | 17.7 kK | 5.7 kK |
| 70 deg | 1.05 | 17.5 kK | 5.7 kK |

- Luminance: Hosek's absolute diffuse horizontal illuminance is within 30 % of the IESNA
  clear-sky fit used by the builder, and its relative luminance distribution correlates with
  the CIE S 011 / ISO 15469 standard clear sky type 12 (log-luminance r > 0.85 at 20-45 deg,
  outside 10 deg of the sun).
- Colour: horizontal skylight is 9-18 kK, slightly above the Planckian locus like the CIE
  daylight locus (Duv +0.005 to +0.008 vs +0.003), with a bluer zenith for sun elevations up
  to ~60 deg; sun + sky on a horizontal plane is 5.4-5.7 kK, i.e. typical daylight between
  D50 and D65 (CIE 015:2018). The
  `kloofendal_43d_clear` HDRI converted with `--sky-spectrum daylight` gives 11.3 kK
  horizontal / 17 kK zenith, consistent with Hosek at the same sun elevation (14.6 kK).
- The manifest records `horizontal_cct_k`, `zenith_cct_k` and Duv for every spectral sky.

Limitations: the basis has three degrees of freedom, so spectral detail beyond the daylight
model (e.g. the ozone Chappuis band shape, O2/H2O absorption lines) is smoothed; the Hosek
model has no spectral data above 720 nm (the basis extrapolates as daylight); turbidity is
not tied to the sun model's Angstrom aerosol coefficient.

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
When `--lamp-posts on` (scene variety) already places median posts, the night luminaires
are mounted under those heads (`pole_source` in the manifest) instead of adding a second
set of poles, and the illuminance diagnostic uses their spacing/height. All lamps (street
and vehicle) follow the road alignment via `highway_variety.place`.

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
headlamp light; re-checked after the PNG normal-map fix: keeping it still raises the road's
pixel-to-pixel deviation by ~50 % at 512 spp and darkens the headlamp-lit road by ~8 %); sign/marking retroreflection is isotropic in azimuth and has no dew behaviour (wet markings:
see [Wet road](#wet-road-puddles-and-vehicle-spray-tools-highway_wetpy)).
`--haze` works at dusk/night too: the natural sources (twilight/night sky, moon, moonlit sky)
are put at the top of the medium and the manifest's reference illuminance is the MC road
illuminance under the medium (see "Haze" below); vehicle lamps and luminaires sit inside the
medium and pbrt's `volpath` attenuates and scatters them (glow/veiling around lamps).

## Wet road, puddles and vehicle spray (`tools/highway_wet.py`)

Opt-in; the defaults (`--road-wetness dry`, no `--puddles`, no `--spray`) write a byte-identical
scene (`TestBuilderWet.test_dry_default_unchanged`).

```bash
venv/bin/python tools/build_highway_scene.py --road-wetness wet --puddles --spray               # day
venv/bin/python tools/build_highway_scene.py --time-of-day night --road-wetness wet --puddles \
    --spray --haze mist                                                                       # night
venv/bin/python tools/build_highway_scene.py --time-of-day night --road-wetness flooded --wet-marking-rl 50
```

**Wet asphalt** (`damp|wet|flooded`). Every road material (asphalt / wear tiles / sealant /
markings) becomes a pbrt `coateddiffuse` with a water interface (`eta` 1.333) over the dry
spectral body:

- *Darkening* — Lekner & Dorf, "Why some things are darker when wet", Appl. Opt. 27, 1278
  (1988), doi:10.1364/AO.27.001278: light refracted into the water film is diffusely reflected by
  the porous body (albedo a) and partly trapped by total internal reflection; the wet body
  albedo is a_w = (1−r_e)(1−r_i) a / (1−r_i a) with the hemispherical Fresnel reflectances of
  water r_e = 0.066 (outside) and r_i = 1 − (1−r_e)/n² = 0.475 (inside). Dark asphalt (a ≈ 0.1)
  falls to ≈ 0.52 a (the paper's "about half"). The coating in pbrt already applies the
  (1−r_e)/(1−r_i a) part with water's n, so the builder writes `spd/wet_<name>.spd` with the dry
  spectrum transformed per wavelength (dry clear coat, `eta` 1.5, removed by inversion) —
  spectral, not RGB. The road texture maps modulate this spectral scale linearly; that is
  exact at the texel mean and within ±8 % for texels at ×0.4/×1.6 of it.
- *Specular water surface* — microfacet roughness by level (damp 0.15, wet 0.05, flooded 0.02)
  chosen so the luminance coefficient q integrated to the CIE average Q0 (CIE 47-1979 "Road
  lighting for wet conditions"; R/W tables as in CIE 144:2001) lands in the CIE wet classes;
  the asphalt normal map is dropped for wet/flooded (water fills the texture).

| level | roughness | model Q0 | CIE reference Q0 |
|---|---|---|---|
| dry (existing material) | – | 0.058 | R2/R3 0.07, R1 0.10 |
| damp | 0.15 | 0.074 | (< W1 0.11, partly dry) |
| wet | 0.05 | 0.142 | W2 0.15 (W1 0.11–W4 0.25) |
| flooded | 0.02 | 0.245 | W4 0.25 |

Q0 here is the CIE average over the road-lighting solid angle (β 0–180°, tan γ ≤ 12) at an
observation angle of 1°, computed analytically (`highway_wet.q0`, Lekner–Dorf body + GGX
water specular); `TestPbrtWetCoating` renders a flat plate (normal incidence, 45° view)
with pbrt's stochastic `coateddiffuse` against a Lambertian plate of the same albedo: with a
smooth water interface pbrt reproduces the Lekner–Dorf body BRDF to 0.1 % (a = 0.1: 0.5624 vs
0.5626; a = 0.3: 0.6244 vs 0.6249); with the wet roughness 0.05 it is 0.9 % / 3.7 % below the
analytic body + GGX model used for Q0 (0.6576 vs 0.6636; 0.6345 vs 0.6586 — pbrt's rough
interface also changes the light trapped in the film). The wet coatings use `"float thickness"
1e-4`: pbrt's `LayeredBxDF` attenuates by exp(−thickness/|cos θ|) even with zero medium albedo,
and the default 0.01 darkened the body by up to 6 %.

**Puddles** (`--puddles`, wet/flooded, needs `--road-wear` ≠ none). Water collects where the
rut-depth proxy — the wheel-path map of `tools/highway_road_wear.py`, normalised and modulated
by a 6 m longitudinal unevenness — exceeds a level (wet 0.8, flooded 0.6): ≈1 % (wet) and a few
% (flooded) of the road area, concentrated in the wheel paths. Puddles are a smooth water
surface (roughness 0) over the wet body (pbrt `mix` with the puddle mask); the road geometry
stays flat (no depressed water surface, no waves/rain ripples).

**Wet markings** (night / `--retroreflective on`). The retroreflective paint's R_L is replaced
by EN 1436:2018 wet classes: wet → RW2 35 mcd·m⁻²·lx⁻¹ (white) / RW1 25 (yellow), the wet-recovery
condition measured by ASTM E2177; flooded → RR1 25 (continuous wetting, ASTM E2176);
`--wet-marking-rl` overrides. Dry 300/200 → wet 35/25 is ≈ 12 % of dry, consistent with FHWA
(FHWA-HRT-15-062 and the wet-retroreflectivity studies cited there) reporting wet R_L of
standard paint/beads at a small fraction of dry. Sign sheeting keeps its dry R_A.

**Spray** (`--spray`, wet/flooded). A spray medium behind every vehicle (width + 0.6 m,
height ≤ 2.5 m, length 10 m + speed/6 trimmed before the following car), purely scattering,
Henyey–Greenstein g = 0.85 (drops 0.1–0.4 mm ≫ λ: forward-peaked, near-unit albedo; cf. Hansen
& Travis 1974). Peak extinction 0.2 m⁻¹ for a heavy vehicle at 90 km/h on a wet road — the
maximum measured by Otxoterena Drake et al., J. Wind Eng. Ind. Aerodyn. 217, 104734 (2021),
doi:10.1016/j.jweia.2021.104734 — scaled linearly with speed above 30 km/h, ×0.5 for cars and ×2
for flooded (assumptions, stated in the manifest). Density decays exponentially downstream and
with height; it is rendered as 4 homogeneous slabs along the plume, each with the mean extinction
of its cross-section (a `uniformgrid` medium casts spurious black sun "shadows" in pbrt-v4 here, so
the lateral wheel-track and vertical structure is averaged out). Static scenes (`--traffic-speed-kmh 0`) use the nominal lane speeds
(125/110/95, oncoming 105 km/h) for spray. Spray forces `volpath`; with `--haze` the spray
boxes are nested inside the haze medium. The (invisible) medium box extends 0.5 m below the road
so the road plane lies inside it (a box face a few cm above the 2-triangle road plane is crossed
inconsistently by pbrt's ray offsets and blacks out the road), and plumes are clipped 0.5 m ahead
of the camera, which is never placed inside a spray medium.

The manifest records everything under `road.wet` (level, Q0 model vs CIE, darkening ratio,
puddle area fraction, marking classes, per-vehicle spray extinction).

## Haze, fog and distant terrain

`--haze clear|hazy|mist|fog` (default `none`) puts the camera, every surface and the light paths
inside a pbrt-v4 participating medium (`tools/highway_atmosphere.py`) and switches the integrator
to `volpath`. `--distant-terrain auto|none|hills` adds a ring of hills from 2.5 to 30 km
(`tools/highway_backdrop.py`, on by default whenever `--haze` is set) so that aerial perspective
has something to act on; with `--haze none` the default scene is unchanged.

```sh
venv/bin/python tools/build_highway_scene.py --haze fog                      # preset
venv/bin/python tools/build_highway_scene.py --haze fog --visibility-m 80     # ADAS sweep
venv/bin/python tools/build_highway_scene.py --haze hazy --haze-angstrom 1.3 --haze-g 0.7
```

**Optical model.** The medium is parameterised by the meteorological visibility V at road level.
Koschmieder's law with the WMO 2 % contrast threshold gives the total extinction at 550 nm,
sigma_ext = -ln(0.02)/V = 3.912/V (Koschmieder 1924; WMO-No. 8, 2018). It is split into
Rayleigh scattering, sigma_R = 1.16e-5 /m x (lambda/550)^-4.08 (sea-level air; Bucholtz 1995), and
aerosol, sigma_aer(lambda) = (3.912/V - sigma_R(550)) x (lambda/550)^-alpha with the Ångström
exponent alpha (Ångström 1929): ~1.3 continental haze, ~0.5 mist, 0 for fog whose droplets are
much larger than the wavelength (grey fog). The aerosol single-scattering albedo omega sets
sigma_a/sigma_s. The phase function is Henyey-Greenstein; pbrt takes one g per medium, so g is
the scattering-weighted mean of the aerosol g and Rayleigh (g = 0), recorded as `hg_g_effective`.
Haze aerosol g ~ 0.7 (Shettle & Fenn 1979), fog droplets ~0.85 (Mie, Deirmendjian 1969).

| preset | V | alpha | g | omega | vertical profile | tau_550 (vertical) | volpath maxdepth |
|---|---|---|---|---|---|---|---|
| `clear` | 30 km | 1.3 | 0.70 | 0.95 | exponential, H = 1.2 km | 0.16 | 8 |
| `hazy` | 8 km | 1.0 | 0.70 | 0.90 | exponential, H = 1.0 km | 0.49 | 8 |
| `mist` | 2 km | 0.5 | 0.75 | 0.98 | 300 m layer | 0.68 | 24 |
| `fog` | 150 m | 0 | 0.85 | 0.999 | 150 m radiation-fog layer | 4.5 | 64 |

All parameters can be overridden (`--visibility-m`, `--haze-angstrom`, `--haze-g`,
`--haze-albedo`, `--haze-profile exponential|layer`, `--haze-height-m`, `--haze-maxdepth`). The
medium is a `uniformgrid` that is horizontally uniform over +-200 km and varies only with height:
density 1 at the road (so V holds at the camera) and either exp(-y/H) or a layer of depth H with
a smooth top. Each medium interaction counts as a bounce in `volpath`, so `maxdepth` is raised
with optical depth to resolve multiple scattering (heavy fog is mostly diffuse light).

**Radiometry with a medium.** The sun and sky lights keep their clear-sky values (the sun SPD
already includes the transmission of the clear atmosphere above, the HDRI sky its radiance)
and are taken to be at the *top* of the haze layer: pbrt's `distant`/`infinite` lights shine from
outside the medium, so their light is attenuated and scattered on the way down exactly as in
reality, and the road under fog is darker and diffuse-lit. Because the converter needs the
illuminance that actually reaches the road, the builder computes it with a plane-parallel Monte
Carlo model of the *same* medium (same sigma(lambda), g, omega and profile, Lambertian ground
of albedo `--haze-ground-albedo` 0.15 for ground-sky multiple reflection; photometrically weighted
over 400-700 nm; sun at its elevation, sky with the sky map's angular distribution) and writes
it to `lighting.reference_illuminance_lux` (and `_exr_lux` with the same 683 x Y-integral scale as
without medium). Everything about the calculation is in the manifest's `atmosphere` block
(`road_illuminance`: top-of-layer sun/sky, direct and total transmittances, diffuse fraction,
and `no_medium_horizontal_illuminance_lux`). `TestPbrtHighwayHaze` checks it against pbrt: a Lambertian ground probe (±5 km quad, camera 0.2 m) rendered with `volpath` under each preset and under no medium gives the same illuminance ratio as the manifest within 5 % (measured: clear 0.955 vs 0.959, hazy 0.841 vs 0.854, mist 0.883 vs 0.888, fog 0.680 vs 0.688). Keep such probes ≲10 km across: on a 300 km float32 quad the hit-point error biases the shadow rays by tens of percent. With `--haze-light-reference road` the lights
are instead rescaled so the road receives the no-medium illuminance (useful to isolate the
contrast loss of fog from the change in exposure); the manifest then records the scale applied.

**Dusk and night.** With `--time-of-day dusk|night` the same MC model runs over the natural
sources of `highway_night.natural_sources` (`atmosphere.road_illuminance_sources`): twilight sky
(its RGB map's angular distribution), uniform night sky (cosine-weighted), moon (direct beam at
its elevation, moon SPD) and moonlit sky, each V(λ)×SPD weighted. Their sum under the medium is
the manifest's reference/natural illuminance; `atmosphere.road_illuminance.sources` lists
top-of-layer illuminance and transmittances per source, and `--haze-light-reference road`
scales all natural sources (lamps are never rescaled). Headlamps, tail lamps and luminaires are
inside the medium — pbrt attenuates and scatters them, which gives the glow around lamps; they
are reported separately under `lighting.artificial` as before. `TestPbrtHighwayHaze` checks the
dusk case against pbrt with the same ground probe.

**Distant terrain.** The backdrop is a polar heightfield centred on the ego position: ridges
specified by the elevation angle they subtend (peaks ~0.8 deg at 3 km to ~2.2 deg at 30 km, a
shallow valley along the road), minus the Earth-curvature drop r^2/(2 R_eff), R_eff = 7/6 R_earth
(standard refraction). Cover is an `fbm` mix of the scene's grass and a closed forest canopy
reflectance (visible 0.02-0.06, red edge to ~0.3; Moody et al. 2005, ECOSTRESS spectral library).
It is procedural, so no new assets are fetched; `--seed` changes the skyline.

**Render cost** (1280x720, 256 spp, 8 threads, same view; see the PR for images):

| scene | integrator / maxdepth | wall time | × default |
|---|---|---:|---:|
| default (no haze) | path / 8 | 148 s | 1.00 |
| clear | volpath / 8 | 281 s | 1.89 |
| hazy | volpath / 8 | 440 s | 2.96 |
| mist | volpath / 24 | 362 s | 2.44 |
| fog | volpath / 64 | 683 s | 4.61 |

1280×720, 256 spp, same camera view, 8-thread VM, pbrt-v4 CPU build. Haze presets also switch on the distant hills (64×720-vertex ring, ~92 k triangles); the time is the total for medium + hills.

## Scene variety (`--seed`)

`tools/highway_variety.py` turns the fixed scene into a reproducible family of scenes for
datasets. Without `--seed` the scene is the reference one (fixed traffic, straight level road,
vegetation placed with seed 7); the only change is the textured barrier materials below
(`--barrier-materials flat` restores the old ones). `--seed N` draws, from one generator in a
fixed order, so the same N always gives the same `highway.pbrt` and manifest:

- **Traffic** in all six lanes (lanes < 0 are the oncoming carriageway): per-lane density
  4-22 veh/km (HCM LOS A-C), exponential headways with a 10 m minimum gap, ego-lane traffic
  starting 14 m ahead, up to 320 m. Heavy-vehicle share 5-25 % (HCM default 10-25 %), weighted
  to the right-hand lanes: articulated trucks (DAF CF tractor + box trailer), rigid box trucks,
  vans (also 8 % of light vehicles) and buses; cars are the three CC0 car models.
- **Paint**: car basecoats drawn from the DuPont 2012 Global Automotive Color Popularity Report
  world shares (white 23 %, black 21 %, silver 18 %, grey 14 %, red 8 %, blue 6 %, beige/brown
  6 %, green 1 %, yellow/orange 3 %); vans and box trucks use a white-dominated fleet mix. Each
  vehicle gets its own spectral SPD (`spd/carpaint_<colour>_<nn>.spd`, lightness x0.9-1.08)
  under the existing clear coat. Bus and tractor liveries are the authors' textures.
- **Alignment**: tangent -> clothoid -> circular arc -> clothoid -> tangent (radius 700-2500 m
  either way, spiral 60-180 m or none, total turn 30 deg; AASHTO Green Book minimum radius
  ~600-700 m at 110-120 km/h) and a grade of up to +/-4 % between two 250 m parabolic vertical
  curves, back to level after ~1.3 km. Geometry is built in a straight frame and every mesh
  (refined along the road so chords stay within millimetres) is bent onto the alignment; terrain
  on the inside of the bend is compressed laterally so it never folds. Instances (vehicles,
  trees) are placed with the local heading and pitch. The camera stays at chainage 0, before
  the curve/grade starts.
- **Structures**: 12 m twin-arm lamp columns on the median every 36-50 m (BS 5489-1 /
  EN 13201 spacing ~3.5-4x mounting height; head positions are recorded in the manifest
  `variety.lamp_posts.heads` for night lighting), an overhead sign gantry (5.6 m clearance,
  three seeded guide-sign legends) and an overpass (5.3 m headroom, median pier, abutments).
- **Vegetation**: 1-3 tree species (Poly Haven island tree, Searsia lucida, fir and pine
  saplings) and 1-3 shrub species, scaled to plausible heights, with the per-species spacing
  stretched so the total density is unchanged.

Individual flags override the seeded draw: `--curve-radius` (m, + bends right, 0 straight),
`--clothoid-length`, `--grade` (%), `--lamp-posts/--gantry/--overpass on|off`. Every draw is
recorded in the manifest under `variety`. Lighting is untouched: the sun/sky `illuminance`
values and `reference_illuminance_lux` are identical for every seed (the reference is the
unoccluded horizontal illuminance, so shadows from new structures do not change it).

**Barrier materials**: the median barrier is weathered grey concrete (spectral reflectance
0.22-0.28, Levinson & Akbari 2002 weathered concrete 0.2-0.3) modulated by the ambientCG
Concrete031 luminance map; W-beam rails, posts, lamp columns and the gantry are weathered
hot-dip galvanised steel, a pbrt `mix` of a rough zinc conductor (n, k from Werner et al.,
J. Phys. Chem. Ref. Data 38, 1013 (2009)) and a diffuse zinc-carbonate patina (0.27-0.30),
with the patina fraction (mean ~0.7) taken from the ambientCG Metal032 roughness map. Both use
world-space planar texture mapping, so they need no UVs and survive the alignment warp.

**Heavy vehicles** (CC-BY-4.0, from Sketchfab via the public Objaverse mirror, so no login is
needed; attribution in `config/highway_assets.yaml`): "Truck DAF CF 75.310" and "Mercedes-Benz
Sprinter 2006" by Max-5532, a box trailer by cmitche1, a box truck by roy.gearloft.in and a town
bus by own.guest. The fetcher unpacks the `.glb` files (`kind: glb`), drops vertices outside
the glTF accessor bounds (some exports contain corrupt ones), and the builder scales each model
to its real length (robust bounding box) and maps materials by name: paint -> spectral
basecoat, glass -> thin dielectric, tyres -> rubber, others -> the authors' textures/colours.

## Known approximations

- Daytime markings use a glass-bead-like rough clear coat over a spectral binder and sign
  sheeting is diffuse; with `--time-of-day dusk|night` (or `--retroreflective on`) both switch
  to the patched pbrt `retroreflective` material (see "Night and dusk" below).
- By default the sky radiance is RGB, upsampled by pbrt; only the sun is spectral (see
  [Spectral sky](#spectral-sky) for the opt-in spectral sky).
- Texture colour maps only modulate luminance. Wet road: flat puddles (no depression, no
  ripples/raindrop impacts on the water), no rain streaks, no tyre-track drying, spray density
  scaling (car factor, flooded ×2, linear speed) assumed; wet markings use EN 1436 class
  minima, not a measured wet retroreflection function.
- Haze: one HG phase function per medium (Rayleigh folded into g), horizontally uniform
  medium, flat Earth for the medium (the backdrop has curvature); without `--haze` the horizon
  is sharp and aerial perspective comes only from the sky map.
- Asset car materials other than paint/glass/tyres are the authors' RGB values.

## In-car camera effects

`tools/highway_incar.py` (hooks in `build_highway_scene.py`) and `tools/render_time_slices.py`
make the view look like a windscreen-mounted ADAS camera. Everything is opt-in; without these
flags the scene and manifest are unchanged.

```bash
venv/bin/python tools/build_highway_scene.py \
    --exposure-s 0.004 --rolling-shutter-line-time-us 15 \
    --ego-speed-kmh 100 --traffic-speed-kmh lanes \
    --windscreen --windscreen-dirt 0.03 --windscreen-rain 0.08 --vms
venv/bin/python tools/render_time_slices.py scenes/generated/highway --spp 64 --bands 24 --jobs 2
```

**Windscreen** (`--windscreen`, `--windscreen-rake-deg 27`, `--windscreen-distance-m`,
`--windscreen-radius-h-m/-v-m` for curvature). A closed laminated shell, 2.1 mm glass /
0.76 mm PVB / 2.1 mm glass, raked 27 deg from horizontal (so the optical axis meets it at
~64 deg incidence), attached to the ego car. pbrt `dielectric` (n = 1.52; PVB, n ~ 1.48, is
treated as index-matched) bounding a `homogeneous` absorbing medium, so it needs the `volpath`
integrator (switched on automatically). The absorption is a smooth model of green iron-bearing
soda-lime glass: the Fe2+ band at ~1050 nm absorbs red/NIR, Fe3+ and the PVB UV absorber cut
below ~380 nm (Bamford, *Colour Generation and Control in Glass*, 1977; Volotinen et al.,
J. Non-Cryst. Solids 354, 2008). Luminous transmittance (CIE A, ISO 9050) is 0.80 at normal
incidence (legal minimum 0.70: UN ECE R43, FMVSS 205 / ANSI Z26.1) and ~0.67 along the camera
axis because of the oblique Fresnel losses; T is ~0.45 at 800 nm and ~0.16 at 1000 nm. Use
`--windscreen-transmittance-csv nm,T` for a measured curve. The default distance keeps the
glass ~1 cm clear of the lens (pinhole: 5 cm; realistic: front element + 1 cm).
`--windscreen-dirt f` adds a stochastic-alpha `diffusetransmission` film (soil reflectance,
40 % diffuse transmission) with mean coverage f; `--windscreen-rain f` adds non-overlapping
spherical-cap water drops (n = 1.333, log-normal base radius, median 0.8 mm,
`--rain-contact-angle-deg 45`) covering fraction f of the glass the camera sees (mesh sizes:
~1000 drops per 10 % coverage). The windscreen sits in front of the camera, not over the road,
so `lighting.reference_illuminance_*` (scene illuminance) is unchanged: the glass attenuation
shows up in the rendered radiance/irradiance and therefore in the electrons, as it would in a car.
`manifest["windscreen"]` records the geometry and the normal/axis luminous transmittance.
With `--haze`, the glass exterior medium defaults to `haze` (`--windscreen-outside-medium` overrides),
so rays leaving the windscreen stay in the atmosphere medium.

**Exposure and motion blur** (`--exposure-s T`). The camera gets
`shutteropen 0 / shutterclose T` and `manifest["exposure"]["integration_time_s"] = T`;
`pbrt_spectral_exr_to_electrons.py` uses that as the integration time (an explicit
`--integration-time-s` that differs prints a warning), so the blur and the electron count
use the same exposure. Pass the same value to `apply_emva_noise.py --integration-time-s` for
dark current. Ego motion (`--ego-speed-kmh`, `--ego-yaw-rate-deg-s`) animates the camera and
the windscreen; each car gets a forward velocity (`--traffic-speed-kmh`: one value, a per-car
comma list, or `lanes` = 125/110/95 km/h by lane and 105 km/h oncoming) via
`TransformTimes 0 <span>` + `ActiveTransform EndTime` (`span` covers the rolling-shutter
readout). Speeds go into `manifest["cars"]` and `manifest["ego_motion"]`. pbrt-v4 cannot
animate area lights ("Animated area lights are not supported"), so a car whose include has an
`AreaLightSource` is kept static and listed in `manifest["incar_warnings"]`.

**Rolling shutter** (`--rolling-shutter-line-time-us`). pbrt has a global shutter, so
`render_time_slices.py` renders row bands (`--pixelbounds`) with the shutter window of the
band's centre row, `[r * line_time, r * line_time + T]`, and composites them. With `--bands B`
the timing error is <= (yres / 2B) line times (24 bands at 720 rows: 15 rows). Each band re-reads
the scene, so render time grows by ~B x scene-load time; `--jobs` runs bands concurrently.

**LED flicker** (`--vms`, `--vms-pwm-hz 100`, `--vms-duty 0.25`, `--vms-phase-ms`). A
roadside dot-matrix variable-message sign: amber AlInGaP LEDs (592 nm, 17 nm FWHM;
Schubert, *Light-Emitting Diodes*, 2006), 24 mm dots on a 40 mm pitch,
`--vms-luminance-cd-m2` time-averaged face luminance (EN 12966 class L3 ~ 6000 cd/m2 is the
default), so each dot runs at L / (fill x duty) while on. Flickering emitters use a generic
contract, `highway_incar.write_emitter(out_dir, id, lines_for_level, pwm=PWM(f, duty, phase),
exposure_window_s=...)`: it writes `emitters/<id>.pbrt` (the level averaged over the exposure,
or the duty cycle without one), `emitters/<id>.on.pbrt` and `.off.pbrt`, and returns the
`Include` line and a `manifest["emitters"]` entry. Any emitter registered this way (e.g.
vehicle LED lamps) flickers. `render_time_slices.py` splits each band's window at the PWM
edges of all emitters and renders every sub-interval with the matching on/off includes, so a
4 ms exposure of a 100 Hz / 25 % sign captures 2.5 ms of on-time or none at all, depending on
the phase and the row (bands across the sign). A plain `pbrt` render of the same scene uses the
exposure-averaged level instead (no row dependence). Samples per pixel are split in
proportion to slice duration, so the composite has the same total spp as a plain render.
