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
`--global-illuminance-lux`, `--road` (see below), `--verge`, `--vegetation`, `--traffic none`,
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

## Road and verge surfaces

`--road` picks the carriageway texture and `--verge` the ground beside it, all ambientCG CC0
2K maps listed in `config/highway_assets.yaml` with checksums. As for the original asphalt,
only the luminance of the colour map is used, normalised to mean 1. It scales a measured
reflectance spectrum (`ROAD_SURFACES` / `VERGE_SURFACES` in `tools/build_highway_scene.py`),
so the albedo comes from the spectrum and the texture adds detail.

| `--road` | Surface | Tile | Spectrum |
|---|---|---|---|
| `asphalt031` (default) | fine-aggregate asphalt | 2.0 m | `asphalt_aged` |
| `asphalt026c` | cracked asphalt | 2.0 m | `asphalt_aged` |
| `asphalt023s` | smooth, fresh fine asphalt | 1.25 m | `asphalt_new` |
| `asphalt030` | grey worn asphalt | 2.2 m | `asphalt_aged` |
| `asphalt021` | cracked dark asphalt | 1.85 m | `asphalt_aged` |
| `concrete046` | concrete pavement | 2.4 m | `concrete` |
| `pavingstones070` | granite setts | 1.15 m | `concrete` |
| `pavingstones128` | concrete block paving | 3.5 m | `concrete` |

`--verge grass004` (default), `gravel023` (grey gravel, 1.5 m tile, `concrete`) or
`ground062s` (sandy soil, 1.6 m, `soil`). The distant fields stay grass.

Tile sizes are ambientCG's stated dimensions. The Road0xx series was left out because it has
lane markings baked in. Road wear models asphalt only, so `--road-wear` defaults to `none` on
concrete and paving, and asking for wear there is an error. The setts and the gravel use a grey
concrete spectrum, not a measured granite or gravel one.

Without road wear, `--road-detile auto` (default) breaks up the texture repeat, which otherwise
shows as faint banding across the road (every 2.4 m on concrete):
- *Asphalt and concrete046* are re-synthesised into a 6 m periodic tile with the wear model's
  variance-preserving stochastic tiling (below). Concrete uses offsets only, no rotations, so its
  form marks keep one direction. Luminance and normals are re-tiled; roughness stays 0.35.
- *Concrete and paving* are also multiplied by a mean-1 tonal mask: 60 m periodic, 3 m correlation,
  SD 8 %. These are nominal slab/batch-to-batch values, not measured pavement statistics.
- *Paving* (setts, block paving) is not re-synthesised, because blending offset copies would break
  the joint pattern. The stone-by-stone pattern still repeats every 1.15 m (setts) or 3.5 m
  (blocks); only its brightness varies.

Mean reflectance is unchanged. `--road-detile off` writes the old scene file. The manifest
records `road.detile` (`null` when wear is on or the road is untextured).

## Road surface wear

`--road-wear none|light|moderate|heavy` (default `moderate` on asphalt roads, `none` otherwise) ages the road with seeded,
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
  are smooth coated diffuse by day; with retroreflection on (dusk/night) they become cube-corner
  reflex reflectors (see "Reflex reflectors" under Night and dusk).
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
# Prague sky model (opt-in; download the dataset yourself, see below); visibility 20-131.8 km
venv/bin/python tools/build_highway_scene.py --sky prague --sun-elevation 43 \
    --prague-dataset ~/data/PragueSkyModelDatasetSWIR.dat --prague-visibility-km 59.4
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
- `--sky prague` (opt-in): the Prague sky model (Wilkie et al. 2021, "A fitted radiance and
  attenuation model for realistic atmospheres", ACM TOG 40(4):135; Vevoda et al. 2022, "A Wide
  Spectral Range Sky Radiance Model", Pacific Graphics), ground-level sky radiance. A NumPy port
  of the authors' Apache-2.0 C++ (`tools/prague_sky.py`, licence in
  `third_party/licenses/pragueskymodel-LICENSE.txt`) reads the dataset by memory map; it matches
  the reference CLI to 6e-8 relative (float32 output) at sun elevations -2 to 62 deg with
  interpolated visibility/albedo. **The dataset is not in open-cam**: the authors distribute it
  (Google Drive, linked from https://github.com/PetrVevoda/pragueskymodel) without a stated
  licence, so download it yourself and pass `--prague-dataset PATH` or set
  `OPEN_CAM_PRAGUE_DATASET`. Tested with the 280-2480 nm `PragueSkyModelDatasetSWIR.dat`
  (55 channels of 40 nm, visibility 20-131.8 km, ground level only); the other datasets have
  the same format. The 40 nm channels (constant within each band, as in the reference code)
  are integrated against the CMFs over 360-830 nm and given the daylight-basis metamer like
  `hosek`; wavelengths above 830 nm are not rendered by the spectral film, so the model's NIR
  range is not used yet. Ground albedo 0.1; the solar disc still comes from the builder's sun
  model. At 40 deg sun elevation and 59.4 km visibility the Prague sky gives 13.3 klx diffuse
  horizontal (Hosek turbidity 3: 15.3 klx), horizontal CCT 11.8 kK (Hosek 14.6 kK) and zenith
  CCT 16.2 kK (Hosek 40.4 kK).
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
into a visible Lambertian lens quad.

**Rear lamps** (`tools/highway_taillamps.py`, `--tail-lamps detailed`, default). Each traffic
vehicle gets a rear lamp cluster per side that follows its body: a 1 cm grid over the lamp outline
is ray-cast from behind onto the car asset's meshes, and the lens and emitters are bilinear meshes
2 mm and 4 mm proud of the hit surface. The outlines (`LAYOUTS`) follow each asset's own lamp
geometry: the BMW M6 lamp glass, the Pontiac GTO's slot lamps behind the chrome louvres and the
vintage car's round fender lamps. The box proxy car and the heavy vehicles (flat bounding-box
rear) get generic layouts; trucks, the van and the bus have amber / red / clear chambers.
* *Photometry at H-V (Lambertian, L = I / A):* tail 6 cd, stop +80 cd per side and a centre
  high-mounted stop lamp (CHMSL) of 40 cd on the BMW and the proxy (UN ECE R7 R1 4–17 cd,
  S1 60–185 cd, S3 25–80 cd). The manifest (`lighting.artificial.tail_lamps[*].sides`) gives
  each side's lens and emitting area, intensity and peak luminance.
* *LED lamps* (BMW, proxy, bus): the tail function lights a 2 cm contour light guide round the
  lens edge (~580 cd/m² on the BMW); braking adds the inner area (~6400 cd/m²). Red LED spectrum
  626 nm, 17 nm FWHM.
* *Incandescent lamps* (Pontiac, vintage car, trucks, van): the whole red chamber glows, with a
  bulb hot spot at the centre (luminance weight 1 + 4 exp(−r²/2σ²), σ = 2 cm). Braking raises
  the same chamber (dual filament). Spectrum: 2856 K behind a 610 nm long-pass red lens.
* *Unlit lenses:* glossy `coateddiffuse` with nominal red, amber or clear reflectances, so they
  reflect other cars' headlamps.
* *Rear fog lamps* (`--rear-fog-lamps auto|on|none`; `auto` = on when the haze visibility is below
  `--rear-fog-visibility-m`, default 50 m as in StVO §17(3); the visibility is the MOR, converted from
  `--visibility-definition koschmieder` and taken inside a `--fog-bank`, so the default `--haze fog`
  at 150 m leaves them off): one 200 cd
  lamp (UN ECE R38: 150–300 cd) on the car's left side, as R48 requires when only one is fitted in
  right-hand traffic. It's placed at least 100 mm from the stop lamp: low in the bumper (BMW,
  proxy), beside the combination lamp (trucks, tractor), below the vertical cluster (van) or inboard
  of it (bus). The Pontiac and the vintage car have none, as cars of their era weren't fitted with
  them. StVO §17 only permits them below 50 m visibility; `auto` turns them on for any `--haze fog`
  scene. The manifest gives `tail_lamps[*].rear_fog` (intensity, area, peak luminance) or `null`.
* *Beam shape* (`--rear-lamp-beam shaped`, default; `lambertian` = cosine emitters): the
  H-V intensities above are kept and the intensity falls off away from straight behind as
  I(h, v) = I(H-V) f(h) g(v). For tail and stop lamps f and g go through the ratios of the
  FMVSS 108 one-section test-point minima (49 CFR 571.108 Table VIII taillamp and Table IX
  stop lamp, e.g. tail H-V 2.0, H-10 0.8, 5U-20 0.3 cd), within 0.07 at every test point;
  so 10° to the side a tail lamp gives 40 % and 20° to the side 15 % of its H-V intensity.
  The scaled intensities meet every FMVSS minimum and stay below its maxima. ECE R7 tests the
  same kind of grid (H, 5° and 10° up/down, out to 20° left/right); its text wasn't available to
  check, so the shape follows the FMVSS ratios. The rear fog lamp has a nominal R38-style
  field (≥ 75 % of H-V within ±10° H, ±5° V, so ≥ 150 cd; R38 text not checked). Beyond the
  test grid the beam keeps falling to zero at 90°. Lamps whose one emitter carries tail and
  stop (bulbs, LED fallback) use the intensity-weighted mix; the CHMSL uses the stop shape.
  Emitters stay diffuse area lights (so the lens is seen lit), and pbrt patch 0016 scales their
  radiance by shape / cos θ from equal-area maps written to `textures/rear_beam_*.exr`; this is
  exact for patches facing straight back, while body-conforming lenses tilt a few degrees. The
  manifest gives `tail_lamps[*].beam` (`fmvss108_ratios` or `lambertian`). `TestPbrtAngularAreaLight`
  renders a lamp from the test directions and checks the radiance follows the map.
* *Not modelled:* left/right asymmetry of real beams (inboard vs outboard), the inner optics
  (pillow lenses, reflector facets) and sparkle, rear reflex reflectors, indicators flashing and reversing lamps lit. Heavy
* *Direction indicators* (`--indicators none|lead|all`, `--indicator-side left|right|hazard`) and
  *reversing lamps* (`--reversing-lamps none|lead|all`), both off by default. `lead` is the nearest
  car ahead in the camera lane. Indicators are amber (592 nm LED, or 2856 K behind a 575 nm
  long-pass lens), 100 cd per side, flashing at 1.5 Hz with 50 % on-time (UN R48 asks for
  90 ± 30 flashes/min). Each car's flasher is a `highway_incar.PWM` emitter in
  `manifest["emitters"]`, with a per-car phase, so `tools/render_time_slices.py` renders it on or
  off per row. A still frame without `--exposure-s` shows the time average (half intensity).
  Reversing lamps are white (6000 K LED or 2856 K), 150 cd per side, steady. The intensities are
  nominal values in the usual UN R6 / R23 ranges; I couldn't check the regulation texts. The
  lamps sit in the amber and clear chambers of the trucks, van, bus and tractor. Cars whose
  cluster has only a red lens get an amber outer third and a clear inner third when either
  option is on (the red tail and stop lamps keep their intensities over the smaller area). The
  manifest gives `tail_lamps[*].indicator` and `tail_lamps[*].reversing` (or `null`). Detailed
  tail lamps only.
* *Not modelled:* the lamps' angular intensity distribution (they are Lambertian, while R7 lets
  the intensity fall off away from H-V), the inner optics (pillow lenses, reflector facets) and
  sparkle, rear reflex reflectors, and front and side indicators. Heavy
  * *Not modelled:* left/right asymmetry of real beams (inboard vs outboard), the inner optics
    (pillow lenses, reflector facets) and sparkle, rear reflex reflectors, and front and side indicators. Heavy
  vehicles use the flat rear of their bounding box.

`--tail-lamps simple` restores the old flat 0.30 × 0.10 m red quads at the bounding-box rear
corners, which float beside cars whose bounding box includes the mirrors.

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
- Reflex (cube-corner) reflectors, with retroreflection on: the `ra` of each reflector is its
  coefficient of luminous intensity (CIL, mcd/lx) divided by its area, x1.2, and the lobe is
  narrower than sheeting. Every tabulated (observation, entrance) minimum is met:
  - car rear reflectors (red, FMVSS 108 Table XVI-a / SAE J594: 420 mcd/lx at 0.2/0 deg, 140 at
    20 deg, 3 at 1.5/20): w = 0.32 deg, q = 12. One patch per side on the body (`LampLayout.reflex`,
    on the BMW's own reflector material), every model but the vintage car; per-car areas in
    `tail_lamps[*].reflex_reflectors`;
  - procedural cyclist (16 CFR 1512 Tables 1-2): red rear and white front reflectors, amber
    pedal and spoke reflectors; front/rear/spoke w = 0.33 deg, q = 3.5 (30-50 deg entrance
    points), pedals w = 0.375 deg; listed under `vru.reflectors`. The imported bicycles,
    motorbikes and scooters (`--vru-bike`, `--vru motorcyclist`, parked two-wheelers) don't get
    reflectors: their meshes have no reflector parts to attach the material to;
  - raised pavement marker lenses: ASTM D4280 minima as commonly cited (white 3.0 cd/fc,
    yellow 0.6x, red 0.25x; the standard text was not checked), over the sloped 3150 mm^2 lens
    face, q = 0 because the prisms face along the road rather than along the lens normal;
  - urban cone and drum bands (`--work-zone`): texels of the asset's white bands (bright,
    near-neutral; `street_furniture.retro_band_mask`) mix to the Type III `sheet_white` sheeting.
  The windscreen camera sits ~0.7 m above the headlamps, so the observation angle is ~2-4 deg at
  10-20 m and the reflectors only light up strongly from ~40 m out.
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
stays flat (no depressed water surface).

**Raindrop ripples** (`--rain-ripples auto|off`, default `auto`: on when `--rain-mmh` > 0). The
standing water — the puddles, and the whole film at `flooded` — gets a normal map of impact
rings, one 6 m tile (2 mm texels) on the road-wear UVs. Impacts are a Poisson process at the
Marshall-Palmer rate of drops ≥ 0.5 mm reaching the road (∫N(D) v(D) dD: ≈750 m⁻² s⁻¹ at
1 mm/h, ≈3100 at 10 mm/h, ≈6400 at 40 mm/h); each leaves a ring wave packet moving out at
0.23 m/s (the minimum phase speed of capillary-gravity waves, λ = 17 mm), with an amplitude
∝ D that decays with a 0.25 s time constant and spreads as 1/√r. The ring speed and wavelength
are textbook water-wave values; the decay time and the 0.15 peak slope of a ring from a 2 mm
drop are nominal, not fitted to measurements. The map is a frozen snapshot: it doesn't move
during `--exposure-s` or between frames. Not modelled: splash crowns, secondary droplets,
bubbles, and ripples on the thin `wet` film outside the puddles. The manifest's
`road.wet.ripples` records the impact rate, the ring count and the rms and 99th-percentile
slopes of the map.

**Tyre tracks** (`--tyre-tracks F`, 0–1, default 0 = off; `damp`/`wet`, no `--puddles`, needs
`--road-wear` ≠ none). Tyres squeeze the film out of the wheel paths, so they dry first on a
drying road. The wheel-path map (normalised) sets a dryness mask, smoothstep 0.3–0.8 scaled by
`F`, that mixes the wet film with the dry road (pbrt `mix`; the dry road's textures are kept
under `dry:` names). It can't be combined with `--puddles`, which put the water *in* the
wheel-path ruts. `road.wet.tyre_tracks` records the dry area fraction.

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
with height; it is rendered as one homogeneous box per vehicle with the plume-mean extinction,
so the downstream, wheel-track and vertical structure is averaged out. With `--spray` the road is
split into 2 m (25 m beyond 100 m) strips: on the default two 1.5 km road triangles pbrt-v4 loses
track of the spray medium for rays that leave a box and hit the road (black road patches). Static scenes (`--traffic-speed-kmh 0`) use the nominal lane speeds
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

**Optical model.** The medium is parameterised by the visibility V at road level. By default
(`--visibility-definition mor`) V is the meteorological optical range (MOR) used in weather
reports, METAR and fog-chamber tests: the path that reduces a collimated beam to 5 %, so the total
extinction at 550 nm is sigma_ext = -ln(0.05)/V = 2.996/V (WMO-No. 8, ch. 9; ICAO Annex 3; NOAA
Federal Meteorological Handbook No. 1). `--visibility-definition koschmieder` uses the classic
2 % visual range of a black object instead, sigma_ext = -ln(0.02)/V = 3.912/V (Koschmieder 1924;
Middleton 1952). That was open-cam's convention before this option and gives 1.31x the
extinction for the same V; use it to reproduce older scenes. sigma_ext is split into
Rayleigh scattering, sigma_R = 1.16e-5 /m x (lambda/550)^-4.08 (sea-level air; Bucholtz 1995), and
aerosol, sigma_aer(lambda) = (sigma_ext - sigma_R(550)) x (lambda/550)^-alpha with the Ångström
exponent alpha (Ångström 1929): ~1.3 continental haze, ~0.5 mist, 0 for fog whose droplets are
much larger than the wavelength (grey fog). The aerosol single-scattering albedo omega sets
sigma_a/sigma_s. The phase function is Henyey-Greenstein; pbrt takes one g per medium, so g is
the scattering-weighted mean of the aerosol g and Rayleigh (g = 0), recorded as `hg_g_effective`.
Haze aerosol g ~ 0.7 (Shettle & Fenn 1979), fog droplets ~0.85 (Mie, Deirmendjian 1969).

| preset | V (MOR) | alpha | g | omega | vertical profile | tau_550 (vertical) | volpath maxdepth |
|---|---|---|---|---|---|---|---|
| `clear` | 30 km | 1.3 | 0.70 | 0.95 | exponential, H = 1.2 km | 0.12 | 8 |
| `hazy` | 8 km | 1.0 | 0.70 | 0.90 | exponential, H = 1.0 km | 0.37 | 8 |
| `mist` | 2 km | 0.5 | 0.75 | 0.98 | 300 m layer | 0.52 | 24 |
| `fog` | 150 m | 0 | 0.85 | 0.999 | 150 m radiation-fog layer | 3.45 | 64 |

All parameters can be overridden (`--visibility-m`, `--visibility-definition`, `--haze-angstrom`, `--haze-g`,
`--haze-albedo`, `--haze-profile exponential|layer`, `--haze-height-m`, `--haze-maxdepth`).

**Droplet (Mie) fog, opt-in: `--fog-droplets advection|radiation`.** Henyey-Greenstein with
g = 0.85 has the right mean cosine for fog but the wrong shape. `tools/fog_mie.py` computes
Lorenz-Mie scattering (Bohren & Huffman BHMIE) of liquid water (Hale & Querry 1973 index) for
the Shettle & Fenn (1979) modified-gamma droplet distributions: `advection` (heavy advection
fog, modal radius 10 um) and `radiation` (moderate radiation fog, 4 um). The size-averaged
phase function (450-650 nm, weighted by scattering) goes to pbrt as a 700-bin table
(`"float phasetheta"`/`"float phasevalues"`, pbrt patch 0013; bins down to 0.01 deg near the
forward direction), and the spectral extinction and albedo replace the Ångström law,
`--haze-g`, `--haze-albedo` and `--haze-phase`. V still fixes sigma_ext(550). Compared with
HG of the same g:

| | advection (g = 0.872) | radiation (g = 0.853) | HG g = 0.85 |
|---|---|---|---|
| share of scattered light within 1 deg / 5 deg / 20 deg | 0.40 / 0.50 / 0.67 | 0.16 / 0.46 / 0.65 | 0.006 / 0.13 / 0.63 |
| 4 pi p(90 deg) (side scatter) | 0.020 | 0.037 | 0.123 |
| 4 pi p(180 deg) (back scatter, glory) | 0.41 | 0.63 | 0.044 |
| sigma_ext(1000 nm) / sigma_ext(550 nm) | 1.014 | 1.033 | (alpha = 0: 1) |

So Mie fog puts much more light into a sub-degree diffraction spike (a tighter core round
lamps), much less into side scatter, and about 10x more straight back (the veil from your own
headlamps). It is also grey: for 4-10 um droplets the extinction rises by only 1-3 % from 400 to
1000 nm, so the model gives NIR no visibility advantage in fog (that needs sub-micron haze
droplets). pbrt's phase function is wavelength-independent, so the diffraction peak width is
the visible average. The Rayleigh lobe is dropped (w_R <= 0.01 at V <= 2 km). The road
illuminance MC samples the same table. `tests/test_fog_droplets.py` checks the Mie code against
the BHMIE reference case, the table sampler, and pbrt single-scatter ratios at 30/90/170 deg
plus a white furnace.

**Fog banks and patchy fog, opt-in: `--fog-bank Z0:Z1`, `--fog-patchiness A`.** By default the
medium is horizontally uniform. pbrt patch 0014 multiplies the uniformgrid density by a horizontal
map m(x, z) in [0, 1]. The map is bilinear on a 10 m x 5 m lattice over x = -400..400 m and
z = -200..2000 m (z = distance ahead of the camera), and is clamped beyond it.
`--fog-bank Z0:Z1` gives full density between Z0 and Z1 (Z1 may be `inf`), with logistic edges
that rise from 10 % to 90 % over `--fog-bank-ramp-m` (default 40 m), and `--fog-bank-outside`
(default 0.05) elsewhere. `--fog-patchiness A` cuts smoothed Gaussian-noise holes of depth up to
A into the map (`--fog-patch-scale-m` correlation length, default 60 m; `--fog-patch-seed`).
`--visibility-m` is the visibility where m = 1 (inside the bank). The manifest's road-illuminance
reference is still the full-density slab. Because m <= 1, pbrt's density majorant stays valid
and the transport is unbiased. Thin regions only cost extra null collisions.
`tests/test_fog_patchy.py` checks the map and a pbrt transmittance through a bank against
exp(-sigma * integral of m).

`--haze-phase mixed` (opt-in) replaces the single effective Henyey-Greenstein lobe with the
physical mixture `p = w_R p_Rayleigh + (1 - w_R) p_HG(g)`, `w_R = sigma_R / sigma_s` at 550 nm
(pbrt patch 0011, medium parameter `"float rayleighfraction"`). Both modes have the same mean
cosine; the mixture puts more light into side and back scatter in clear air (w_R 0.12 for
`clear`, 0.03 `hazy`, ~0 in `mist`/`fog`). pbrt's phase function is wavelength-independent, so
w_R is fixed at its 550 nm value. The
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
and `no_medium_horizontal_illuminance_lux`). `TestPbrtHighwayHaze` checks it against pbrt: a Lambertian ground probe (±5 km quad, camera 0.2 m) rendered with `volpath` under each preset and under no medium gives the same illuminance ratio as the manifest within 5 % (measured, manifest vs pbrt: clear 0.967 vs 0.963, hazy 0.886 vs 0.874, mist 0.912 vs 0.911, fog 0.744 vs 0.745). Keep such probes ≲10 km across: on a 300 km float32 quad the hit-point error biases the shadow rays by tens of percent. With `--haze-light-reference road` the lights
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
  `--vegetation-pool extended` adds 13 more Poly Haven CC0 species to the draw: jacaranda,
  island tree 01, a small broadleaf tree, a fir tree (about 18 m) and a fir sapling, Searsia
  burchellii, shrub 03, wild rooibos, three grass clumps, a fallen dead trunk and a stump
  (`SPECIES_EXTENDED` in `tools/highway_variety.py`; trunks, stumps and grass count as
  shrubs). Island tree 03 is left out: its mesh includes a sand and rock base. Without
  `--seed` it uses a fixed mix of island tree 02, fir tree, small tree, shrub 02 and grass.
  The default `classic` pool leaves every seed's draw unchanged. Fetch the extra assets with
  `python tools/fetch_highway_assets.py` (about 870 MB at 1k; the fir tree alone is 460 MB
  and 7.0 M triangles, the jacaranda 200 MB and 3.9 M; a 1280-px day frame with both peaks
  at 6 GB in pbrt).
  Several Poly Haven files are a row of 2-7 versions of one plant: the fir tree, fir and pine
  saplings, both Searsias, wild rooibos, shrubs 02 and 03 and the medium grass clumps.
  `--plant-variants split` (default) splits them into separate plants
  (`tools/plant_variants.py`: shapes are grouped by ground footprint, so each trunk stays with
  its crown) and places one randomly chosen version, centred, at each placement. Spacing is
  divided by the number of versions, so the plant count matches the old rows. Each version keeps
  its own height from the file. The version choice has its own random stream; the positions,
  scales and rotations are still seeded from `--seed`, but there are more placements, so they
  differ from the old rows. `--plant-variants rows` writes the old scene file. Bermuda grass is
  a sparse patch of single blades and stays whole. The manifest's `vegetation` entries record
  `plants_in_file`.

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

**Passenger cars, extended pool** (opt-in `--car-pool extended`; CC-BY-4.0, Sketchfab via the
same Objaverse mirror, attribution in `config/highway_assets.yaml`): 13 modern cars (hatchback,
saloon, estate, SUV, pickup, taxi) join the 3 CC0 cars in `--seed` traffic: "Tesla 2018 Model 3" by uchiha.321abc, "Renault Logan 2004" by goooddenis, "Daewoo - Nexia/Cielo" by SevKa, "Honda Civic Type R -98 (Free Asset)" by tiedtke, "2001 Crown Victoria Taxi Game Prop" by haoliu95, "Nissan Titan 4x4" by David_Holiday, "Range Rover Evoque" by minghau, "2015 Audi Q7" by razor24, "Volkswagen Passat 3D Scan" by zdenkoroman, "2015 Hyundai Tucson" by razor24, "2018 Mercedes Benz E63 AMG S Wagon" by sinnik, "Mazda CX-5" by tnnv, "VW Tiguan 2021" by source5525.
Fetch them with `tools/fetch_highway_assets.py car_tesla_model3 car_renault_logan car_daewoo_nexia car_honda_civic_ek car_crown_victoria_taxi car_nissan_titan car_range_rover_evoque car_audi_q7 car_vw_passat_b6 car_hyundai_tucson car_mercedes_e63_estate car_mazda_cx5 car_vw_tiguan`. `highway_variety.CARS_GLB` gives each
model its nominal manufacturer length (not checked against data sheets), forward axis and paint
materials; cars with a textured livery or body (taxi, Titan, Passat scan, Logan) keep the
authors' colours. Rear lamps are traced onto the body like the CC0 cars: where the asset has a
rear-lamp material (Tesla, Evoque, Q7, Tucson, E63, CX-5, Tiguan) the lamp fills that lens,
looking through clear outer lens/window materials; otherwise it fills a generic cluster region on
the body. Widths come from the asset's bounding box and include the mirrors. The default
`classic` pool leaves existing `--seed` scenes unchanged.

## Measured / spectral materials (opt-in)

Three independent flags in `tools/highway_materials.py`; all default off, and with them off the
generated scene is byte-identical to before (`tests/test_highway_materials.py::DefaultOffTest`).

```bash
venv/bin/python tools/highway_materials.py fetch   # optional prefetch into the gitignored scenes/assets/measured/
venv/bin/python tools/build_highway_scene.py --car-paint measured --spectral-library usgs --fluorescent-sign yellow-green
venv/bin/python tools/validate_measured_materials.py   # furnace + fluorescence + swatch checks -> out/measured_materials/
```

| flag | what changes | data source (licence) |
|---|---|---|
| `--car-paint measured` | basecoat of the prepared cars and proxy cars -> pbrt-v4 `measured` material: blue -> `ilm_solo_m_68`, silver -> `ilm_l3_37_metallic`, darkgreen -> `ilm_l3_37_dark_green`, white -> `cm_white`, red -> `vch_dragon_eye_red`; black / gray keep the analytic coat | RGL material database, Dupuy & Jakob 2018, ACM TOG 37(6) 274, doi:10.1145/3272127.3275059 (every entry CC0 1.0); `*_spec.bsdf`, sha256-pinned, fetched at build time and copied into `<out>/measured/` |
| `--spectral-library usgs` | `asphalt_aged.spd` (also used by road wear), `concrete.spd`, `galvanized.spd` -> measured spectra; car-asset `rgb eta/k` conductors -> Cr optical constants; near-neutral dark RGB trim (max <= 0.12) -> black ABS spectral shape scaled to the asset's luminance | USGS Spectral Library v7, Kokaly et al. 2017, USGS DS 1035, doi:10.5066/F7RR1WDJ (public domain): Asphalt GDS376 (black road, old), Concrete GDS375, Galvanized sheet metal GDS334, Plastic ABS GDS341; Cr: Rakic et al. 1998, Appl. Opt. 37, 5271 (refractiveindex.info, CC0) |
| `--fluorescent-sign yellow-green\|orange` | adds a 0.9 m plain diamond on the right verge (z = 95 m) with the patched `fluorescent` material | dye spectra are **fitted, not measured** (see below) |

**Fluorescence.** `third_party/patches/0002-fluorescent-material.patch` adds a `fluorescent`
material: Lambertian reflectance R plus a separable Donaldson (excitation/emission) matrix
\(M(\lambda_{em},\lambda_{ex}) = x(\lambda_{ex})\,e(\lambda_{em})\,\lambda_{ex}/\lambda_{em}\) for
\(\lambda_{ex}<\lambda_{em}\) (x = quantum yield x absorptance, e = unit-area photon emission
spectrum; the \(\lambda_{ex}/\lambda_{em}\) factor converts photon to energy yield). The path/volpath
integrators pick the reradiation branch with probability `fluorescenceprobability` (default 0.5),
sample the excitation wavelength uniformly and continue the path at the excitation wavelengths
(Lambertian reradiation) - real wavelength-shifting transport, not a colour boost. The ASTM D4956
Type XI dye spectra are a logistic absorption edge + Gaussian emission band fitted so the
bispectral D65 colour sits at the centroid of the 23 CFR 655 Subpart F Appendix Table 3 colour box
with the Table 3a typical fluorescence luminance factor (yellow-green: xy (0.411, 0.548), Y 92.5,
Y_F 20.0; orange: xy (0.590, 0.381), Y 53.0, Y_F 15.0). ASTM E991 bispectral measurements of real
sheeting were not available. Not modelled: retroreflection of the fluorescent sheeting at night,
legends on the fluorescent sign, polarisation, fluorescence in the bidirectional (BDPT/MLT) and GPU
integrators, UV below 360 nm (pbrt film range) although daylight UV excites real sheeting.

**Validation** (`tools/validate_measured_materials.py`, 64 spp):

- White furnace (unit uniform environment, one bounce, theta_o < 76 deg): max directional albedo
  Spectralon 0.989 (mean 0.950 vs ~0.99 nominal for SRS-99), `ilm_l3_37_metallic` 0.665,
  `irid_flake_paint1` 0.227, `ilm_solo_m_68` 0.101; `cm_white` reaches **1.078** - the RGL data
  are not exactly energy conserving and pbrt does not enforce it.
- Fluorescent sheet vs Lambertian white, D65 at 0/0: per-bucket total radiance factor matches the
  analytic bispectral value to 0.008 (yellow-green, peak 1.17) and 0.023 (orange, peak 3.10), path
  and volpath identical; both rendered colours lie in their Type XI boxes.
- Swatches (sphere mean, D65 sun + sky, CIEDE2000 analytic vs measured): white 11.6, silver 15.3,
  blue 15.1, darkgreen 11.1, red 2.4; fluorescent vs reflected part only: yellow-green 6.9,
  orange 15.7. The measured paints are different physical paints from the analytic curves, so
  these are differences, not errors.

Approximations: the RGL paints are a stand-in palette (ILM film-prop paints and TeckWrap vinyl
films for white/red), not measured OEM automotive paints; RGL BRDFs are isotropic and tabulated
to ~80 deg incidence; heavy GLB vehicles (trucks/buses) keep their textured analytic paint. No
measured asphalt BRDF exists in RGL, so the asphalt keeps the analytic coated-diffuse lobe
(roughness 0.35, and the road-wear roughness model) with only the spectral albedo measured; the
lobe is not fitted to CIE 144 road reflection tables. USGS spectra are lab bidirectional
(ASD, 350-2500 nm) reflectances of single samples.

## Traffic signs with real artwork (`--sign-set`, `tools/highway_signs.py`)

Opt-in row of real traffic signs on posts beside the right shoulder; the default scene is unchanged.

```bash
python tools/fetch_highway_assets.py $(python -c "import yaml; d=yaml.safe_load(open('config/highway_assets.yaml')); print(' '.join(k for k, a in d['assets'].items() if a['kind'] == 'sign_svg'))")
python tools/build_highway_scene.py --sign-set stvo --time-of-day night --out-dir out/signs_night
```

- *Artwork:* public-domain SVGs from Wikimedia Commons (asset kind `sign_svg`, pinned SHA-256): 8 German StVO
  signs (101, 138, 205, 206, 267, 274-60, 306, 350) and 4 US MUTCD signs (R1-1, R2-1, W1-1R, W11-2). Preparing them
  needs `cairosvg`. Each SVG becomes an alpha cutout (so octagons, triangles and discs keep their outline) and one
  mask per sheeting colour (nearest of white/red/yellow/blue/green/black; colours under 1 % of the face are
  merged into their nearest neighbour to drop anti-aliasing fringes).
- *Materials:* each colour region is its own spectral material, chained with `mix`. By day the sheeting is diffuse
  (`sheeting_*` analytic reflectances). At night/dusk with retroreflection on, every colour is the `retroreflective`
  sheeting of the night model (ASTM D4956 Type III minima x margin). Blue has no D4956 entry in the model and uses
  8 % of white, a nominal value. `--sign-sheeting type1|type3|type11` scales R_A by the white-minimum ratios
  (70/250, 1, 580/250); these are nominal, not checked against the standard.
- *Layout:* first sign at `--sign-start-m` (25 m), then every `--sign-spacing-m` (15 m), `--sign-offset-m` (1 m) beyond
  the paved edge, lower edge at `--sign-bottom-m` (2 m). Sizes are nominal (StVO size class 2, MUTCD conventional
  road) and stored per asset as `width_m`.
  The row replaces the procedural "SPEED LIMIT 65" plate at 60 m; the green guide sign stays.
- *Ground truth:* `highway_manifest.json` -> `signs` lists each sign's code, distance, world bounding box, size,
  colour fractions, source and licence.
- *Not modelled:* sheeting ageing and dirt, embossed legends, sign-face sheen, sign-specific observation-angle
  tables.

## SWIR (`--spectral-lambda-max` > 830, `tools/highway_swir.py`)

Day scenes can be rendered past 830 nm with the 2500 nm pbrt build; `--swir-spectra auto` then
splices G173 sun/sky and measured-leaf tails onto the visible spectra and `--film-lambda-min/max`
choose the rendered band. See [SWIR.md](SWIR.md#highway-scene-in-the-swir-toolshighway_swirpy).

## Known approximations

- Daytime markings use a glass-bead-like rough clear coat over a spectral binder and sign
  sheeting is diffuse; with `--time-of-day dusk|night` (or `--retroreflective on`) both switch
  to the patched pbrt `retroreflective` material (see "Night and dusk" below).
- By default the sky radiance is RGB, upsampled by pbrt; only the sun is spectral (see
  [Spectral sky](#spectral-sky) for the opt-in spectral sky).
- Texture colour maps only modulate luminance. Wet road: flat puddles (no depression; raindrop
  ripples are a static normal map, no splashes; falling rain is opt-in, see
  [Falling rain](#falling-rain---rain-mmh-toolshighway_rainpy)), tyre tracks only from the
  wheel-path map (no per-vehicle wet tracks or spray-wetted lanes), spray density
  scaling (car factor, flooded ×2, linear speed) assumed; wet markings use EN 1436 class
  minima, not a measured wet retroreflection function.
- Haze: one HG phase function per medium (Rayleigh folded into g), horizontally uniform
  medium, flat Earth for the medium (the backdrop has curvature); without `--haze` the horizon
  is sharp and aerial perspective comes only from the sky map.
- Asset car materials other than paint/glass/tyres are the authors' RGB values unless `--spectral-library usgs` (chrome and near-black trim only, see below); sign sheeting is diffuse by day unless `--fluorescent-sign`.

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

**Static scenes: `--linear`.** With no ego or traffic motion, radiance is linear in each
emitter's on/off state, so `render_time_slices.py --linear` renders one frame with every PWM
emitter off (B) and one per emitter with only that emitter on (E_e), then composes
`L(row) = B + sum_e f_e(row) (E_e - B)` with each row's exact on-fraction `f_e(row)`. That is
1 + N_emitters full-frame renders at the full spp, independent of `--bands` and the number of
PWM edges, with no band timing error. It refuses scenes whose manifest records any camera or car
velocity. All renders share one `--seed`, so B and E_e are correlated and most of the path noise
cancels in the difference.

## Vulnerable road users: AEB pedestrian / cyclist targets (`tools/highway_vru.py`, opt-in)

`--vru pedestrian|child|cyclist|motorcyclist|escooter` adds one Euro NCAP-style AEB test target; the default (`none`) leaves
the scene unchanged.

```bash
# obstructed child stepping out between two parked cars, 25 m ahead
venv/bin/python tools/build_highway_scene.py --vru child --vru-obstruction parked-cars --vru-distance-m 25
# cyclist riding away in the right lane, at night under the ego low beams
venv/bin/python tools/build_highway_scene.py --vru cyclist --vru-motion along --vru-offset-m -1.2 --time-of-day night
```

| flag | meaning |
|---|---|
| `--vru-distance-m` | target distance ahead of the camera (default 30 m) |
| `--vru-offset-m` | target centre from the right carriageway edge; negative = into the lane (default 0.4 m, i.e. on the shoulder) |
| `--vru-motion` | `crossing` (facing right-to-left, nearside scenarios) or `along` (longitudinal) |
| `--vru-gait-phase` | stride / pedal phase in [0, 1) |
| `--vru-speed-kmh` | target speed during an exposure (`--exposure-s`); default nominal 5 km/h walking, 15 km/h cycling, 30 km/h motorcycling; `0` = static |
| `--vru-obstruction parked-cars` | two parked cars on the shoulder with a 1 m gap around the target |
| `--vru-clothing` | `ncap` (black top, blue trousers, as the Euro NCAP targets), `dark`, `light`, `red`, `yellow` |
| `--vru-skin-tau` | melanin optical depth of the skin reflectance (`iqlab.skin`) |

- *Geometry:* procedural articulated meshes (ellipsoids and tapered tubes with smooth normals) built to
  nominal target sizes: adult 1.80 m, child 1.154 m, bicycle about 1.8 m long with 0.70 m wheels. The
  ISO 19206 / Euro NCAP documents are paywalled, so these sizes are nominal and were not checked against
  the standard text. No third-party asset is used.
- *Spectra:* measured USGS splib07a fabrics and plastics (`spectra/usgs_nir`, 350-1100 nm) for clothing;
  ColorChecker light skin with a melanin tilt for skin. The ColorChecker data stop at 730 nm and are held
  constant beyond, so skin NIR reflectance is too low (real skin rises towards ~800 nm). All diffuse, no
  subsurface scattering, no fluorescence.
- *Ground truth:* `highway_manifest.json` → `vru` has the target's world-space bounding box, anchor
  point, distance, pose and obstruction cars.
- *Traffic:* same-direction traffic cars within 12 m of the target are dropped so a moving car
  does not hide it; the manifest lists them under `vru.removed_traffic`. Use `--traffic none` for an empty road.
- *Motion:* without `--exposure-s` the scene is static. With it, the target moves along its heading
  at `--vru-speed-kmh` over the whole exposure span (including the rolling-shutter readout), so it
  is motion-blurred together with the ego and traffic motion. The procedural bicycle's wheels roll,
  turning by v·t/r about their axles. The builder stops if a wheel would turn more than 150° in the
  span, because pbrt interpolates the rotation between shutter open and close. The manifest's `vru`
  has `speed_kmh`, `velocity_world_mps`, `displacement_m` and `wheel_rotation_deg`, and its
  `bbox_world_m` is at shutter open. The nominal speeds are typical Euro NCAP test speeds, not
  checked against the protocol text.
- *Not modelled:* limb motion during the exposure. pbrt-v4 animates only rigid transforms, so legs,
  arms and cranks keep the `--vru-gait-phase` pose; at 70 rpm the cranks turn about 4° in 10 ms.
  Also not modelled: wheel rotation on imported models (their wheels are part of one mesh), bicycle
  retroreflectors, and the soft EPT target's fabric texture.

**Imported two-wheelers under the rider.** `--vru-bike <model>` puts the
procedural rider on one of the Objaverse CC-BY models from `--two-wheelers`
(`tools/two_wheelers.py`, fetch IDs listed there): a bicycle for `--vru cyclist`
(default: the procedural bicycle), or a motorbike, moped or scooter for the new
`--vru motorcyclist` (default `moto_80s`; full-face helmet instead of
the cycle helmet), or the kick e-scooter for `--vru escooter` (default
`kickscooter_electric`; a standing rider with one foot ahead of the other on the deck, cycle
helmet, nominal 20 km/h). The Yamaha FZ150 and Solifer meshes lean ~28° and ~17° on their side
stands; under a rider they are stood upright about the travel axis (`two_wheelers.DELEAN`,
roll from a refitted slope of x against height), with the side stand still out. Parked ones keep
leaning. The rider is posed from contact points stored per model in
`two_wheelers.RIDER` (saddle; grips; bottom bracket, footpegs or floorboard),
read off side and top point-cloud profiles of each prepared mesh to a few
centimetres. The torso lean is chosen so the arms reach the grips. The manifest's
`vru.bike` records the model, its nominal length and scale, and `bbox_world_m`
covers rider and model. Limitations: the models' colours are sRGB texture maps,
not measured spectra; with `--exposure-s` the model moves rigidly but its wheels don't turn
(the rider's lamps stay at their shutter-open position); the bars of imported models can't steer (single mesh).

**Steering and lean (riders).** `--vru-turn-radius-m R` (+ = to the rider's left) poses a steady
turn at the target speed: lean φ = atan(v²/(g R)) for a point-mass rider, and the kinematic
steer δ = atan(wheelbase/R), with no trail, caster or tyre slip. For example, 15 km/h on
R = 10 m gives φ = 10.0° and δ = 6.0°. `--vru-lean-deg` and `--vru-steer-deg` set either angle
directly. The lean rolls bike and rider together about the tyre contact line. The steer turns
the procedural bicycle's fork, front wheel and handlebar about the head-tube axis, and the
hands follow the grips; on imported models only the lean applies (`vru.turn.steer_applied`).
The rider's body doesn't counter-lean or shift its weight. With `--exposure-s` the target still
moves in a straight line, because the heading changes by only v·t/R during the exposure. The
manifest's `vru.turn` gives the angles, and `bbox_world_m` includes the lean.

**Two-wheeler lamps** (`--vru-lamps auto|on|off`; `auto` = on at dusk and night for
`cyclist`, `motorcyclist` and `escooter`). A white front lamp and a red rear lamp, both small Lambertian
quads with nominal on-axis intensities: bicycle 1000 cd front (about the 10 lx at 10 m that
StVZO bicycle headlamps must reach) and 4 cd rear, motorbike 1000 cd front and 6 cd rear (as a
car tail lamp). I haven't checked these against StVZO, UN R50 or R113. Positions are estimated
from the contact points (front about 10 cm ahead of and below the grips; rear below and behind the
saddle; on the e-scooter the stem and the back of the deck, with the bicycle intensities), not traced
from the models. They lean with the bike. They light the road only as diffuse emitters (no beam shape or
cut-off), and a crossing target shows them edge-on, so they are mostly visible with `--vru-motion along`.
The manifest gives `vru.lamps` (or `null`). Parked two-wheelers stay unlit.

## Falling rain (`--rain-mmh`; `tools/highway_rain.py`)

    venv/bin/python tools/build_highway_scene.py --rain-mmh 10 --time-of-day night --exposure-s 0.01

Off by default. Places water drops (`dielectric`, η = 1.333) in the camera's view pyramid from
0.5 m out to `--rain-range-m` (default 12 m) and lets each fall at terminal velocity during the
pbrt shutter, so drops render as motion-blurred streaks whose length and faintness follow from
the exposure (Garg & Nayar 2007). Rain needs a shutter: without `--exposure-s` it uses 4 ms. It
also sets `--road-wetness wet` when the road is dry or damp.

* **Drop sizes:** Marshall & Palmer (1948), N(D) = 8000 e^(−ΛD) m⁻³ mm⁻¹, Λ = 4.1 R^−0.21 mm⁻¹;
  only drops above `--rain-dmin-mm` (0.5 mm) are placed (≈ 0.26 M drops at 10 mm/h, 12 m, 31° fov).
* **Fall speed:** Atlas, Srivastava & Sekhon (1973), v = 9.65 − 10.3 e^(−0.6 D) m/s, so a 2 mm drop
  streaks 2.6 cm in 4 ms; `--rain-wind-ms` adds a crosswind drift.
* **Visibility:** the manifest gives the geometric extinction (2 × cross-section) of the full
  distribution and the equivalent visibility (≈ 1.9 km at 10 mm/h). For rain beyond the drop range
  add `--haze mist --visibility-m <that>`.
* **How visible the streaks are:** a real drop is far thinner than a pixel beyond about 1 m
  (a 1.5 mm drop at 7 m covers ~0.3 px width and ~4 % of the shutter time per pixel), so most
  streaks change a pixel by about 1 %, below the render noise at a few hundred spp; at 256 spp they show
  up mostly as fine speckle rather than lines. Only drops within 1–2 m, or a long exposure,
  give clear streaks. Raise `--pixelsamples` to resolve them.
* **Not modelled:** oblate or oscillating drops, splashes on the road (ripples on puddles: see
  `--rain-ripples` under [Wet road](#wet-road-puddles-and-vehicle-spray-tools-highway_wetpy)), rain on the
  windscreen (see `--windscreen-rain`).

## Winter / snow (`--snow`, `--snowfall-mmh`; `tools/highway_snow.py`)

    venv/bin/python tools/build_highway_scene.py --snow slush --snowfall-mmh 1 --sky hosek --sun-elevation 10

Off by default. `--snow verges` covers the verges and distant terrain with snow and adds plough
banks (0.30–0.45 m, between the paved edge and the guard rail; `--no-snowbanks` removes them),
leaving the road clear. `slush` adds grey slush everywhere except two wheel tracks per lane
(lane centre ± 0.85 m, 0.6 m wide, jittered) and sets `--road-wetness wet` so the tracks are wet
asphalt. `packed` covers the whole road with compacted snow, hiding the markings.

* **Reflectance:** spherical albedo of a deep snow layer from asymptotic radiative transfer
  (Kokhanovsky & Zege 2004), r = exp(−√(α · 13 d)), with α from the ice absorption index of
  Warren & Brandt (2008) (`spectra/ice/warren_brandt_2008.csv`, 350–1200 nm) plus an impurity term
  ∝ λ⁻¹. Fresh 0.25 mm verge snow is 0.99 in the visible and drops in the NIR (0.74 at 1030 nm);
  `--snow-grain-mm` and `--snow-dirt clean|light|heavy` change it. Road slush (1.5 mm, heavy dirt)
  is about 0.6 and carries a smooth water film. Impurity levels are nominal. Snow is Lambertian:
  the forward-scattering BRDF of snow and its sub-surface blur are not modelled.
* **Falling snow:** `--snowfall-mmh R` (melted rate) scatters static spherical flakes in the view
  pyramid out to `--snowfall-range-m` (default 15 m), with the Gunn & Marshall (1958) size
  distribution and flake bulk density `--snowflake-density` (0.1 g/cm³). The manifest gives the
  equivalent extinction (2 × geometric cross-section) and visibility (≈ 900 m at 1 mm/h); add
  `--haze mist --visibility-m <that>` for the snowfall beyond the flake range. Flakes are static
  (no motion streaks) and the count grows with range³ (≈ 0.34 M at 1 mm/h, 15 m, 31° fov).
* **Not modelled:** snow on trees, cars and signs; snow-covered markings in `slush`.

## Parking garage / surround-view fisheye (`tools/build_garage_scene.py`)

    venv/bin/python tools/build_garage_scene.py --out-dir scenes/generated/garage --view front
    cd scenes/generated/garage && pbrt garage.pbrt        # writes out/garage_spectral.exr

An enclosed concrete deck (no daylight): a 6.5 m aisle, 16 perpendicular 2.5 × 5.0 m bays a side,
pillars every three bays with a hazard band, painted walls, a 2.6 m ceiling and green exit signs.
`--occupancy` (default 0.7) fills bays with the highway car assets (or `--proxy-cars`), nose to
the wall. The manifest records every bay's floor corners and occupancy (slot-detection ground
truth) and the pillar positions.

* **Camera:** `--view front|rear|left|right` mounts the camera on the ego car's bumper or mirror
  (pitched down 20°, 25°, 45°, 45°). The default is the realistic `config/lenses/fisheye.10mm.dat`
  (155.9° Muller fisheye, no glass sidecar so no dispersion) on a 34 mm-diagonal 4:3 film, so the
  image circle is cropped top and bottom. `--camera pinhole --fov N` is available for quick tests.
* **Lighting:** ceiling rows over the aisle and both bay rows, every 5 m. `--lighting mixed`
  (default) puts LED battens (CIE LED-B4, 4400 lm) over the aisle and twin-T8 fluorescent fittings
  (`--fluorescent-type f2|f11`, 8300 lm) over the bays; `fluorescent` or `led` uses one type
  throughout. `--lights-off-fraction` (default 0.15) leaves fittings dead. Fittings are flat
  Lambertian diffusers (luminance = flux / (π × area)); fluxes are nominal catalogue values.
* **Mains flicker:** `--mains-flicker on` (default) gives each fitting a sinusoidal light-output
  ripple at twice `--mains-hz` (50 or 60). Its depth follows a nominal percent flicker, (max − min)
  / (max + min): 35 % for `--fluorescent-ballast magnetic` (default), 5 % for `electronic`, and
  `--led-percent-flicker` (default 10 %) for the LED battens. The three ceiling rows sit on the
  three supply phases, 120° apart, so they ripple out of step and largely cancel in the ambient
  light; `--mains-phases 1` puts every fitting on one phase, which gives strong rolling-shutter
  bands. The peak luminance is set so the
  time average is the nominal value, so a plain render is unchanged. With `--exposure-s` (and
  `--rolling-line-time-s`, default 15 µs) the manifest records the exposure, and
  `tools/render_time_slices.py --linear` renders the per-row banding a short exposure gets (each
  row is one flicker emitter; the slice mode steps the ripple in 16 levels per period). Real
  ballast and driver waveforms are not sinusoidal, and the percent-flicker values are nominal, not
  measured. `--mains-flicker off` writes the fittings inline, as before.

## Unlit rural road at night (`tools/build_rural_scene.py`)

    venv/bin/python tools/build_rural_scene.py --out-dir scenes/generated/rural
    cd scenes/generated/rural && pbrt rural.pbrt          # writes out/rural_spectral.exr

A two-lane rural road (3.25 m lanes, 0.5 m paved strips, 6 m / 12 m broken centre line) with grass
verges and procedural trees, no street lighting. Night or dusk only (`--time-of-day`, moon via the
`highway_night` options); `--road-wetness` works as on the highway (no puddles/spray).

* **Glare:** an oncoming car at `--oncoming-distance-m` (default 120 m) with `--oncoming-beam high`
  (default); the ego uses `--ego-beam low` (default) or `high`. The high beam is the passing beam
  plus an analytic H-V-centred driving beam, `highway_night.high_beam_cd`, 70 kcd per lamp
  (nominal: ECE R112/R113 need at least 30 kcd on H-V, R48 caps the sum at 430 kcd). The highway
  builder's lamps are unchanged; `write_light_assets(..., high_beam=True)` writes the extra maps.
* **Reflector posts:** 1 m white posts with a black band, every 50 m on both sides, 0.5 m beyond
  the paved edge; right side one 40×180 mm reflector, left side two 60 mm reflectors (German
  Leitpfosten layout). Reflectors use the retroreflective sheeting model (nominal, not measured
  cube-corner data).
* **Deer:** `--deer roe|red|none` (default roe, 0.70 m shoulder height; red 1.20 m) at
  `--deer-distance-m` (default 40 m), `--deer-offset-m`, `--deer-heading-deg` (180 = crossing right-to-left).
  Procedural ellipsoids; analytic brown fur reflectance rising into the NIR; retroreflective eyes
  for tapetum eyeshine. The manifest records the world-space box and eye positions.

## Urban intersection, night and wet (`tools/build_urban_scene.py`)

    venv/bin/python tools/build_urban_scene.py --out-dir scenes/generated/urban [--sign-set stvo] [--vru pedestrian]
    cd scenes/generated/urban && pbrt urban.pbrt          # writes out/urban_spectral.exr

A signalised four-way crossing of two-lane streets (3.5 m lanes, 2.3 m parking lanes, 3 m
sidewalks on 0.15 m kerbs), right-hand traffic, ego camera waiting on the near approach. Defaults:
`--time-of-day night --road-wetness wet`; `--time-of-day day --road-wetness dry` gives the daytime
counterpart. The builder reuses the highway modules rather than duplicating them: spectra, cars and
camera/film options (`build_highway_scene`), night sky, moon, headlamps and retroreflection
(`highway_night`), wet asphalt and markings (`highway_wet`; `--puddles`/`--spray` are highway-only),
sign artwork (`--sign-set`, row placed beyond the junction) and AEB targets (`--vru`, placed on the
near zebra crossing by default).

* **Traffic signals:** five 3-aspect 200 mm heads (two primaries and one secondary for the ego,
  two for the cross arms showing the opposite phase). `--signal-state red|amber|green`. Each lit
  aspect is an LED area light (analytic LED spectra 626/592/505 nm) whose time-averaged luminance
  times lens area equals a nominal 300 cd on-axis intensity (EN 12368 order of magnitude, not a
  measured value). Signals are DC by default; `--signal-pwm-hz/--signal-duty` make them PWM.
* **Shop LED signs:** `--shop-signs` (default 4) red/blue/green panels on the ground floors,
  400 cd/m² time-averaged, PWM at `--shop-pwm-hz 100 --shop-duty 0.3` (nominal).
* **Facades:** seeded 3–6 storey lots with window grids; at dusk/night `--lit-windows` (default
  0.35) of them are 15 cd/m² 3000 K diffuse emitters.
* **Street lights:** single-arm 8 m columns on both kerbs, 25 m apart, staggered, with the same
  full-cutoff luminaire model as the highway (`--streetlights led_4000k|hps|none`).
* **Cars:** parked cars along both kerbs (unlit), one oncoming and one waiting on the cross street
  (both with headlamps at night).
* **Flicker:** every signal aspect and shop sign is in `manifest["emitters"]`, so
  `tools/render_time_slices.py scenes/generated/urban --linear` renders LED flicker / rolling
  shutter (pass `--exposure-s` to record the exposure in the manifest).
* **Imported signal heads (opt-in):** `--signal-model nyc80s|backplate12` replaces the procedural
  black housing with an Objaverse CC-BY mesh (`signal_nyc80s`: 1980s New York-style 3-aspect head;
  `signal_12in_backplate`: 12-inch head with a yellow backplate). The head is scaled so its aspect
  pitch is 0.3 m, and its lenses sit 1 cm behind the procedural emitter positions. The lit aspect stays
  the same LED emitter (luminance, spectrum, PWM), as a 200 mm disc with the same area as the
  procedural square, so `manifest["emitters"]` and the photometry are unchanged. Unlit aspects show
  the mesh's own textured lenses. Default `--signal-model procedural` is the original housing.
* **Street furniture (opt-in):** `--street-furniture pavement` puts seeded Poly Haven CC0 props on both
  pavements: fire hydrants and bins at the kerb, benches, utility cabinets and a drum against the
  facades, kept 1.5 m clear of street-light columns and signal poles. `--work-zone cones|drums` closes
  the near-right parking lane between z = 8 and 20 m with a taper of Objaverse CC-BY cones (0.75 m,
  nominal) or channelizer drums (1.0 m, nominal), plus concrete barriers. Parked cars there are
  removed. Placements use their own RNG, so the rest of the seeded scene is unchanged, and
  `manifest["street_furniture"]["items"]` lists each item's asset, position, yaw and world box.
  Fetch the assets with `tools/fetch_highway_assets.py` (ids `prop_*`, `cone_*`, `drum_*`,
  `signal_*`). Limitations: the props use their sRGB base-colour textures, not measured spectra.
  At dusk/night the white cone and drum bands are retroreflective Type III sheeting (see Night and dusk).
* **Parked two-wheelers (opt-in):** `--two-wheelers parked` adds 21 Objaverse CC-BY models
  (`tools/two_wheelers.py`): groups of 2-4 bicycles across the pavement next to the kerb, the odd
  kick e-scooter parked along the kerb, and 2-3 motorbikes, mopeds or scooters in each empty parking
  bay, angled 30 deg towards the traffic. Bicycle groups keep 1.2 m clear of street-light columns,
  signal poles and street furniture. Each model is scaled to a nominal overall length for its type
  (bicycles 1.6-1.85 m, motorbikes 2.0-2.3 m, scooters 1.6-1.8 m, kick e-scooter 1.08 m; not
  checked against data sheets) and turned so its front faces the given heading. Placements use
  their own RNG, so the rest of the seeded scene is unchanged, and `manifest["two_wheelers"]["items"]`
  lists each model's asset, kind, position, heading and world box. Fetch the assets with
  `tools/fetch_highway_assets.py` (ids `bike_*`, `moto_*`, `moped_*`, `scooter_*`,
  `kickscooter_electric`). Limitations: sRGB base-colour textures, not measured spectra; no lamps,
  and their reflectors are diffuse, not retroreflective (the meshes have no separate reflector parts); no bike stands; nobody rides them.
  The Poly Haven hydrant and bin files contain two variants side by side, which are split into
  separate objects.

### Facade kits (`--facade-style kits`)

Off by default (`procedural`). `kits` assembles each street lot's front from one
of two Poly Haven CC0 modular kits by James Ray Cock: *Modular Urban Apartments
Facade* (plaster) and *Modular Factory Facade* (brick). Fetch them with

```bash
python tools/fetch_highway_assets.py kit_urban_apartments kit_factory
```

Both kits use a 3 m grid. `tools/facade_kits.py` pairs each flat 3 m wall panel
with its insert (`wall_window_<X>` + `window_<X>`, `wall_door_<X>` + `door_<X>`;
angled bays, corners and garage doors are skipped). It fills each lot with
`round(width / 3)` bays, stretched by up to ±17 % to fit, and 3 m storeys: a
door bay and ground-floor windows on the ground floor, one window type on the
upper floors, a cornice between storeys, a crown on top and piers at the lot
ends. Every part is a pbrt object instance. The lot box moves 0.4 m behind the
wall plane. A room-back quad 0.35 m behind each opening uses the scene's
`window_dark` / `window_lit` materials, so `--lit-windows` still lights rooms
at night. Kit choices draw from their own seeded RNG; cars and emitters match
the default scene. The manifest's `facades.kits` lists each lot's kit and parts.

Limitations: the kits' colours come from their sRGB texture maps (diffuse), not
measured spectra, and the window frames' metallic flag is ignored. Glass is a
thin dielectric. Kit buildings are lower than the procedural ones (3 m storeys
against 3.2 m plus 0.8 m), and the cross-street lots stay procedural.
`fetch_highway_assets.py` now records each glTF shape's `node` name and
layout `origin` in `prepared.json` (re-prepare older kits with `--force`).

### Facade textures (`--facade-textures mixed`)

Off by default (`none`, the plain spectral boxes). `mixed` puts one of eight
ambientCG CC0 maps on each street lot: red, factory-red and orange brick
(`facade_brick` spectrum), light-painted and beige brick, white and yellow
render (`facade_render`), and smooth concrete (`concrete`). Fetch them with

```bash
python tools/fetch_highway_assets.py tex_bricks059 tex_bricks085 tex_bricks092 tex_bricks060 \
    tex_bricks093 tex_concrete034 tex_paintedplaster017 tex_paintedplaster004
```

As for the road surfaces, each wall's reflectance is still its spectrum; the
colour map only contributes its luminance, normalised to mean 1, plus a normal
map. So brick and render keep the same mean brightness as the untextured
boxes, but the yellow render and red brick are not tinted by their photo
colours. UVs are metric (the manifest's `tile_size_m`; the two plaster maps
have no published size and use a nominal 2 m). Lots draw their map from their
own seeded RNG, so cars, windows and emitters match the default scene. The
procedural windows and lit windows are unchanged; the manifest's
`facades.textures` lists the map used on each lot. Cross-street lots stay
untextured.
