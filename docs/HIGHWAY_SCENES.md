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
headlamp light); sign/marking retroreflection is isotropic in azimuth and has no wet-road
or dew behaviour.
`--haze` cannot be combined with dusk/night yet: the night reference illuminance has no
medium model, so the builder refuses the combination rather than write a wrong manifest.

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
- The sky radiance is RGB, upsampled by pbrt; only the sun is spectral.
- No wet road; texture colour maps only modulate luminance.
- Haze: one HG phase function per medium (Rayleigh folded into g), horizontally uniform
  medium, flat Earth for the medium (the backdrop has curvature); without `--haze` the horizon
  is sharp and aerial perspective comes only from the sky map.
- Asset car materials other than paint/glass/tyres are the authors' RGB values.
