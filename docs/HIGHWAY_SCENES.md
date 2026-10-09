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

- No retroreflective BSDF in pbrt: markings use a glass-bead-like rough clear coat over a
  spectral binder; sign sheeting is diffuse. Night/headlight retroreflection is not modelled.
- By default the sky radiance is RGB, upsampled by pbrt; only the sun is spectral (see
  [Spectral sky](#spectral-sky) for the opt-in spectral sky).
- No wet road; texture colour maps only modulate luminance.
- Haze: one HG phase function per medium (Rayleigh folded into g), horizontally uniform
  medium, flat Earth for the medium (the backdrop has curvature); without `--haze` the horizon
  is sharp and aerial perspective comes only from the sky map.
- Asset car materials other than paint/glass/tyres are the authors' RGB values.
