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
and `no_medium_horizontal_illuminance_lux`). `TestPbrtHighwayHaze` checks it against pbrt: an
infinite Lambertian ground probe rendered with `volpath` under fog and under no medium gives the
same illuminance ratio as the manifest, within 5 %. With `--haze-light-reference road` the lights
are instead rescaled so the road receives the no-medium illuminance (useful to isolate the
contrast loss of fog from the change in exposure); the manifest then records the scale applied.

**Distant terrain.** The backdrop is a polar heightfield centred on the ego position: ridges
specified by the elevation angle they subtend (peaks ~0.8 deg at 3 km to ~2.2 deg at 30 km, a
shallow valley along the road), minus the Earth-curvature drop r^2/(2 R_eff), R_eff = 7/6 R_earth
(standard refraction). Cover is an `fbm` mix of the scene's grass and a closed forest canopy
reflectance (visible 0.02-0.06, red edge to ~0.3; Moody et al. 2005, ECOSTRESS spectral library).
It is procedural, so no new assets are fetched; `--seed` changes the skyline.

**Render cost** (1280x720, 256 spp, 8 threads, same view; see the PR for images):

RENDER_COST_TABLE

## Known approximations

- No retroreflective BSDF in pbrt: markings use a glass-bead-like rough clear coat over a
  spectral binder; sign sheeting is diffuse. Night/headlight retroreflection is not modelled.
- The sky radiance is RGB, upsampled by pbrt; only the sun is spectral.
- No wet road or road curvature; texture colour maps only modulate luminance.
- Haze: one HG phase function per medium (Rayleigh folded into g), horizontally uniform
  medium, flat Earth for the medium (the backdrop has curvature); without `--haze` the horizon
  is sharp and aerial perspective comes only from the sky map.
- Asset car materials other than paint/glass/tyres are the authors' RGB values.
