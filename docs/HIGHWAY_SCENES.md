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

## Known approximations

- No retroreflective BSDF in pbrt: markings use a glass-bead-like rough clear coat over a
  spectral binder; sign sheeting is diffuse. Night/headlight retroreflection is not modelled.
- The sky radiance is RGB, upsampled by pbrt; only the sun is spectral.
- No wet road or road curvature; texture colour maps only modulate luminance.
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
