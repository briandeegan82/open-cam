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

## Known approximations

- No retroreflective BSDF in pbrt: markings use a glass-bead-like rough clear coat over a
  spectral binder; sign sheeting is diffuse. Night/headlight retroreflection is not modelled.
- The sky radiance is RGB, upsampled by pbrt; only the sun is spectral.
- No participating medium (haze), wet road, or road curvature; texture colour maps only modulate
  luminance.
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
