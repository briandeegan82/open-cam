# Highway Scenes

`tools/build_highway_scene.py` generates parameterized autonomous-driving
highway scenes — multi-lane road with MUTCD-style markings, guard rails, road
signs, cars, spectral sun and a Hosek-Wilkie sky — that flow through the
standard pipeline: `pbrt --gpu` spectral render →
`pbrt_spectral_exr_to_electrons.py` → `apply_emva_noise.py`.

---

## Quick start

```bash
# one-time: fetch CC0 car meshes (optional; procedural fallback exists)
venv/bin/python tools/fetch_highway_assets.py

# fast end-to-end smoke run (480x320, 128 spp)
scripts/generate_highway.sh demo --preset smoke

# full-quality scene with custom traffic and low sun
scripts/generate_highway.sh rush \
  --num-cars 8 --seed 3 --sun-elevation 25 --sun-azimuth 210 --turbidity 4
```

Outputs: scene + manifest under `scenes/generated/highway/<name>/`, spectral
EXR under `out/`, electrons NPZ + RAW16 + preview PNGs via the standard noise
stage.

---

## What is spectral (and what is not)

| Surface | Definition | Notes |
|---|---|---|
| Asphalt, lane paint, grass, tires, sign sheeting | measured-style SPD curves in `spectra/surfaces/highway/` | analytic approximations of published reflectances |
| Car paint | SPD base + `coateddiffuse` clearcoat | six paints: white/black/silver/gray/red/blue |
| Guard rails, posts, trim | pbrt built-in `metal-Al` / `metal-Ag` spectra | |
| Sun | `spectra/illuminant/interpolated/solar_direct_am15.csv` (ASTM G-173 shape) | `--illuminant` swaps in D65 etc. |
| Sky dome | Hosek-Wilkie EXR (RGB, upsampled by pbrt) | diffuse fill only — see below |
| Sign legends | RGB PNG texture | the **only** RGB surface; `--spectral-signs` swaps to plain SPD panels |

**Retroreflection is not modeled** (pbrt has no retroreflective BRDF); sign
sheeting and lane paint are bright diffuse. Manifests record
`"retroreflection": "approximated_diffuse"`. Fine for daytime sun-lit scenes;
do not use for nighttime headlight studies without adding a proper BRDF.

## Sun / sky model

- The **distant sun** carries the full spectral SPD and is the only direct
  sun: the builder post-processes the makesky EXR to **excise the baked solar
  disk** (otherwise the sun would be double-counted) and to fill the lower
  hemisphere with horizon color (removes an octahedral-seam sawtooth artifact
  at the horizon). The processed dome (`*_diffuse.exr`) is cached next to the
  raw one under `scenes/assets/sky/`.
- Sun position: `--sun-elevation` (deg above horizon) and `--sun-azimuth`
  (0 = behind camera, 90 = camera's right, measured clockwise). The dome is
  rotated so its glow matches the requested azimuth (the baked sun azimuth is
  detected from the EXR itself, so makesky conventions can't drift).
- Intensity balance: distant-light scale is set so direct-horizontal :
  diffuse-horizontal illuminance = `--sun-sky-ratio` (default 5.0, clear-sky).
  Absolute level is irrelevant — the electrons stage renormalizes.

## Radiometric calibration (important)

Chart-style lux normalization (`calibration.illuminant_override_csv`) assumes
a uniformly-lit flat target and **does not apply** to a 3D scene with sky, sun
and shadows. Highway runs must use EXR autocalibration:

```yaml
model:
  pbrt_spectral_exr:
    radiometric_autocalibration: mean_photopic_lux
```

`config/camera_models/research_highway_wide22.yaml` is a committed derivative
of `research_realistic_wide22` with this enabled; `generate_highway.sh` uses
it by default. Three calibration entries in that model matter and must stay
consistent (see the YAML comments): `irradiance_scale_W_m2nm_per_unit: 1.0`
(the chart-mode unit factor must not rescale autocal output), **no**
`illuminant_override_csv` (it would apply the lux normalization twice), and
`focal_length_mm: 22` with the builder's 8 mm aperture (correct f/2.75
radiometry).

`--target-illuminance-lux` sets the **mean sensor-plane photopic lux**
(E = πL/(4N²)·τ, not scene illuminance).

**Exposure vs. full well:** the research sensor's full well is 4 800 e⁻. The
`TARGET_LUX=125` default (with 0.005 s exposure) puts the road at ~30% of full
well with a mostly unclipped blue sky under the default 5:1 sun-dominant
illumination — a well-balanced daylight frame. Raise it 2–4× when the scene
mean is dominated by an in-frame sun (`--visible-sun-disk` glare runs) or for
overcast (`--sun-sky-ratio` lowered), since autocal normalizes the *mean* and
bright outliers eat the budget.

## Car assets

`tools/fetch_highway_assets.py` downloads the CC0 Kenney Car Kit
(sha256-pinned in `config/highway_assets.yaml`; if the pinned URL rots the
fetcher re-scrapes the asset page), splits each OBJ into semantic slots by
group/material globs (wheels → tire SPD, body → paint SPD) and writes binary
PLYs under `scenes/assets/cars/<model>/` (gitignored). Available models:
sedan, sedan_sports, hatchback, suv, suv_luxury, van, truck, delivery, taxi,
police.

- Kenney bodies are a single group, so windows share the paint SPD (the
  procedural fallback car has separate glass). Stylized proportions: scaled to
  real length they are wider than real vehicles.
- Offline: `--local-zip path/to/kenney_car-kit.zip`; or skip entirely —
  `--procedural-cars` (or absent assets) uses parameterized box cars.
- Explicit traffic: `--cars "sedan:1:35:red,truck:0:80:white"`
  (`model:lane:distance_m:paint`, lane 0 = rightmost); random:
  `--num-cars N --seed S`.

## Scene parameters (selection)

```
--lanes 3 --lane-width 3.7 --road-length 400 --no-guardrails
--signs "speed_limit:120:65,exit:300:12"     # type:distance[:text]
--sun-elevation 45 --sun-azimuth 135 --turbidity 3 --sun-sky-ratio 5
--camera realistic --lensfile config/lenses/wide_22mm.dat --aperture-diameter-mm 8
--cam-height 1.4 --cam-lane 0 --look-distance 60 --focus-distance 20
--xres 960 --yres 640 --pixelsamples 4096 --spectral-nbuckets 64
--preset smoke        # 480x320 / 128 spp / 200 m road / opaque glass
--opaque-glass        # coateddiffuse glass (less noise than dielectric)
```

Everything is recorded in the manifest (`highway_<name>_manifest.json`),
including car placements, sun/sky scales and the calibration recommendation —
enough to regenerate or post-process downstream datasets.

## Low-sun / high-glare scenarios

```bash
GLARE=1 scripts/generate_highway.sh sunset \
  --visible-sun-disk --sun-elevation 6 --sun-azimuth 183 --turbidity 5 --num-cars 4
```

Two pieces combine:

- `--visible-sun-disk` replaces the delta distant light with an emissive disk
  of the true 0.53° angular diameter delivering the same irradiance (`twosided`
  area light at 1 km). The sun becomes visible in-frame and shadows gain the
  correct ~0.5° penumbra. Rendered disk:sky radiance contrast lands at
  ~5×10⁵ — the right order for the real sun.
- `GLARE=1` runs the lens stray-light stage (`tools/apply_spectral_psf.py`)
  on the spectral EXR before the sensor: veiling glare, bloom halo,
  180°-rotated ghost, and N-blade diffraction starburst. Parameters live under
  `lens.post_psf.stray_light` in the highway camera model
  (`sigma_pixels: 0` — no extra blur on top of the traced lens). The stage
  writes `out/highway_<name>_spectral_glare.exr`, keeping the clean EXR.

With mean-lux autocal, a blinding sun in-frame depresses the rest of the scene
(the sun eats the lux budget) — realistic mean-metering auto-exposure
behavior. At the default `TARGET_LUX=125` the road sits at ~1% of full well;
use `TARGET_LUX=1000–2000` for a glare run to bias exposure toward the road
(~10–25% full well) while the sun region saturates.

### Sun/sky photometric calibration (`SKY_DOME_PHOTOMETRIC_K`)

pbrt upsamples the RGB sky dome to illuminant spectra whose photopic scale
differs from the SPD-based sun by a large constant. `build_highway_scene.py`
corrects with `SKY_DOME_PHOTOMETRIC_K = 78000` (empirical, ±8% across sky
chromaticity), making the rendered direct:diffuse ratio match
`--sun-sky-ratio` (verified 5.02 vs 5.0 target). To re-derive after a pbrt
upgrade: build a scene with sky (note the manifest's `lighting.sun.scale`),
build sun-only (`--no-sky --sun-scale <that value>`) and sky-only
(`--sun-scale 1e-12`) variants, render all three, and compare mean road
radiance — `K_new = K_old × target_ratio / rendered_ratio`.

## Performance / GPU notes

- Everything the builder emits is GPU-path compatible (basic textures only,
  GPU-supported materials, plymesh + object instancing). See
  [GPU vs CPU Rendering](gpu_vs_cpu_rendering.md).
- Budget: full scene ≪ 500 MB GPU memory; 960×640 @ 64 buckets is ~160 MB of
  film. spp guidance matches the pipeline table: 512 iterate / 4096 final.
- Dielectric car glass adds fireflies at low spp — smoke preset forces
  `--opaque-glass`.

## Known approximations

- Retroreflection → bright diffuse (manifest-recorded).
- Sky dome is RGB-upsampled (sun is spectral; dome is fill).
- Kenney car windows share body paint; procedural cars have glass.
- No atmospheric scattering/fog medium, no road wetness, flat terrain.
