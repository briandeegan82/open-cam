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
- **Wear maps** (one 97.5 m period, 2 cm texels, pbrt `planar` mapping; `--road-wear-texel-m`):
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
- **Raised pavement markers**: 100 x 100 x 18 mm geometry instances (ASTM D4280 height limit
  ~20 mm). They sit in every other broken-line gap (2N = 24.4 m, MUTCD 3B.11-3B.14), with
  clear/red lenses on lane lines and amber on the yellow edge, and about 10% are missing. Lenses
  are smooth coated diffuse (daytime look only; retroreflection is not modelled).
- **Normal maps**: pbrt-v4 decodes 8-bit JPEGs as sRGB even when a linear encoding is requested,
  which tilted the asphalt normals and rendered the road ~7x too dark. The asphalt and grass
  normal maps (with or without wear) are therefore converted to PNG
  (`highway_road_wear.linear_normal_map`).

## Known approximations

- No retroreflective BSDF in pbrt: markings use a glass-bead-like rough clear coat over a
  spectral binder; sign sheeting is diffuse. Night/headlight retroreflection is not modelled.
- The sky radiance is RGB, upsampled by pbrt; only the sun is spectral.
- No participating medium (haze), wet road, or road curvature; texture colour maps only modulate
  luminance.
- Asset car materials other than paint/glass/tyres are the authors' RGB values.
