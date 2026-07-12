# Lens Models

This guide covers the lens model system: PBRT realistic lens prescriptions, the four built-in lens models, per-class assignment rules, and how to use the assignment tool.

---

## Background

Every camera recipe requires an optics model. Prior to the current lens model system, all cameras — regardless of sensor class — used the same 22mm prescription at a fixed 4mm aperture. A Canon EOS 20D (6.4 µm pitch, APS-C) and an iPhone 8 (1.4 µm pitch, phone) were rendered through identical optics, which is physically inconsistent.

The current system provides class-appropriate lens models for all 60+ production camera recipes, covering four distinct focal length and aperture classes.

---

## Lens Prescription Files

Lens `.dat` files live in `config/lenses/`. They use PBRT's `ReadFloatFile` lens format: four whitespace-separated columns per surface in millimetres.

### File Format

```
# radius   thickness   ior   semi-aperture
  35.0      5.5        1.65    16.0
   0.0      6.2        0.0     15.0    # n=0 marks aperture stop
 -82.5      3.5        1.52    15.0
  ...
```

- **radius**: radius of curvature [mm]. Positive = centre of curvature to the right. 0 = flat surface (used for aperture stop)
- **thickness**: axial spacing to next surface [mm]
- **ior**: index of refraction of the medium to the right. 0 = air gap / aperture
- **semi-aperture**: clear aperture half-diameter [mm]

### Available Prescription Files

| File | Focal length | Design | Source |
|------|-------------|--------|--------|
| `wide_22mm.dat` | 22 mm, ~38° HFOV | Nakamura wide angle | MLD p. 360, scaled |
| `dgauss.50mm.dat` | 50 mm, f/2 | Double-Gauss (Tronnier) | MLD p. 312 / US patent 2,673,491 |
| `telephoto.250mm.dat` | 250 mm EFL, f/5.6 | Sigler super-achromate | MLD p. 175, scaled |
| `fisheye.10mm.dat` | 10 mm, ~156° FOV | Muller 16mm/f4 fisheye | MLD p. 164, scaled |

All prescriptions were retrieved from `mmp/pbrt-v4-scenes` (MIT licence) or scaled from *Modern Lens Design* (MLD) by Warren J. Smith.

---

## Lens Model YAML Files

Each lens model YAML (`config/lens_models/<name>.yaml`) specifies the PBRT realistic camera parameters. They deep-merge on top of `config/lens_models/default.yaml`.

### Schema

```yaml
lens:
  camera: realistic
  realistic_lensfile: config/lenses/wide_22mm.dat
  realistic_aperture_diameter_mm: 12.2    # sets effective f-number
  realistic_focus_distance: 1.0           # metres
  post_psf:
    sigma_pixels: 0.5                     # geometric residual blur
```

### Built-in Lens Models

| Model YAML | Prescription | Aperture [mm] | Effective f/ | Target class |
|-----------|-------------|--------------|-------------|-------------|
| `phone_wide_f18.yaml` | `wide_22mm.dat` | 12.2 | f/1.8 | Smartphone (< 1.5 µm) |
| `compact_wide_f28.yaml` | `wide_22mm.dat` | 7.9 | f/2.8 | Compact/bridge (1.5–3.0 µm) |
| `normal_dgauss_50mm_f28.yaml` | `dgauss.50mm.dat` | 17.9 | f/2.8 | APS-C / Micro Four Thirds (3.0–4.0 µm) |
| `normal_dgauss_50mm_f20.yaml` | `dgauss.50mm.dat` | 25.0 | f/2.0 | Full-frame / large-pitch (≥ 4.0 µm) |

The `telephoto.250mm.dat` and `fisheye.10mm.dat` prescriptions are available for research configs but are not automatically assigned to production recipes.

---

## Automatic Lens Assignment

`tools/assign_lens_models.py` assigns the appropriate lens model to every camera recipe based on sensor class or pixel pitch.

### Usage

```bash
# Preview changes without writing:
python tools/assign_lens_models.py --dry-run

# Apply assignments:
python tools/assign_lens_models.py
```

### Assignment Rules (highest priority first)

1. **`source.sensor_class`** in the sensor model YAML:

   | Sensor class | Assigned lens model |
   |-------------|-------------------|
   | `phone` | `phone_wide_f18` |
   | `compact` | `compact_wide_f28` |
   | `mft` | `normal_dgauss_50mm_f28` |
   | `apsc` | `normal_dgauss_50mm_f28` |
   | `fullframe` | `normal_dgauss_50mm_f20` |
   | `old` | `normal_dgauss_50mm_f20` |

2. **`sensor.pixel_pitch_um`** fallback (when no `sensor_class`):

   | Pixel pitch | Assigned lens model |
   |------------|-------------------|
   | < 1.5 µm | `phone_wide_f18` |
   | 1.5–3.0 µm | `compact_wide_f28` |
   | 3.0–4.0 µm | `normal_dgauss_50mm_f28` |
   | ≥ 4.0 µm | `normal_dgauss_50mm_f20` |

Run `assign_lens_models.py` after adding new camera recipes or modifying sensor pixel pitch values.

---

## Manual Lens Override

Override the lens model for a specific run in `pipeline.yaml`:

```yaml
lens_type_override: realistic
lens_overrides:
  realistic_lensfile: config/lenses/telephoto.250mm.dat
  realistic_aperture_diameter_mm: 44.6    # 250 mm / f5.6
  realistic_focus_distance: 10.0
```

Or override within a camera recipe YAML:

```yaml
lens_model: normal_dgauss_50mm_f28
lens:
  realistic_focus_distance: 2.0     # closer focus for this recipe
```

---

## Adding a New Lens Prescription

1. Place the `.dat` file in `config/lenses/`
2. Create a lens model YAML in `config/lens_models/<name>.yaml`:

```yaml
lens:
  camera: realistic
  realistic_lensfile: config/lenses/my_new_lens.dat
  realistic_aperture_diameter_mm: 20.0
  realistic_focus_distance: 1.0
```

3. Reference the lens model from a camera recipe:

```yaml
# config/camera_recipes/my_camera.yaml
lens_model: my_new_lens_name
```

### Testing the Lens

Render a pinhole scene first to verify the scene geometry, then switch to the realistic lens:

```yaml
lens_type_override: pinhole   # quick sanity check
# then:
lens_type_override: realistic
```

---

## Focus Distance and Scene Units

PBRT scene units are metres. The `realistic_focus_distance` key sets the distance at which the lens is focused. For the default ColorChecker scene at 4m camera distance:

- `realistic_focus_distance: 4.0` — chart is in focus
- `realistic_focus_distance: 1.0` — chart is out of focus (blurred by depth of field)

The pipeline's `lens_overrides.realistic_focus_distance` in `pipeline.yaml` overrides the camera model value at run time (without editing the camera model file).

---

## Lens Model Physics in PBRT

When `lens.camera: realistic` is set, PBRT:
1. Loads the lens prescription and builds an internal surface list
2. For each camera ray, traces through all lens surfaces using Snell's law
3. Handles total internal reflection and vignetting naturally from geometry
4. Computes the exact exit pupil and samples it for each pixel

This produces physically accurate:
- Bokeh (out-of-focus blur shape from lens design)
- Vignetting (light fall-off at image edges from barrel geometry)
- Geometric distortion (barrel/pincushion from focal length variation)
- Chromatic aberration (if the prescription includes dispersion — current `.dat` files use fixed IOR)

---

## See Also

- [Camera Models](camera_models.md) — `lens` section in camera model YAML
- [Optics & PSF Theory](../theory/04_optics.md) — diffraction and Gaussian PSF
- `tools/assign_lens_models.py` — automated lens assignment
- [PBRT lens format](https://pbrt.org/fileformat-v4.html#camera-types) — official PBRT realistic camera docs
