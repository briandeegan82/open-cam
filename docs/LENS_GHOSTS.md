# Traced lens ghosts (`tools/lens_ghosts.py`)

Opt-in, physically traced two-reflection lens ghosts ("flare ghosts") derived from the sequential
lens prescription in `config/lenses/*.dat`. Off by default; when off, no pipeline stage is added and
outputs are byte-identical (regression tests in `tests/test_lens_ghosts.py`).

## Model

1. **Prescription.** `load_lens()` reads the pbrt `RealisticCamera` format (`radius thickness eta
   aperture`, mm, `radius == 0` = aperture stop; `tools/lens_prescription.py`), places the film at
   the pbrt focus position for `focus_distance_m`, and stops the iris to `aperture_diameter_mm`.
2. **Ghost paths.** For K refracting surfaces every pair `i < j` gives one two-reflection path:
   transmit to `j`, reflect, travel back to `i`, reflect, transmit to the film (Hullin et al. 2011).
   `wide_22mm.dat` has 12 refracting surfaces, so 66 paths. Optional `sensor_reflectance` adds the
   12 paths sensor -> surface `i` -> sensor.
3. **Coatings.** `tools/lens_coatings.py` computes the spectral, angle-dependent, unpolarised (s/p
   average) reflectance of every air-glass interface with the thin-film characteristic matrix
   (Born & Wolf sec. 1.6; Macleod ch. 2): `uncoated` (Fresnel), `mgf2` (quarter-wave MgF2,
   n = 1.38, Dodge 1984), `qhq` (quarter-half-quarter broadband AR, Lockhart & King 1947), or a
   custom layer list. Cemented glass-glass interfaces are left uncoated (Fresnel).
4. **Trace.** A collimated grid of `pupil_samples^2` rays per source direction is traced exactly
   (sphere intersection, vector Snell's law, aperture and polygonal-iris clipping at every surface)
   along every path. Each ray carries the product of the interface R/T factors at its actual angle
   of incidence, per wavelength bucket. Its film position gives the ghost footprint, shape and
   position vs field angle.
5. **Splatting.** Each grid cell's energy is spread over the quad spanned by its four traced corners
   (Hullin et al. 2011), so footprint density (defocus, caustics) is energy-conserving.
6. **Sources.** Pixels of the spectral EXR above `source_threshold_relative x max` (or
   `source_threshold_abs`) are grouped into `cluster_px` blocks (up to `max_sources`). Each block is a
   collimated source at the field angle of its pixel (inverse primary trace for `camera: realistic`,
   pinhole mapping otherwise). Its ghost image is its per-bucket spectral flux times the path's
   ghost/primary ratio, added to every `S0.*` channel.
7. **Pipeline position.** `tools/run_pipeline.py` runs the stage on the spectral EXR after pbrt and
   the optional post-PSF but *before* the radiance -> electrons conversion, so the sensor QE weights
   the ghost spectrum correctly. It runs for realistic cameras too, because pbrt traces no
   inter-reflections. The parametric veiling glare/halo/ghost/starburst model in
   `tools/apply_spectral_psf.py` (`post_psf.stray_light`) is unchanged and can be combined with this
   stage for the diffuse scatter that this model leaves out.

## Usage

Recipe: `config/camera_recipes/research_realistic_wide22_traced_ghosts.yaml` (lens model
`config/lens_models/research_realistic_wide22_traced_ghosts.yaml`, `lens.traced_ghosts` block).

```yaml
lens:
  traced_ghosts:
    enabled: true
    coating: mgf2            # uncoated | mgf2 | qhq | {layers: [[n, thickness_nm], ...]}
    coating_design_wavelength_nm: 550.0
    surface_coatings: {}     # per-surface override, e.g. {0: uncoated}
    film_diagonal_mm: 35.0   # pbrt realistic default
    iris_blades: 0           # 0 = circular, >= 3 = regular polygon
    sensor_reflectance: 0.0  # e.g. 0.02 to include sensor <-> lens ghosts
    pupil_samples: 64        # 128 for publication figures (sharper vignetting edges, ~3x slower)
    source_threshold_relative: 0.01
```

Standalone CLI on any pbrt spectral EXR:

```bash
venv/bin/python tools/build_straylight_test_scene.py --camera realistic \
  --lensfile config/lenses/wide_22mm.dat --aperture-diameter-mm 8 --focus-distance 3 \
  --film spectral --spectral-nbuckets 16 --spectral-lambda-min 400 --spectral-lambda-max 700 \
  --xres 640 --yres 400 --pixelsamples 64 \
  --out-scene scenes/generated/straylight_ghosts.pbrt --film-output out/ghosts/straylight_wide22.exr
third_party/pbrt-v4/build/pbrt scenes/generated/straylight_ghosts.pbrt
for c in uncoated mgf2 qhq; do
  venv/bin/python tools/lens_ghosts.py --enable --exr-in out/ghosts/straylight_wide22.exr \
    --exr-out out/ghosts/straylight_wide22_ghosts_$c.exr --lens-file config/lenses/wide_22mm.dat \
    --camera realistic --aperture-diameter-mm 8 --focus-distance-m 3 --film-diagonal-mm 35 \
    --coating $c --report-json out/ghosts/report_$c.json
done
PYTHONPATH=tools venv/bin/python tools/plot_lens_ghost_validation.py \
  --base-exr out/ghosts/straylight_wide22.exr \
  --ghost-exr-pattern "out/ghosts/straylight_wide22_ghosts_{coating}.exr" \
  --out-png out/ghosts/lens_ghost_validation.png --out-json out/ghosts/validation_metrics.json
```

The report JSON lists the sources, the total ghost/source energy, the fraction landing in frame,
and the brightest paths (surface pairs).

## Validation (`tests/test_lens_ghosts.py`)

- Thin films: R + T = 1 for lossless stacks; quarter-wave minimum
  `R = ((n_s - n_c^2) / (n_s + n_c^2))^2`; Brewster zero for p; TIR; s/p agree at normal incidence.
- Single surface pair, analytic: plane-parallel plate. Ghost energy `T^2 R^2` and lateral shift
  `2 t tan(theta_t)` match closed forms.
- Ghost position vs source angle: the vectorised tracer is checked against an independent scalar
  meridional ray trace for several paths and angles, and against the paraxial (ABCD) ghost
  matrices at small angles.
- Energy: primary + all ghosts <= 1 per wavelength. Two-reflection energies match a non-sequential
  Monte Carlo trace (random R/T at every interface) within its statistical error. The rasteriser
  conserves energy.
- Iris: hexagon/circle footprint area ratio `3 sqrt(3) / (2 pi)`.
- Opt-in: defaults disabled, pipeline dry-run adds no stage when off, EXR unchanged when disabled,
  parametric `apply_stray_light` output unchanged.

Numbers from `wide_22mm.dat` (EFL 22.0 mm, 8 mm stop, focus 3 m), `tools/plot_lens_ghost_validation.py`:

| | uncoated | MgF2 | QHQ |
|---|---|---|---|
| R per air-glass surface, 550 nm, normal incidence (n = 1.62) | 5.60 % | 0.65 % | 0 (design point) |
| mean R, 420-680 nm | 5.60 % | 0.91 % | 0.25 % |
| sum of 66 ghosts / primary, 550 nm, 10 deg field | 4.3e-2 | 1.0e-3 | 1.7e-4 |
| primary transmittance, 550 nm, 10 deg field | 0.536 | 0.936 | 0.994 |
| stray-light scene: ghost energy / image energy | 2.9e-2 | 1.4e-3 | 4.6e-4 |

Ghost chief-ray film height, exact trace vs paraxial matrices, field angle <= 3 deg: max deviation
1.3 um (6 brightest paths).

Validation figure (`tools/plot_lens_ghost_validation.py`): coating spectra, ghost position vs angle
(exact vs paraxial), total ghost energy vs angle, and the stray-light scene with uncoated / MgF2 /
QHQ coatings.

## Limitations (not modelled)

- Geometric optics only. No diffraction from the iris on the ghosts, and no coherent interference
  between ghosts (interference *inside* coatings is modelled).
- Constant glass index per element, i.e. no dispersion. pbrt's `RealisticCamera` is also
  non-dispersive. No bulk absorption.
- Two reflections only. Four-reflection paths are about R^2 weaker (<1e-3 of two-reflection paths
  with AR coatings).
- No barrel/mount scatter, no sensor cover glass / IR filter (approximate with `sensor_reflectance`),
  and no veiling glare from the diffuse scene. Use `post_psf.stray_light` for that.
- Sources are treated as at infinity (collimated). Below-threshold scene content casts no ghosts.
- Extended sources are sampled at `cluster_px` (default 8 px; flux-weighted centroid per block).
  Strongly magnified ghosts of an extended source therefore show faint `cluster_px`-step contours.
  Lower `cluster_px` for smoother results, at a run time that scales with the number of sources
  (observed 4-9 s per source, all 66 paths, `wide_22mm.dat`, `pupil_samples: 64`, 8 CPU cores).
- Polarisation: unpolarised average per interface, no polarisation state carried along the path.
- The `qhq` high-index layer (n = 2.10) is a representative value, not a specific catalog material.
- Ghost footprints are sampled by `pupil_samples^2` rays. Strongly magnified ghosts show
  grid-quantised vignetting edges at the default 64. Use 128 for figures.

## References

- M. B. Hullin, E. Eisemann, H.-P. Seidel, S. Lee, "Physically-based real-time lens flare
  rendering", ACM Trans. Graph. 30(4), 108 (SIGGRAPH 2011), doi:10.1145/2010324.1965003.
- S. Lee, E. Eisemann, "Practical real-time lens-flare rendering", Computer Graphics Forum 32(4),
  1-6 (EGSR 2013), doi:10.1111/cgf.12145.
- ISO 18844:2017, "Photography - Image flare measurement" (flare metric; not implemented here).
- M. Born, E. Wolf, *Principles of Optics*, 7th ed., Cambridge University Press, 1999, sec. 1.5-1.6.
- H. A. Macleod, *Thin-Film Optical Filters*, 4th ed., CRC Press, 2010.
- M. J. Dodge, "Refractive properties of magnesium fluoride", Appl. Opt. 23, 1980-1985 (1984), doi:10.1364/AO.23.001980.
- L. B. Lockhart, P. King, "Three-layered reflection-reducing coatings", JOSA 37, 689 (1947), doi:10.1364/JOSA.37.000689.
