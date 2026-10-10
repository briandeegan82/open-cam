# IQ lab: lens flare and veiling glare

- `tools/build_flare_test_scene.py` builds two kinds of pbrt scene:
  - `veiling_glare`: an ISO 18844-style black-hole target. A uniform emissive field fills the view and has nine zero-reflectance holes: one at the centre, four on the diagonals at 0.7 field, and four at the edges at 0.75 field.
  - `point_sweep`: one small bright disc emitter on black for each field angle, from 0 to `--max-field-deg`.

  Use `--camera realistic` (lens file plus aperture) or `perspective`.
- pbrt's RealisticCamera has no inter-reflections, so after rendering, add the traced two-reflection ghosts with `tools/lens_ghosts.py` (Hullin et al. 2011).
- `tools/run_flare_test.py`:
  - `--ghost-sweep` reports ghost energy relative to the primary image against field angle and coating (`uncoated`, `mgf2`, `qhq`). It needs no render.
  - `--veiling-exr` scores rendered images for veiling glare % per hole.
  - `--point-exr` scores rendered images for stray-light fraction outside `--exclude-radius-px` and for the ghost peak relative to the source peak.
- `tools/iqlab/flare.py` holds the metrics. They find the holes and sources in the image itself, so they work through distortion and on real captures. Glare % = 100 × (hole core mean − hole true radiance) / mean of the ring around the hole.

The metrics follow the ISO 18844 / ISO 9358 structure but are not certified.
