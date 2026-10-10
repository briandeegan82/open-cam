# open-cam IQ lab: IS&T Electronic Imaging 2027 companion paper (draft)

This is the companion to `paper/ei2027/`. It covers the IQ-lab metrics, the ground-truth validation and the recipe scorecard. The manuscript only quotes numbers from `tables/*.tex`, and those files are generated from the tool outputs by `scripts/make_tables.py`.

## Regenerate (from the repository root)

```bash
# 1. Ground-truth validation (about 10 s)
venv/bin/python tools/validate_iqlab_metrics.py --out-dir out/iq_lab/validation
# 2. Recipe scorecard: every recipe in config/camera_recipes on all four scenes (several hours on 8 cores)
venv/bin/python tools/run_recipe_scorecard.py --xres 480 --yres 320 --pixelsamples 64 \
    --out-dir out/iq_lab/scorecard --figure
# 3. Tables, macros and figure
venv/bin/python paper/ei2027_iqlab/scripts/make_tables.py \
    --scorecard out/iq_lab/scorecard/scorecard.json \
    --validation out/iq_lab/validation/iqlab_validation.json
# 4. PDF
cd paper/ei2027_iqlab && latexmk -pdf open_cam_iqlab_ei2027.tex
```

- Step 1 needs `tools/validate_iqlab_metrics.py` (PR #39).
- Step 2 needs `tools/run_recipe_scorecard.py` (PR #40). It caches one row per recipe in `out/iq_lab/scorecard/work/rows/`, so an interrupted sweep resumes where it stopped.

Default authorship is Brian Deegan, University of Galway. `ist.sty` and `logo.png` are copied unchanged from `paper/ei2027/`.
