# open-cam — IS&T Electronic Imaging 2027 paper

Manuscript, bibliography, figures and the scripts that regenerate every number
in the paper. Nothing outside `paper/ei2027/` is modified; all scripts call the
existing open-cam tools.

Default authorship: Brian Deegan, University of Galway (edit `\author{}` in
`open_cam_ei2027.tex` to change). Template: IS&T proceedings template
(`ist.sty`, `logo.png` copied unchanged).

## Build the PDF

```bash
cd paper/ei2027
latexmk -pdf open_cam_ei2027.tex      # pdflatex + bibtex; output open_cam_ei2027.pdf
```

Requires TeX Live with `pdflatex`, `bibtex`, `latexmk`, `microtype`, `booktabs`, `hyperref`
(Ubuntu: `texlive-latex-extra texlive-fonts-recommended latexmk`).

## Regenerate results (from the repository root)

```bash
# 1. Spectral pbrt-v4 renders (CPU; ~15 min on 8 cores at the default sample counts)
paper/ei2027/scripts/render_scenes.sh          # SPP_CC=1024 SPP_EDGE=512 by default
# 2. Pipeline stages + all figures and JSON summaries
paper/ei2027/scripts/make_figures.sh
```

`render_scenes.sh` writes to `paper/ei2027/build/` (git-ignored, large):

| Output | Scene | Camera |
|---|---|---|
| `cc_D65.exr`, `cc_A.exr` | `build_colorchecker_scene.py`, 960x640, 64 buckets 360–830 nm, 1024 spp | realistic, `wide_22mm.dat`, 8.756 mm aperture, focus 4 m (as `config/pipeline.yaml`) |
| `edge_{f2_focused,f2_defocus,f56_defocus}.exr` | `build_image_quality_targets.py --target slanted_edge`, 5° edge at 3.2 m, 32 buckets 400–700 nm, 512 spp | realistic, `dgauss.50mm.dat`; aperture 25 / 25 / 8.93 mm, focus 3.2 / 2.6 / 2.6 m |

`make_figures.sh` then runs, with `--target-illuminance-lux 500 --integration-time-s 0.12`
(the `config/pipeline.yaml` defaults):

- `tools/pbrt_spectral_exr_to_electrons.py` for the `default` and `iphone_8` recipes;
- `tools/apply_emva_noise.py` (iphone_8, seed 0, Malvar demosaic) → `build/pipeline_{D65,A}/`;
- `tools/validate_demosaic_linear.py` and `tools/validate_emva_model.py`;
- the figure scripts:

| Script | Figure | Numbers |
|---|---|---|
| `scripts/fig_colorchecker.py` | `figures/fig_colorchecker.pdf` | `figures/colorchecker_summary.json` |
| `scripts/fig_ptc.py` | `figures/fig_ptc.pdf` (EMVA1288 photon transfer, iphone_8, 256x256, 29 levels, seed 1288) | `figures/ptc_summary.json` |
| `scripts/fig_mtf.py` | `figures/fig_mtf.pdf` (slanted-edge MTF, green QE×IRCF and per bucket) | `figures/mtf_summary.json` |
| `scripts/fig_cfa.py` | `figures/fig_qe.pdf`, `figures/fig_cfa_mosaics.pdf` | `figures/cfa_summary.json` |

Notes:

- `fig_cfa.py` computes non-Bayer channels (W, Ye, Cy) and mosaics **outside** the
  pipeline from the same spectral EXR using `sensor_radiometry` weights. The pipeline
  itself samples/demosaics Bayer phases only; `default_rccb`/`default_ryycy` are RGB
  proxies on a Bayer lattice. The paper states this explicitly.
- `fig_mtf.py` uses its own 50%-crossing edge fit (with per-row flat-fielding) and the
  repository's ESF binning/LSF/MTF functions, because `sfr_analysis.row_edge_positions`
  is biased by render noise on these edges (reports ≈ −2.3° for a ≈ −5.0° edge).
- The per-bucket MTF50 shows an unexplained periodic ripple; see the paper.

## Environment used

Ubuntu 22.04 VM, 8 CPU cores; pbrt-v4 CPU build at `third_party/pbrt-v4/build/pbrt`
(`docs/BUILD_PBRT.txt`); Python venv from `requirements.txt`; TeX Live 2022 (`pdflatex` 1.40.22).
