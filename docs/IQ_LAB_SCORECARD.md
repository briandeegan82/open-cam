# IQ lab: recipe scorecard

`tools/run_recipe_scorecard.py` runs the four IQ-lab test scenes through the full pipeline for every camera recipe in `config/camera_recipes/`, then writes one comparison report.

```bash
# full sweep: all recipes, 720x480, 64 spp
venv/bin/python tools/run_recipe_scorecard.py --out-dir out/iq_lab/scorecard --figure
# quick subset
venv/bin/python tools/run_recipe_scorecard.py --recipes default default_hdr_dcg iphone_8 \
    --xres 360 --yres 240 --pixelsamples 16
```

Outputs: `scorecard.csv` (one row per recipe), `scorecard.json` (settings, optics groups, timing, rows), `scorecard.md` (table plus best per metric) and, with `--figure`, `scorecard.png`.

## What runs

| Scene | Builder | Metrics |
|---|---|---|
| HDR chart | `build_hdr_dr_chart.py` | DR to saturation at SNR = 1 and SNR = 10, SNR at 18 % of saturation, lowest level with CDP ≥ 0.9 |
| Diorama | `build_iq_diorama.py` | slanted-edge MTF50 and edge acutance, dead-leaves texture acutance |
| Skin chart (D65) | `build_skin_tone_chart.py` | mean and max ΔE00 over the skin patches |
| Black-hole flare | `build_flare_test_scene.py --targets veiling_glare` | median veiling glare % over the holes |

**Renders.** The spectral EXR depends only on the scene and the optics, so recipes are grouped by optics: `lens.camera`, plus pinhole/thin-lens FOV and lens radius, or realistic lens file and aperture. Each scene is rendered once per group. The 84 recipes fall into 13 groups (8 realistic lens/aperture, 4 thin-lens, 2 pinhole), so 34 scene renders replace 340. Every camera is focused on the target plane, and the camera distance is scaled with tan(FOV/2) so each chart fills the same part of the frame. A realistic lens's FOV is taken from its paraxial EFL and pbrt's 35 mm film diagonal. The HDR chart is emissive and the builder supports a pinhole only, so all recipes share one HDR render. DR, SNR and CDP are therefore sensor metrics and do not include lens transmission or vignetting.

**ROIs.** Pinhole and thin-lens cameras use the builders' projected ROIs. For realistic lenses, ROIs come from a 4-spp `Film "gbuffer"` render in world coordinates. A target's ROI is the bounding box of the pixels whose world hit point lies inside the target's inner rectangle. This follows the lens's real projection, including distortion and the image flip. Pinhole ROIs with a paraxial-equivalent FOV were checked against this and do not work: patch means were 30–60 % off.

**Sensor (per recipe).** Each EXR goes through `apply_emva_noise` with the recipe's camera model: QE, IRCF, EMVA noise, defects, CFA, HDR architecture and ADC. Electrons are recovered as `(raw − black) × K`, or as `hdr_e` from `hdr_linear.npz` for HDR recipes. Exposure is set per scene as follows:

- Diorama and skin chart are spot-metered on the white patch: ColorChecker white at 50 % of full well, and the skin chart's 90 % white at 60 %. Metering uses the brightest demosaiced channel (so the panchromatic W of RGBW does not clip) and, in the diorama, also the 90th percentile of the slanted-edge card, so the edge's white side can neither clip nor cross an HDR readout transition. The metering ceiling is the full well, or the ADC ceiling if lower (charge-binned quad Bayer keeps the photodiode's conversion gain, so its 10-bit ADC clips at about 4800 e⁻, not the 19200 e⁻ binned full well); for HDR pixels it is the first readout transition (e.g. the DCG high-gain ceiling), so the diorama and skin chart are scored on the primary readout. Metering iterates until it is within 2 % of the target. In the diorama's slanted-edge ROI, pbrt fireflies (pixels above 1.5× the white plateau) are replaced with their 3×3 median. The scorecard renders the diorama with the window and lamp switched off (`DIORAMA_UNIFORM_LIGHT`), so the cards are lit by the uniform fill panel alone, as in a lab. With them on, the lamp lit the edge card about 5× brighter than the ColorChecker and unevenly: the white side rose by about 18 % across the edge ROI. The measured MTF then levelled off near 0.5, and MTF50 for `default` ranged from 0.07 to 0.20 cy/px with the sensor-noise seed alone. Under uniform fill it is 0.37–0.40 cy/px. HDR and glare are covered by their own scenes.
- The HDR chart's brightest patch is placed at 90 % of the recipe's HDR saturation.
  The chart is always rendered at least 960 px wide and with at least 128 spp, whatever `--xres`/`--pixelsamples` say. The patches are only about 2 % of the frame width, and a 4×4 CFA tile leaves just one guide site per tile. The chart is captured twice with different noise seeds, so SNR and DR use temporal noise, free of pbrt Monte-Carlo noise and PRNU. CDP uses the first capture. For binned readouts such as 2×2 quad-Bayer charge binning, ROIs are scaled to the binned raster and the CFA collapses to its binned tile (`binned_layout`).
- The flare scene uses the 95th percentile at 50 %.

**Processing.** Gradient demosaic (`cfa_mosaic.demosaic`), then per-channel white balance on the white patch, then the recipe's spectral ColorChecker CCM under D65 (`cfa_mosaic.colorchecker_spectral_ccm`). The skin chart is never used to fit the CCM. SNR, DR and CDP use electrons at the guide-channel sites only (G, or C for RCCB-type CFAs), so no demosaic correlation enters them.

**Texture noise.** The diorama is rendered twice with different pbrt sampler seeds (`"integer seed" [1]` added to the scene's `Sampler`; pbrt's `--seed` option leaves the image unchanged) and captured once from each render with different sensor-noise seeds. The PSD of (I₁ − I₂)/√2 therefore contains both sensor and Monte-Carlo render noise, and is subtracted from the dead-leaves PSD, following ISO 19567-2's noise compensation. The texture MTF is normalised to its value at f ≤ 0.05 cy/px. If the noise PSD exceeds the capture PSD there, the texture is buried in noise and texture acutance is reported as NaN. This happens on RGBW, whose noise-free least-squares CCM has a W coefficient of about 14.

**Viewing condition.** Acutance uses `monitor_100pct` (CPIQ CSF).

## Caveats

- `lens.post_psf` (Gaussian post-PSF, stray-light halo) is not applied. Realistic lenses get their blur from pbrt's lens trace. Pinhole and thin-lens recipes have no lens MTF beyond defocus.
- Veiling glare is what pbrt's path tracer delivers: no coating or ghost model in the spectral render. Pinhole groups are about 0. If fewer holes are found than the scene has (9), e.g. because a large CFA colour matrix drives the luma negative, veiling glare is reported as NaN rather than a median of the remaining holes. For traced ghosts, see `run_flare_test.py --ghost-sweep`.
- ΔE00 includes the recipe's QE and IRCF. Recipes without an IR-cut filter (`ircf_csv: null`) colour-correct worse. That is a property of the recipe, not a scorer error.
- The metrics follow published methods (ISO 12233 slanted edge, CPIQ acutance, IEEE P2020 CDP, ISO 19567-2 dead leaves). They are not certified implementations.

**Long sweeps.** Each recipe's row is cached in `<out>/work/rows/<recipe>.json`, keyed on resolution, spp, seed and scenes. A rerun skips finished recipes, and skips any optics group whose recipes are all cached. If a recipe raises, it is reported under *Failed recipes* in `scorecard.md` and the sweep continues.
