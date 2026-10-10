# 08 - Image-quality lab

```bash
opencam-gui demo iq-lab            # or: PYTHONPATH=gui venv/bin/python -m opencam_gui.ui.cli demo iq-lab
opencam-gui demo iq-lab --scenario hdr_dcg
```

The IQ lab runs a single test scene through a single camera recipe, using the same code as the recipe scorecard (`tools/run_recipe_scorecard.py`, see `docs/IQ_LAB_SCORECARD.md`).

**Run** does the following:

1. Builds the scene and renders it with pbrt through the recipe's own optics.
2. Runs the recipe's sensor chain: QE, IR-cut filter, EMVA noise, CFA, HDR architecture, demosaic, white balance and colour-correction matrix.
3. Scores the result.

The preview shows the processed image with the measurement ROIs drawn on top. The table lists the metrics for the selected scene. A built PBRT binary is required: run `tools/build_pbrt.sh` first.

| Scene | ROIs drawn | Metrics |
|---|---|---|
| Diorama | slanted edge (orange), dead leaves (blue), ColorChecker white used for metering and white balance | MTF50, edge acutance, texture acutance |
| HDR chart | low (blue) and high (orange) half of every level | DR at SNR = 1 and SNR = 10, SNR at 18 %, CDP ≥ 0.9 level |
| Skin chart (D65) | every skin patch | ΔE00 mean and max |
| Flare target | every black hole | median veiling glare |

The HDR chart is shown as a log map of the guide-channel electrons over 120 dB. It is always rendered at least 960 px wide with 128 spp.

## Lecture scenarios

- **Edge and texture acutance.** The slanted edge measures sharpness on a high-contrast step. Dead leaves measures it on low-contrast texture. Raise samples per pixel to 64 or more, because pbrt Monte-Carlo noise inflates texture acutance.
- **Dynamic range: linear vs dual conversion gain.** Same chart and optics, two pixel architectures. Compare DR at SNR = 1. On the scorecard smoke run it was about 62 dB for `default` and 81 dB for `default_hdr_dcg`.
- **Skin-tone accuracy through a phone lens.** The colour-correction matrix comes from the sensor's own spectral ColorChecker fit, never from the skin chart.
- **Veiling glare.** Light scattered into the black holes of a bright field. pbrt's spectral render has no coating or ghost model.

Low-resolution runs (360×240, 16 spp) take about 10–40 s per scene on a laptop CPU. Realistic lenses also render a 4-spp gbuffer to locate the ROIs.
