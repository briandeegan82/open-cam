# IQ lab: HDR dynamic-range chart

`tools/build_hdr_dr_chart.py` writes `hdr_dr_chart.pbrt` and `chart.json`. The chart has 21
log-spaced emissive patches spanning 120 dB by default (1e6:1, 6 dB steps). Each patch is split
in two:

- the left half (luminance L) is used for SNR and dynamic range;
- the right half (L·k, with k = (1+C)/(1−C) and C = 0.2 by default) gives every level a fixed
  Michelson contrast, so the same chart also produces the IEEE P2020 CDP-vs-luminance curve.

The patches are diffuse area lights on a zero-reflectance background, imaged by a pinhole camera.
That means there is no inter-patch light transport, and each ROI sees its patch's radiance exactly
(checked against a real render in `tests/test_iqlab_hdr_dr.py`). `chart.json` stores each
half-patch ROI in raster pixels. Note that pbrt `LookAt` places world +x on the raster left.

`tools/run_hdr_dr_test.py` scores the chart with `iqlab`:

- **Model mode** (default): each level is simulated as a uniform patch through the recipe's pixel
  model: `hdr_pixel` DCG, split-pixel, LOFIC or 3-exposure merge, or a single linear readout for
  non-HDR recipes. Two frames give temporal SNR. The brightest half-patch sits at saturation
  (`--top-fraction`).
- **Measured mode**: `--hdr-npz frame1.npz [frame2.npz]` scores real or rendered linear electrons,
  such as `noisy_hdr/hdr_linear.npz` from `apply_emva_noise.py` run on the rendered chart.

Reported values per recipe:

- per-patch mean, noise, SNR, saturated fraction, CDP and sensor-plane lux;
- `dr_snr1_db` and `dr_snr10_db`, limited by the chart: brightest unsaturated patch divided by
  the SNR = 1 or SNR = 10 signal;
- `dr_snr*_to_saturation_db`, measured from the saturation level;
- `theory_dr_snr*_db`, from `hdr_pixel.theory_snr`.

`--figure` adds plots of SNR and CDP against signal.

Model mode, default recipes, SNR = 1 dynamic range to saturation (measured / theory):

| recipe | DR SNR=1 | DR SNR=10 |
|---|---|---|
| default | 62.6 / 62.8 dB | 33.0 dB |
| default_hdr_dcg | 80.7 / 80.7 dB | 46.9 dB |
| default_hdr_split_pixel | 104.3 / 104.3 dB | 70.8 dB |
| default_hdr_lofic | 92.9 / 92.6 dB | 60.2 dB |
| default_hdr_3exp | 118.7 / 118.3 dB | 86.7 dB |

Model mode does not include optics: flare and veiling glare limit real scene DR. For an
end-to-end number, render the chart through a realistic lens and score it in measured mode.
