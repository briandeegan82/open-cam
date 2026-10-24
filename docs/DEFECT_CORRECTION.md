# Opt-in dynamic defect pixel correction (DPC)

`tools/defect_correction.py` models the dynamic defect pixel correction found on-sensor or early in an ISP. It
runs on the noisy digital raw mosaic in `apply_emva_noise.py`, after the ADC (and digital CFA binning) and before
the raw is written. Each pixel is compared with its nearest same-colour neighbours. A pixel is replaced by their
median if it lies above their maximum (hot) or below their minimum (cold) by more than the expected noise:

```
sigma_DN = sqrt(max(median - black, 0) / K + (sigma_d / K)^2)
hot  if x > max(neigh) + threshold_sigma * sigma_DN + threshold_relative * (median - black)
cold if x < min(neigh) - (same margin)
```

Comparing against the neighbour range rather than the median keeps real edges: a pixel on a step edge always has
same-colour neighbours on its own side of the edge.

**It is off by default.** It pairs with the defect *injection* in `emva.defect_pixels` (hot/stuck pixels). It is a
camera processing stage, not a way to clean up pbrt render noise. In under-converged regions the mean comes mostly
from rare bright samples, so DPC removes real signal there along with the noise (see the measurement below). It
also removes real glints and point reflections, as a real camera's DPC does.

## Configuration

```yaml
processing:
  defect_correction:
    enabled: true
    threshold_sigma: 6.0       # margin in units of the expected shot + read noise
    threshold_relative: 0.0    # extra margin as a fraction of the neighbour median signal
    neighbours: 8              # nearest same-colour neighbours (4 or 8 typical)
    polarity: both             # hot | cold | both
```

Works with `bayer` and with generic periodic `cfa.layout` tiles (after binning, the binned layout is used). HDR
captures are rejected (`ValueError`): DPC per HDR exposure or on the merged output is not modelled yet.
`run_stats.json` records `defect_correction`: `corrected_count`, `corrected_fraction`, `hot_count`, `cold_count`,
the DN removed, and the parameters used.

## Measurement: night highway, wet road

Default research camera, night wet-road highway with pbrt patches 0022–0024, auto exposure (99th percentile at
80 % of full well). Signal removed is from the 1024 spp full frame. Pixel noise is from two independent 256 spp
crops (`std(A − B)/√2/mean`) with the same sensor seed, so it is the render noise plus sensor noise.

| region | pixel noise, DPC off | DPC on (6σ) | signal removed by DPC |
|---|---:|---:|---:|
| lead car, rear panel | 33 % | 29 % | 2.4 % |
| wet road just ahead of the lead car | 137 % | 119 % | 18.8 % |
| street-lit road, left | 23 % | 19 % | 0.2 % |

DPC corrects 0.88 % of the full frame's pixels at 6σ and 1.37 % at 4σ. Lowering the threshold to 4σ hardly changes
the table. On the road ahead of the car DPC lowers the noise by 13 % but makes the region 19 % too dark, because
that region's mean is carried by rare bright two-bounce samples (ego low beam → lead car → wet road). Use more
samples or better light sampling to fix render noise; use DPC to study what a camera's DPC does to real glints.
