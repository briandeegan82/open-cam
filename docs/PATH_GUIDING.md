# Path guiding (`guidedpath`, pbrt patch 0031)

`Integrator "guidedpath"` is pbrt's path tracer with [Intel Open PGL](https://github.com/RenderKit/openpgl)
(Apache-2.0) path guiding. It is opt-in, unbiased, CPU-only, and has no participating media.

```bash
PBRT_OPENPGL=1 tools/build_pbrt.sh                       # fetches + builds Open PGL v0.7.1 (needs TBB: libtbb-dev)
venv/bin/python tools/build_highway_scene.py --guiding ...  # also tools/build_tunnel_scene.py --guiding
```

## Why

At night the wet road reflects things that are lit, not lights: the rear of the car ahead in the
low-beam hot spot, and the headlamp highlights on its clear coat. Next-event estimation cannot aim
at them, so the path tracer finds them only when a ray reflected by the road happens to land there.
Most samples miss and a few are very bright: the speckle on the wet road. The light-sampler patches
(0022-0024) do not help, because the noise is not in direct lighting (see "Car paint roughness" in
[HIGHWAY_SCENES.md](HIGHWAY_SCENES.md)). Guiding learns, per region of the scene, the directions incident
light comes from and samples those directions.

## How it works

- Every path records its vertices (Open PGL `PathSegmentStorage`). After each render wave
  (1, 1, 2, 4, ... spp) up to `"integer trainingspp"` (default min(128, spp / 4)) the guiding
  field, a k-d tree of parallax-aware von Mises-Fisher mixtures, is refit to that radiance and the
  samples are dropped. Training waves are kept in the image (they are unbiased, just noisier).
- At reflective non-specular vertices the next direction is drawn from the BSDF or, with probability
  `"float guidingprobability"` (default 0.5), from the field times the cosine. The two are combined
  by one-sample MIS with balance-heuristic weights. The BSDF branch keeps the BSDF's own f/pdf ratio,
  because pbrt's layered BSDFs (wet road, car paint) only know an approximate PDF; the estimator stays
  unbiased with either. NEE and emitter hits are MIS-weighted with the same mixture PDF.
- Not guided: specular and transmissive vertices (glass, water passthrough), and volumes.
- The field gets one scalar per sample (the mean over the path's wavelengths); guiding needs where
  light comes from, not its colour.

## Measured

Night, wet road, puddles (`--time-of-day night --road-wetness wet --puddles`), 480x270, proxy cars with
the default paint (alpha 0.03), default guiding parameters. Noise is the per-pixel variance across seeds.

**Correctness.** One ego low beam, matte cars, 2048 spp, 4 seeds each: eight sub-regions of the lit road
agree with the path tracer within 1.4 standard errors, the total within 0.01 %, and the largest
deviation of the 32 spectral buckets is 2.8 standard errors. `tests/test_pbrt_guidedpath.py` repeats a
smaller version of this check.

**One ego low beam**, headlamp-lit road (`cropwindow 0.20,0.80,0.70,1.0`), 256 spp, 2 seeds:

| | RMS noise | p90 per-pixel noise | time |
|---|---:|---:|---:|
| path | 169 % | 102 % | 3.7 s |
| guidedpath | 34 % | 18 % | 5.7 s |
| path, 4096 spp | 42 % | 60 % | 54 s |

**Full scene** (all lights), 256 spp:

| region | path variance | guidedpath variance | equal-time gain |
|---|---:|---:|---:|
| road behind the lead car, 6 seeds | 1720 (one seed pair 4551, others 233 and 381) | 350 (seed pairs 312-410) | 3.4x |
| near road in the ego beam, 2 seeds | 90 | 33 | 1.8x |
| median per-pixel noise, road behind the lead car | 17.1 % | 16.7 % | |

The path tracer's variance there is dominated by rare, very bright samples; guiding removes them.
The typical pixel is barely affected: about half of its variance is direct light (a depth-1 render
keeps 130 of the 250), which guiding does not change. Guided samples cost about 1.5x (1.4-1.6x).
Training length (64, 128, 256 spp) and `guidingprobability` 0.3-0.7 gave the same result within the
seed noise.

**Worse:** the street-light housings (`luminaire`, geometry centimetres from a point light) are about
2x noisier with guiding; they are a few pixels at the top of the frame.

## Limits

- No participating media: haze, fog, spray and `--windscreen` scenes use `volpath`, and the builders'
  `--guiding` is then ignored with a warning. The urban, rural and garage builders always use `volpath`.
- CPU only (`--gpu`/`--wavefront` stop with an error); not combinable with `"bool regularize"` or the caustic
  photon split (`skipcaustics` is a volpath option).
- Delta lights through perfectly smooth surfaces (headlamp light mirrored by puddles or glass onto other
  surfaces) are still zero in any path tracer, guided or not.
