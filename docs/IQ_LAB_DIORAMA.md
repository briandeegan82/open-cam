# IQ lab: procedural diorama

`tools/build_iq_diorama.py` builds a tabletop pbrt scene (`iq_diorama.pbrt`, plus spd files and 16-bit linear PNG textures) that exercises the whole IQ lab in one frame. It needs no external assets.

| element | stresses |
|---|---|
| spectral 24-patch ColorChecker, 12-patch skin chart (`iqlab/skin.py`) | colour, skin tones, white balance |
| dead-leaves card (`iqlab/dead_leaves.py`) | texture MTF / noise reduction |
| 36-spoke Siemens star, 5° slanted edge | resolution, SFR / MTF50 |
| matte, glossy (coated diffuse), metal and glass spheres 0.55 m in front of the cards | focus / depth of field, specular highlights |
| 6500 K window in the back wall (~700× the shadowed wall) and a 2700 K bulb | scene dynamic range, flare, mixed illuminants |

`diorama.json` gives the raster ROIs for every chart patch, card, sphere, the window and a shadow reference. The ROIs come from the pinhole projection, which matches `--camera perspective`. With `--camera realistic` or `thinlens` they are only approximate because of distortion, and the focus defaults to the card plane.

`--glass-roughness R` (default 0, smooth) makes the glass sphere a rough dielectric. At 480 x 320, 256 spp with
the realistic lens and dispersion split, R >= 0.15 removes the fireflies around the glass sphere (85 to 0) and
cuts fireflies over the whole image by about 40 % (229 to 137 pixels). Noise on flat areas is unchanged (6.7 %).
The sphere then no longer acts as a clear refracting target: its caustic spreads out (the peak falls from
about 250x to 30x the local level). Keep the default for caustic and refraction work.

Scoring examples:
- Texture acutance: `iqlab.dead_leaves.texture_mtf` on the dead-leaves ROI.
- Colour: `iqlab.cpiq.chroma_level` on the chart patch ROIs.
- SFR: `sfr_analysis.slanted_edge_sfr` on the edge ROI.
- Scene contrast: window ROI / shadow ROI.

## Caustics: spectral SPPM (opt-in)

`--integrator sppm` switches the diorama from `volpath` to pbrt's stochastic progressive photon
mapping. Photon mapping converges on the focused caustic under the glass sphere (and paths that
reach the lamp via the glass/metal spheres), which a path tracer leaves grainy at any practical spp.

- `--pixelsamples` sets the number of SPPM iterations; `--sppm-photons` (photons per iteration,
  default 200000) and `--sppm-radius` (initial gather radius in metres, default 0.02) set the
  photon budget and the initial blur. The radius shrinks every iteration, so the estimate is
  consistent but biased (slightly blurred caustics/shading) at a finite iteration count.
- The spectral film works with SPPM through pbrt patch `0009-sppm-spectral-film.patch`: per-pixel,
  per-bucket accumulators write the same `RGB` + `S0.<lambda>nm` EXR layout as `volpath`, so the
  sensor model reads it unchanged. Upstream SPPM only supports RGB films.
- Each iteration uses one wavelength set for the whole image (shared by the camera and photon
  passes), so at low iteration counts the image has a global colour error: on the diorama the RGB
  means were 10-15 % off `volpath` at 64 iterations and within 3 % at 1024. Use hundreds of iterations
  or more.
- Realistic lenses work. With a dispersive lens only the hero wavelength reaches each visible point;
  with `--lens-dispersion-split` SPPM does not trace four camera paths but rotates the hero per pixel
  and iteration, which decorrelates the colour noise between pixels.
- CPU only (no GPU/wavefront SPPM), no participating media, and `volpath` remains the default.

Measured on the default diorama (pinhole, 480x320, same seed, about equal time: volpath 1024 spp
104 s vs SPPM 1024 iterations x 100k photons, radius 0.01 m, 108 s), high-pass noise in the RGB
channels:

| region | volpath lum / chroma | spectral SPPM lum / chroma | RGB-film SPPM lum / chroma |
|---|---|---|---|
| floor beside the glass sphere (caustic) | 0.26 / 0.047 | 0.27 / 0.020 | 0.28 / 0.012 |
| ColorChecker white patch | 0.07 / 0.024 | 0.24 / 0.070 | 0.18 / 0.027 |
| matte sphere | 0.11 / 0.027 | 0.20 / 0.059 | 0.19 / 0.021 |

SPPM halves the colour noise in the caustic region but is 2-3x noisier on directly lit diffuse
surfaces at equal time, so it is not a general replacement for `volpath` here. The spectral film
adds colour noise over RGB-film SPPM: it samples 360-830 nm uniformly (RGB films importance-sample
the visible range) and every pixel shares the iteration's four wavelengths. Fewer iterations with
more photons (64 x 2M) was worse everywhere.

## Caustics: photon split inside the path tracer (default)

`--caustics` (default `auto`) chooses how the caustic under the glass and metal spheres is rendered:

| mode | used for | integrator |
|---|---|---|
| `photons` | `auto` with `--integrator volpath --film spectral` | split: `volpath "skipcaustics"` + `sppm "causticsonly"` (below) |
| `path` | `auto` otherwise (RGB film, `--integrator sppm`); use it for `--wavefront`/GPU | plain `volpath` |
| `regularized` | the pre-split default, for comparison with older renders | `volpath "regularize"` |

`regularize` roughens every GGX lobe below 0.3 (to 2x, clamped to 0.1-0.3) after the first non-specular
bounce, so the smooth spheres act as rough ones for the floor's caustic. That cuts caustic noise but
biases it: in a 16384 spp crop of the 960 x 640 `wide_22mm` lab render the caustic core is 33.9 with
`regularize`, 38.6 without and 38.2 with the photon split, while the floor and walls agree within 0.2 %.
More samples don't remove the 12 % deficit, so the default no longer uses it.

The photon split keeps `volpath` for the whole image and hands only the caustic light to
photons (pbrt patch 0021). The builder writes two scenes with the same camera, film and seed handling:

- `iq_diorama.pbrt`: `volpath` with `"bool skipcaustics" true`. It drops the light paths
  E (S|G)* D S+ L: after the first diffuse vertex D bounces non-specularly, emission reached through
  one or more specular (glass, mirror) bounces only is not added. Everything else, including direct
  light at D, MIS, glossy paths and everything after a second non-specular bounce, is unchanged.
- `iq_diorama_caustics.pbrt`: `sppm` with `"bool causticsonly" true`. Visible points are made only at
  the first diffuse surface, camera paths add no emission or direct light, and only photons whose
  bounces after leaving the light were all specular are gathered (L S+ D).

The two passes cover disjoint paths, so `python tools/combine_caustics.py OUT_DIR --render` runs both
and sums them into the usual film output (rendering `iq_diorama.pbrt` alone gives
`iq_diorama_spectral_nocaustics.exr`, the image without the caustic). It also renders unsplit scenes in one
pass, and `--tag _seed1 --seed 1` writes an independent render beside the main one; the recipe
scorecard uses it for both diorama renders. Textures, edges and direct lighting are path traced as
before; only the caustic is a photon estimate, blurred by the gather radius (`--caustic-radius`,
default 5 mm, shrinking every pass as in SPPM; `--caustic-photons`, default min(500k, pixels);
`--caustic-iterations`, default `--pixelsamples`). On the full lab render the photon pass adds about
27 % to the render time and halves the caustic noise (12.1 % to 6.8 % at 1024 spp); floor noise and
slanted-edge MTF50 are unchanged.

Limits: CPU only (the wavefront/GPU integrator ignores both flags, so a split scene rendered there would
count the caustic twice; build with `--caustics path`); caustics through participating
media and BSSRDFs stay with the path tracer; passthrough dielectrics (patches 0017/0020) are skipped as
null surfaces by `volpath` but refract the photons, so scenes with them don't split exactly.
