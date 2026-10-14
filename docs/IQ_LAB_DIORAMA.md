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
