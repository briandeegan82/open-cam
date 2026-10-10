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
