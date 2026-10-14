# Road r-table reflection (`roadrtable` material)

Road brightness under street and tunnel lighting depends on how the asphalt reflects light toward a driver
looking along the road at a grazing angle. Road-lighting standards describe this with *reduced luminance
coefficient tables* (r-tables, CIE 30.2 / CIE 144:2001 / EN 13201-3, classes R1-R4 and C1/C2). open-cam can
render the road with such a table through a patched pbrt material.

**No CIE table ships with the repo.** The values are copyrighted and no verifiable open copy was found.
`config/road_rtables/placeholder_dry_model.csv` is generated from open-cam's own dry-asphalt layered model
(`tools/highway_wet.py`) on a CIE-style grid so the plumbing can be tested. It is **not CIE data**. Builders print a
warning and manifests record `"placeholder": true` whenever it is used.

## Definitions

- gamma: incidence angle of the light from the road normal; tables are indexed by tan(gamma) (0-12).
- beta: angle between the vertical plane of observation and the plane of incidence; beta = 0 means the light
  is beyond the point, facing the observer.
- The observer is about 1 deg above the road (the CIE r-table geometry).
- r(tan gamma, beta) = q cos^3(gamma), where q = L / E_h [cd m^-2 lx^-1] is the luminance coefficient.
  For this observer, q is the BRDF.
- Q0 is the solid-angle mean of q over tan(gamma) <= 12 and all beta. S1 = r(2, 0) / r(0, 0).

## Table files

CSV in `config/road_rtables/`. `--road-rtable R3` resolves to `config/road_rtables/R3.csv`; a path also works.

```
# name: R3
# source: CIE 144:2001 Table ... (or EN 13201-3)
# placeholder: false
# scale: 1e-4            # the standards print 10^4 r
tan_gamma\beta_deg,0,2,5,10,...,180
0,<10^4 r values ...>
0.25,...
```

Any grid is accepted (strictly increasing tan(gamma) and beta, 0 <= beta <= 180, r >= 0).
`python tools/road_rtable.py R3` prints Q0, S1 and the directional albedo as a transcription check. Compare Q0
and S1 with the values printed in the standard.

## pbrt material

`third_party/patches/0007-road-rtable-material.patch` adds `roadrtable` (`RoadRTableBxDF`):

```
MakeNamedMaterial "asphalt" "string type" "roadrtable" "spectrum tint" "spd/asphalt_aged_rtint.spd"
    "float tangamma" [...] "float beta" [...] "float r" [...]   # row-major [tan][beta]
    "float rscale" [1]  "float observationmin" [1.5] "float observationmax" [15]
```

- r is interpolated bilinearly and then divided by cos^3(gamma). tan(gamma) and beta are clamped to the table,
  so lights beyond tan(gamma) = 12 use the last row.
- **Observer elevation (open-cam assumption).** r-tables are defined only for the 1 deg observer. For viewing
  elevations above `observationmin`, the BRDF blends (smoothstep) into a Lambertian lobe with the table's Q0,
  and it is fully Q0 above `observationmax`. Road points close to a camera mounted 1.35 m high are seen at a
  few degrees, so their luminance differs from the 1 deg design value.
- `tint` carries the spectral shape. The builders use rho(lambda) / rho_Y of `asphalt_aged` (luminous
  reflectance 1), so the table sets the luminance and the tint sets only the colour. The table itself is
  achromatic.
- Sampling: the incident direction is importance-sampled from a tabulated q cos(gamma) over (gamma, beta),
  mixed with cosine sampling for the Q0 lobe (at least 10 % cosine). The lobe is flagged diffuse. Energy is not
  renormalised: the table is rendered as given.
- `tools/road_rtable.py` mirrors the C++ evaluation (`RTable.q`, `RTable.brdf`, `RTable.q0`).
  `tests/test_road_rtable.py` checks pbrt renders against it.

## Scene builders

- Tunnel: `tools/build_tunnel_scene.py --road-reflection rtable [--road-rtable R3]`. Luminaires are solved so
  that the r-table road luminance for the 1 deg observer looking along +z equals each zone's design value
  (`direct_road_luminance`). The exterior road luminance is q(sun) E_sun,h + Q0 E_sky,h; the sky is treated as
  uniform. The default (`lambertian`) scene is unchanged.
- Highway: `tools/build_highway_scene.py --road-rtable R3 --road-wear none`. This replaces the dry asphalt,
  plain or textured; the luminance map scales the tint. It cannot be combined with road wear or wetness, whose
  materials are their own layered models.

## Limits

- Only the dry tables are supported. Wet tables (W1-W4) would just be more CSVs, but the wet-road model in
  `tools/highway_wet.py` is separate.
- Observer elevations above about 1.5 deg are an assumption (blend to Q0), not part of the standard.
- Render noise: on the inside-out view most of the near-road noise is daylight from the exit portal reflected
  by the glossy table toward the camera, not the tunnel lamps (128 spp: lamps only 4 %, sky only 160 %). pbrt
  samples the sky over the whole sphere, and almost all of those directions are blocked by the tube. The tunnel
  builder therefore aims part of the sky samples through the portal openings (`--sky-portal-guide`, see
  [TUNNEL_SCENES.md](TUNNEL_SCENES.md)). In the 480 x 320, 128 spp inside-out render this cuts near-road
  noise (std / mean) from 46 % to 30 %, and from 156 % to 97 % with only the sky lit. The remaining noise is
  mostly rare bright paths; use more samples, or the display denoiser for pictures only.
- The placeholder is open-cam's own model: S1 is about 2.9 and Q0 about 0.058, so it is not an R-class. Do not
  quote numbers rendered with it as CIE road luminances.
