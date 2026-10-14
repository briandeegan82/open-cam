# Road-tunnel HDR scene

`tools/build_tunnel_scene.py` builds a spectral pbrt-v4 road tunnel for HDR testing. It produces
two test images:

| view | camera | HDR problem | target |
|---|---|---|---|
| `outside_in` | approach road in daylight, 40 m before the entrance portal | **black hole**: the tube looks dark next to the sunlit exterior | proxy car in the threshold zone, 25 m inside the portal |
| `inside_out` | 60 m before the exit portal, at the start of the exit zone | **white hole**: the exit is bright next to the lit tube | proxy car 15 m before the exit, silhouetted against it |

`tools/run_tunnel_hdr_test.py` renders both views, passes each spectral EXR through
`apply_emva_noise` for a set of recipes and reports per-ROI SNR, clipping and contrast.

```bash
venv/bin/python tools/build_tunnel_scene.py                  # both views -> scenes/generated/tunnel/
(cd scenes/generated/tunnel && mkdir -p out && ../../../third_party/pbrt-v4/build/pbrt tunnel_outside_in.pbrt)
venv/bin/python tools/run_tunnel_hdr_test.py --out-dir out/iq_lab/tunnel_hdr   # render + score + previews
```

The scene needs no downloaded assets. The targets are the highway builder's box proxy cars, and
the sky is the analytic CIE clear sky.

Sky sampling: inside the tube nearly every sky direction pbrt picks is blocked, so daylight from the
portals is very noisy. By default (`--sky-portal-guide 0.5`) the sky light carries the two portal
rectangles (`"point3 guideportals"`, pbrt patch `0010-infinite-light-guide-portals.patch`). For points
behind a portal, half of the sky samples are aimed uniformly over the active portal areas and the other
half use the normal sky-map sampling, and both PDFs are combined, so the estimate stays unbiased. Points
outside the tube are sampled exactly as before. On the inside-out view at 128 spp, the ROI means agree with
unguided sampling within 1 % and the near-road noise drops from 46 % to 30 %. `--sky-portal-guide 0` restores
plain sky sampling.

## Geometry

World units are metres, y is up and the road runs along +z (world +x is image-right; this
was checked against pbrt renders, see `test_projector_mirror_convention`). All dimensions below
are modelling choices, not values from a standard.

- **Road:** two 3.66 m lanes (`LANE_W` from the highway builder) with 1.25 m kerbs to the walls.
  The markings are solid white edge lines and a dashed lane line using the highway `DASH`/`GAP`
  pattern. The road continues 1.5 km past the exit.
- **Tube** (`tube_profile`): vertical walls 4.8 m high, then an elliptical vault up to a 7.0 m
  crown. It is 400 m long (`--tunnel-length`) and lined with the highway `concrete` reflectance.
- **Portals:** a 40 m x 12 m concrete headwall at each end, with the tube cross-section cut out
  (`headwall`).
- **Hillside** (`hill_mesh`): soil over the whole tube, rising 25 m above the headwall tops
  inside the hill. Vertical soil faces next to each headwall close the cut. Grass ground lies
  outside the hill.
- **Luminaires:** two rows of LED line luminaires, one above each kerb at 5.4 m. Each is a
  0.12 m x 1.5 m one-sided Lambertian area light pointing down (`highway_night._quad_light`).
  Spacing is 3 m in the threshold zone, 4.5 m in the transition and exit zones and 9 m in the
  interior zone (`--spacing-*`). The spectrum is `led_4000k` (default) or `hps`, taken from `highway_night.LAMP_SPECTRA`.

## Photometry

| quantity | value | source |
|---|---|---|
| daylight on the road (sun + sky, horizontal) | about 8.1 x 10^4 lx at 45 deg sun elevation (manifest `lighting.reference_illuminance_lux`) | the highway clear-sky model, `highway_sky.clear_sky_illuminance_lux` (sources in [HIGHWAY_SCENES.md](HIGHWAY_SCENES.md)) |
| access-zone luminance L20 (`--l20`) | 3000 cd/m^2 | measured L20 in front of mountain-tunnel portals reached about 3 kcd/m^2 [H23] |
| threshold ratio k = L_th / L20 (`--k`) | 0.025 | example value for 80 km/h and 350 veh/h/lane [H23]; definition of K = L_th / L20 [I22] |
| threshold-zone road luminance L_th | k * L20 = 75 cd/m^2 | inside the 20-80 cd/m^2 threshold range discussed in [H23] |
| interior-zone road luminance (`--interior-luminance`) | 6 cd/m^2 | "benchmark of 6 cd/m^2 in the CIE specification" quoted in [H24]; LED designs in [H24] reach 7.06 and 8.16 cd/m^2 |
| threshold-zone length (`--threshold-length`) | 80 m, in two 40 m halves | modelling choice; [S23] uses 2 x 42 m |
| threshold second half | 0.5 L_th | [S23] |
| transition steps (`--transition-lengths`) | 0.15 / 0.05 / 0.02 L_th over 72 / 89 / 133 m; a step at or below the interior level is lit as interior | [S23] |
| exit zone (`--exit-lengths`) | 3 x then 5 x the interior level, over 30 m + 30 m up to the exit portal | [S23] |

The zones follow the CIE 88 definitions as summarised in [H23] and [H24] (access, threshold,
transition, interior and exit zones; L20 is the average luminance in a 20 deg cone, i.e.
2 x 10 deg, seen from the stopping-distance reference point and aimed at one quarter of the portal
height). I have not read CIE 88 itself, so **this scene does not claim compliance with it**.

**Zone profile** (`--zone-profile stepped`, the default). The road luminance follows the stepped
curve of the Chinese 2014 highway-tunnel lighting guideline as summarised in [S23]. I have not
read that guideline itself either. With the default L_th = 75 cd/m^2 and 6 cd/m^2 interior, the
0.05 and 0.02 L_th steps (3.75 and 1.5 cd/m^2) fall below the interior level, so only the first
transition step remains:

| zone | z from entrance portal [m] | road luminance [cd/m^2] |
|---|---|---|
| `threshold_1` | 0-40 | 75 |
| `threshold_2` | 40-80 | 37.5 |
| `transition_1` | 80-152 | 11.25 |
| `interior` | 152-340 | 6 |
| `exit_1` | 340-370 | 18 |
| `exit_2` | 370-400 | 30 |

Each step is constant: there is no continuous adaptation curve, and the step lengths are fixed in
metres rather than derived from the design speed and stopping distance. `--zone-profile two-step`
gives the earlier model (threshold, then interior up to the exit portal). The zones are written to
`photometry.zones`.

**Exit zone and the inside-out view.** With the default 60 m `--exit-distance` the `inside_out`
camera stands at the start of the exit zone, so the dark reference is the lit exit-zone road
(`exit_zone_road`, 18 cd/m^2 at 12-30 m ahead). This is why the default white-hole range drops
from 54.4 dB (two-step) to 44.9 dB: the exit lighting is there to soften the white hole. Use
`--exit-distance 100` (or more) to put the camera in the interior zone. The dark ROI is then
`interior_road` at 6 cd/m^2 (54.4 dB) and the lit exit zone gets its own `exit_zone_road` ROI. The
far exit-road ROI is then only a few pixels tall. The builder stops with an error if the road
12-30 m ahead straddles a zone boundary.

**Luminaire solve.** For each zone, the builder takes a 7 x 13 grid of road points at the zone
centre (+-6 m, or +-0.4 of a shorter zone; `zone_road_points`). It computes the mean direct
horizontal illuminance there from unit-luminance luminaires of every zone
(`direct_road_illuminance`), including light that spills over from neighbouring zones. It then
solves the resulting linear system (`solve_lamp_luminance`) so that `rho_Y * E / pi` equals each
zone's design road luminance at its centre. Here `rho_Y` is the
luminous reflectance of the asphalt under the lamp spectrum. The asphalt is **Lambertian**
(`diffuse` with the highway `asphalt_aged` albedo, so no road-surface r-table and no specular
lobe), so that this prediction is exact for direct light. The exterior road prediction is
`rho_Y(sun) * E_h / pi`, using the unoccluded sun + sky illuminance. In the default scene the
lamp luminances come out between about 2.3 x 10^4 cd/m^2 (transition, interior) and 1.0 x 10^5
cd/m^2 (`threshold_1`). They are written to `luminaires.luminance_cd_m2`, with per-zone lamp
counts in `luminaires.count_by_zone`.

With `--road-reflection rtable [--road-rtable NAME]` the asphalt is the pbrt `roadrtable` material
instead (see [ROAD_RTABLE.md](ROAD_RTABLE.md)). The luminaires are then solved so that the r-table
road luminance for a 1 deg observer looking along +z equals each zone's design value
(`direct_road_luminance`). The exterior road becomes `q(sun) E_sun,h + Q0 E_sky,h`. The default
table is a placeholder generated from open-cam's own asphalt model, not CIE data. Road points close
to the camera are seen at 2-6 deg, where the material blends toward Q0. With the glossy placeholder,
daylight entering through the exit portal brightens the inside-out exit-zone road well above its
design value (about 3.8x in the 480 x 320 test render, compared with up to 1.8x for the Lambertian
road).

**Predicted vs rendered.** The predictions cover direct light only. In the render, light reflected
by the lining and daylight coming in through the portal add to them. In the small RGB renders used
by the pbrt test, the measured road luminances are about 1.2-1.4x the prediction in the interior
zone, about 1.3x in the exit zone and about 1.5-1.6x in the threshold zone near the entrance. The exterior road is about 1.15x the
prediction, because of light reflected from the headwall and hill. The test accepts 0.9x to 1.8x
of the prediction and +-4.5 dB on the bright/dark ratio. For radiance cameras (pinhole/thinlens)
the EXR luminance is in cd/m^2.

## Manifests

`tunnel_<view>_manifest.json` can be passed straight to `apply_emva_noise` as
`--scene-manifest-json`. It provides `lighting.reference_illuminance_lux` and
`reference_illuminance_exr_lux` in the same way as the highway scene. It also records:

- `rois`: for each ROI, `roi_xyxy` (raster `[x0, y0, x1, y1]`, the pinhole projection of the
  `world_box` shrunk by 15 %), `region` (`exterior` / `dark` / `target`) and, for the road ROIs,
  `predicted_luminance_cd_m2`.
  - `outside_in` ROIs: `exterior_road`, `headwall`, `threshold_road`, `threshold_wall`, `target`,
    plus `sky` when the sky is in frame.
  - `inside_out` ROIs: `exit_zone_road` (and `interior_road` when the camera is in the interior
    zone), `interior_wall` (the tube wall beside the camera), `exit_opening`, `exit_road`, `target`.
  - With `--camera realistic`, the runner derives the ROIs again from `world_box` using a 4-spp
    gbuffer render, because pinhole ROIs do not apply to a real lens.
- `dynamic_range`: the bright ROI, the dark ROI, their predicted ratio and that ratio in dB.
  By default this is `exterior_road / threshold_road` = 32.5 dB for `outside_in` and
  `exit_road / exit_zone_road` = 44.9 dB for `inside_out` (54.4 dB against `interior_road`, see
  above). Sky and exit-opening luminances are
  higher than the road luminances, so the full scene range is larger (see the runner output).
- `target`, `photometry`, `luminaires`, `tunnel` (geometry) and `approximations`.

## Runner

`run_tunnel_hdr_test.py` uses the defaults `default` (linear) plus `default_hdr_dcg`,
`default_hdr_split_pixel` and `default_hdr_lofic`. Exposure metering and sensor runs reuse
`run_recipe_scorecard.run_sensor`:

- linear recipes (`ae`): the 99th percentile is set to 80 % of full well, as in the scorecard;
- linear recipes, second run (`@dark`): the dark ROI is set to 18 % of saturation (a choice: mid
  grey). This shows the opposite failure, where the exterior or exit clips;
- HDR recipes: the frame maximum is set to 90 % of the top readout, as in the scorecard's HDR chart.

A second capture with the next noise seed at the same exposure gives the temporal noise. The SNR
therefore excludes Monte Carlo noise and scene texture. For each ROI, on the CFA guide channel,
the runner reports:

- mean signal [e-];
- SNR [dB];
- clipped fraction: sites at or above 98 % of the effective saturation, or of the top readout for
  HDR recipes.

For each view it also reports:

- the Michelson contrast of the target against its background (`threshold_road` or
  `exit_opening`), next to the same contrast in the scene EXR;
- the measured bright/dark ratio in dB.

Outputs are `tunnel_hdr.json`, `tunnel_hdr.md` and previews. The preview PNGs are display-only
renderings:

- the scene tone-mapped with a global log curve;
- each linear recipe output, white-balanced on the sunlit headwall, with a colour matrix
  applied and clipped at saturation;
- each HDR recipe output, mapped through the same log curve.

## Sources

- [H23] "The influence of dimming-induced luminance change on driving safety in tunnels",
  *Heliyon* 9 (2023) e12928, doi:10.1016/j.heliyon.2023.e12928 (open access, PMC9871225).
- [H24] "The research for designing the LED tunnel light for highway tunnels", *Heliyon* 10
  (2024) e26305, doi:10.1016/j.heliyon.2024.e26305 (open access, PMC10877439).
- [I22] *Int. J. Environ. Res. Public Health* 19 (2022) 8517, doi:10.3390/ijerph19148517
  (open access, PMC9318541): definition of the threshold reduction factor K = L_th / L20.
- [S23] "Modelling and Mitigating Secondary Crash Risk for Serial Tunnels on Freeway via
  Lighting-Related Microscopic Traffic Model with Inter-Lane Dependency", *Int. J. Environ. Res.
  Public Health* 20 (2023) 3066, doi:10.3390/ijerph20043066 (open access, PMC9967854), section 4:
  stepped zone luminances (0.5 L_th; 0.15 / 0.05 / 0.02 L_th, not below the interior level; exit
  3 and 5 x interior) and example zone lengths from "Guidelines for Design of Lighting of Highway
  Tunnels", Ministry of Transport of the People's Republic of China, 2014.
