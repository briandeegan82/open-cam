# Lens surface scatter (veiling glare)

`tools/lens_scatter.py` adds the diffuse stray light of polished lens surfaces to a spectral render.
pbrt's realistic camera traces only the specular primary path, so without it the black-hole
veiling-glare scene reads ~0 % for every lens. The scorecard combines it with the traced
two-reflection ghosts of `tools/lens_ghosts.py` (see [LENS_GHOSTS.md](LENS_GHOSTS.md)).

## Model

For each refracting surface `i` (air/air interfaces and the stop are skipped):

- **Total integrated scatter**, Rayleigh–Rice smooth-surface limit:
  `TIS_i(λ) = (2π σ |n_after − n_before| / λ)²`, with σ the RMS surface roughness.
  TIS falls as 1/λ², so scatter is bluer than the image.
- **Angular profile**, ABg / Harvey: `BSDF(β) ∝ 1 / (B + β^g)`, with β the direction-cosine offset
  from the specular ray and `B = shoulder^g`. The profile is normalised over the unit disc.
- **Propagation to the film**: a paraxial ray-transfer matrix from the surface to the film turns a
  direction-cosine change `β` into a film radius `k_i β` (`k_i = n_i |M_wy|`). Scatter cones are
  truncated where the later clear apertures stop them.
- **Film kernel**: the per-surface kernels are summed into one centred, shift-invariant kernel.
  Each wavelength band is `(1 − TIS(λ)) · I + TIS(λ)/TIS₀ · (K ⊗ I)`, so the scatter conserves energy.

Defaults (`lens_scatter.DEFAULTS`), from the literature for good commercial polish; they are
assumptions, not measurements of these lenses:

| Parameter | Default | Meaning |
|---|---|---|
| `roughness_nm` | 1.5 | RMS surface roughness σ of every refracting surface |
| `abg_g` | 2.0 | ABg slope g (Harvey `g = 2` is a common polished-glass value) |
| `abg_shoulder` | 0.01 | roll-off direction cosine (B = shoulder^g) |
| `reference_nm` | 550 | wavelength at which the kernel is built; other bands scale by (λ₀/λ)² |

## Field dependence and barrel scatter (opt-in)

- `field_zones: N` (default 1, which is shift-invariant as before) builds a kernel at N image
  radii, from the centre to the corner, and blends them with linear weights in the source's
  radius: `out = sum_z K_z * (w_z img)`.
  - *Cone truncation:* off axis, each surface's scatter cone starts from the paraxial chief ray
    at height y_i. A later aperture a_j then cuts it at `(a_j - |y_j|) / (n_i |C_ij|)`, so more of
    the cone goes into the barrel.
  - *Obliquity:* the chief ray's incidence angle scales TIS by the Rayleigh–Rice transmission
    factor `((n2 cos t - n1 cos i) / (n2 - n1))^2`.
  - *Effect:* for `wide_22mm` at f/2.75, the share of the scatter that reaches the film falls
    from 0.89 on axis to 0.79 in the corner of a 14.4 x 9.6 mm frame. Scatter around a corner
    source is up to ~20 % lower than with the shift-invariant kernel.
  - *Cost:* N FFT convolutions per channel instead of one.
- `barrel_reflectance: rho` (default 0, where the truncated scatter is absorbed) sends rho times
  the truncated scatter back to the film as a uniform veiling floor over the frame. This is
  nominal: black anodised barrels are about 0.02–0.05, the geometry isn't traced, and all of the
  re-emitted light is assumed to reach the frame (an upper bound).

## Not modelled

Particulate contamination, edge scatter, out-of-field sources, the sensor cover glass's own
scatter, the angular distribution of the barrel light, and any wavelength dependence of the
ABg shape. Real lenses' ISO 9358 veiling-glare index is usually dominated by barrel and
mount scatter, so these numbers are a lower bound for the optics, not a prediction of a real
lens' VGI.

## In the recipe scorecard

`run_recipe_scorecard.py` post-processes the realistic-lens flare render
(`add_lens_stray_light`): the primary image is multiplied by the coating-model primary
transmittance, surface scatter is applied, and traced ghosts (MgF₂ coating, 40 px source
clusters, 32² pupil samples) are added. `veiling_glare_pct` is then scored as before;
`flare_ghost_to_source` and `flare_scatter_tis` are also reported. `--no-stray-light` scores the
bare pbrt render. Pinhole and thin-lens groups have no prescription and are unchanged.

Black-hole veiling glare, median of the 9 holes, on the CIE Y (luminance) weighted spectral render
(480×320, 64 spp, before the sensor):

| Lens group | pbrt only | + scatter | + ghosts | both | ghost / source energy |
|---|---|---|---|---|---|
| `dgauss.50mm` f/4.1 (12.34 mm) | 0.00 % | 0.05 % | 0.03 % | 0.08 % | 0.25 % |
| `wide_22mm` f/2.75 (8 mm) | 0.00 % | 0.07 % | 0.04 % | 0.11 % | 0.36 % |

The MgF₂ quarter-wave coating reflects least near its 550 nm design wavelength, so the ghost
light is magenta and contributes less luminance than its energy share suggests; the Rayleigh–Rice
scatter (TIS ≈ 0.11 % at 550 nm) is the larger luminance term.

Through the full scorecard (sensor, demosaic, CCM, `veiling_glare_pct`), `dgauss.50mm` f/2:
`nikon_z6` 0.003 % → 0.079 %, `canon_eos_20d` 0.058 % → 0.111 %. The bare-render value is the
sensor floor (noise clipped at zero before demosaicing), which differs by recipe.

## Validation

`tests/test_lens_scatter.py`: ABg normalisation, 1/λ² TIS, the paraxial film-shift coefficient
against a traced ray, last-surface truncation, kernel energy against the analytic integral, and
black-hole glare against the closed-form g = 2 result.
