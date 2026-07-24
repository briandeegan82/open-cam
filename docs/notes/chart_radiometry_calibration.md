# Chart-scene radiometric calibration — analysis and resolution

Status: **RESOLVED for chart scenes** (2026-07-16, §0 below). Sections 1–3 are the
original 2026-07-15 analysis, kept for the evidence trail; §4's "open decision" is
settled by §0. Two follow-ups remain (§0.6).

Goal for this work: **maximum radiometric accuracy**. Every claim below is backed by a
measurement, and every number is reproducible from the commands in the last section.

---

## 0. Resolution: surround-probe scene-illuminance anchoring (2026-07-16)

### 0.1 The mechanism

Both chart builders place a large Lambertian **neutral surround** of flat reflectance
ρ = 0.22 behind the chart (`"rgb reflectance" [0.22 0.22 0.22]`). Three facts make it a
perfect radiometric anchor:

1. The distant light illuminates the chart/surround plane **exactly uniformly**, so any
   spatial variation of `E(x) = lux(π·L(x)/ρ)` is purely lens relative illumination V(x).
2. The chart and surround face +z with nothing in front of them, so **no interreflection
   can reach either** — verified: a chart patch at r=0.11 reads within 0.15% of the
   bare-surround value at the same radius.
3. pbrt normalises light SPDs photometrically, so the anchor is
   **illuminant-independent** — measured E₀(A) = 7170.7 vs E₀(F2_CWF) = 7164.2 raw-EXR
   lux (**0.09%**), and constant across the ColorChecker scene and six Munsell hue
   scenes to **<0.1%** per unit light scale.

The chart occludes the surround centre, so the anchor is measured from a **probe
render**: the generated chart `.pbrt` minus every patch block (camera, film, light and
surround untouched; 3 s on GPU at 256 spp). The centre value `E₀ = E(r→0)` (quadratic
extrapolation over r<0.25; the estimate is model-free to ~0.1%) is the scene illuminance
in raw-EXR photopic lux. Implementation: `tools/scene_illuminance_probe.py`.

### 0.2 The calibration

`calibration.scene_illuminance_reference_exr: E₀` (new mode in
`tools/pbrt_spectral_exr_to_electrons.py`), with `illuminant_override_csv` removed,
`radiometric_autocalibration: off`, `irradiance_scale_W_m2nm_per_unit: 1.0`:

```
E_e(x, λ) = L_exr(x, λ) · rad_to_e · target_lux / E₀
```

- `target_illuminance_lux` means **illuminance on the chart** — a lux meter at chart
  centre. Linear by construction.
- Every global factor pbrt bakes into the EXR (photometric SPD normalisation, the
  realistic camera's aperture weight `cos⁴θ·A_pupil/z²` @ cameras.cpp:946, the
  SpectralFilm's `CIE_Y_integral`, the light scale) is part of the measured E₀ and
  **cancels**; `rad_to_e = π/(4N²(1+m)²)` enters exactly once, by us. This closes the
  "does rad_to_e double-count the aperture?" question from §4.
- Vignetting stays in the data via the EXR's spatial structure, normalised to V(0)=1
  (measured profile: V=0.85 at r=0.53, 0.65 at r≈1.0; 10.4% of pixels outside the
  image circle are zero).

Both generators derive E₀ **per run** (no hand-entered constants) and additionally
**verify every chart EXR** against the probe's radial surround profile (measured
agreement ~0.05–0.1%; hard-fails at 1%), so rig drift between probe and chart render
(light scale, lens, framing) cannot silently rescale a dataset. Reference values for
this rig (light scale 1.0, cosθ = 0.982, wide_22mm @ 8.756 mm, cam_dist 6.0):
**E₀ ≈ 7160–7171** raw-EXR lux (statistical ~0.1%).

### 0.3 Why not FIX-C's frame-mean autocal (supersedes §3 for charts)

`lux(mean EXR)` averages the vignetting profile and the frame content:
the frame-mean-derived anchor (4867) sits **~25–30% below** the true on-axis
illuminance (7165), and varies ±4% across Munsell hue scenes (frame Λ 600–652 under
one illuminant) — content-dependent calibration error. The surround probe has neither
bias. FIX-C itself (autocal normalised on radiance L, not E_raw) **is applied** to the
tool — it is what keeps autocal f-number-responsive for 3D scenes — but chart scenes
no longer use autocal at all.

### 0.4 End-to-end verification (iPhone 8 Bayer, CWF @ 1000 lux, 20 ms)

Fully independent analytic path (repo spectral data only):
`E_img = ρ(λ)·E_chart(λ)·τ/(4N²(1+m)²)·V(r)`, `e = t·ff·A_pix·∫E_img·QE·λ/hc dλ`:

| patch | r | V | channel | analytic | pipeline | ratio |
|---|---|---|---|---|---|---|
| 19 white | 0.53 | 0.853 | R/G/B | 1489 / 2997 / 1067 | 1491 / 3003 / 1071 | 1.001–1.003 |
| 22 neutral .70D | 0.29 | 0.936 | R/G/B | 339 / 692 / 252 | 339 / 693 / 253 | 1.001–1.003 |
| 15 red | 0.11 | 0.984 | R/G/B | 480 / 294 / 78 | 480 / 295 / 78 | 1.001–1.003 |
| 13 blue | 0.46 | 0.877 | R/G/B | 72 / 198 / 243 | 72 / 199 / 244 | 1.001–1.004 |

f-number response (aperture 8.756 → 4.378 mm): **4.000 / 4.000 / 4.000** (expected 4).

### 0.5 A second stale-parameter fix found by the cross-check

`lens_models/phone_wide_f18.yaml` has `realistic_focus_distance: 1.0`, but the chart
generators frame and focus at 6.0; the electrons stage derives `(1+m)²` from the lens
model, overstating it by **3.8%**. Both generators now override
`lens.realistic_focus_distance = cam_dist` in the camera-model configs they write.

### 0.6 Remaining follow-ups

1. **Root-cause guard** in `photometry_calibration_scale`: `illuminant_override_csv` +
   active autocal (the original lux² pair) still only *documented*, not raised on,
   because `sensor_models/default.yaml` ships both keys to every recipe — raising
   changes/breaks every legacy consumer. The new reference mode does raise on any
   combination with the other two mechanisms. Needs its own decision.
2. **Highway/3D semantics under FIX-C**: autocal now normalises on radiance, so for 3D
   scenes `target_illuminance_lux` effectively anchors mean scene luminance (then
   optics apply), not mean sensor-plane lux as `build_highway_scene.py`'s comments
   state. Physically coherent and f-number-responsive, but the comments and any
   downstream expectations should be reconciled before the next highway regeneration.

---

---

## 1. The bug: lux applied twice

`sensor_models/default.yaml` (inherited by every recipe) sets **both**:

- `calibration.illuminant_override_csv` — the chart mechanism
- `pbrt_spectral_exr.radiometric_autocalibration: mean_photopic_lux` — the 3D-scene mechanism

Both scale by `target_illuminance_lux`, in two different places:

| where | factor |
|---|---|
| `_photometry_scale()`, `pbrt_spectral_exr_to_electrons.py:118-128` | `target_lux / ill_in` |
| autocal block, same file ~line 363 | `target_lux / scene_lux` |

Measured, straight from the recorded scale factors in `electrons.npz`:

| target | photometry_scale | autocal_scale | product |
|---|---|---|---|
| 1000 lux | 1.386e-02 | 27.12 | 3.76e-01 |
| 100 lux | 1.386e-03 | 2.712 | 3.76e-03 |

Each scales ×10 for a 10× lux change → **×100 combined**.

**Root cause.** The `elif` branch immediately below branch 1 documents the intent — "autocal
active: lux normalisation handled downstream from the EXR; nothing to do here" — but
branch 1 itself never checks whether autocal is active. It is a missing guard, not a
design disagreement.

`tools/build_highway_scene.py` states the intended split explicitly:

> Calibration note: for this 3D scene use `radiometric_autocalibration: mean_photopic_lux`
> with `--target-illuminance-lux` at the electrons stage; **the chart-style illuminant-CSV
> lux normalization assumes uniform illumination and does not apply here.**

and at line 733: *"3D scene: … target is mean sensor-plane photopic lux"*.

So the two mechanisms are **mutually exclusive**, and `target_illuminance_lux` means
**different things** in each: scene illuminance (chart) vs mean sensor-plane lux (3D).

### Why it went unnoticed

The final sensor-plane illuminance works out to `E_img = target²/L_D65`. At `target=1000`
that is `1000²/72143 = 13.87 lux`, and first-principles for a 1000-lux, ~20%-reflectance
chart at f/2.51 gives **15.43 lux** — within 11%. The baseline looks right *only at
1000 lux*. It diverges 11× at 100 lux and 56× at 20 lux.

---

## 2. Why the obvious fixes fail

### FIX-A — autocal only (drop `illuminant_override_csv`)

Linear ✓, but **exposure stops responding to f-number**.

`E_raw = L * rad_to_e * extra_scale`, and `illuminance_lux_from_irradiance` is linear, so
dividing by `lux(E_raw)` **cancels `rad_to_e = π/(4N²)` algebraically**:

```
E_e = L * rad_to_e * extra * photometry * target/lux(L * rad_to_e * extra)
    = L * photometry * target / lux(L)          <- rad_to_e is gone
```

Measured: f/1.8 → f/4.0 gives ratio **1.00** (expected 4.94). Autocal normalises the optics
out by construction — which is fine for its intended 3D-scene use, where the target *is*
the sensor-plane value.

### FIX-B — chart mode (autocal off + correct per-run illuminant CSV)

Linear ✓, f-number ✓, but **wrong cross-illuminant colorimetry** — a hard blocker for a
dataset whose purpose is comparing illuminants.

pbrt-v4 normalises light SPDs photometrically:

```
// src/pbrt/lights.cpp:266  (DistantLight — the light this scene uses)
sc /= SpectrumToPhotometric(L);     // "scale so that radiance is equivalent to 1 nit"
```

So the rendered EXR carries **no information** about the CSV's absolute lux — pbrt already
divided it out. Chart mode's `photometry_scale = target/L_csv` (note: `irradiance_scale`
cancels here) then divides by it a *second* time, injecting a per-illuminant scale error.

Measured — the error is a **constant across channels**, which is the signature of a pure
scale error rather than a spectral effect:

| | R | G | B |
|---|---|---|---|
| analytic A/CWF (both @1000 lux) | 1.49 | 0.95 | 0.78 |
| measured A/CWF | 0.21 | 0.13 | 0.11 |
| **measured/analytic** | **0.138** | **0.136** | **0.137** |

0.137 ≈ `L_CWF/L_A` = 9993/73611 = **0.1358**. Confirmed independently: `lux(mean radiance)`
is **309.8 (CWF)** vs **312.1 (A)** — 0.7% apart, despite the CSVs differing 7.37×.

Illuminant CSV absolute lux (the chart-mode anchor), for reference:

| CSV | lux at face value |
|---|---|
| D65 | 72142.9 |
| D55 | 68946.1 |
| A | 73610.6 |
| F2_CWF | 9993.4 |

The config anchors on **D65** while these datasets render with **CWF** and **A** — a 7.22×
mis-anchor for CWF even before the pbrt-normalisation problem.

---

## 3. FIX-C — verified  *(applied to the tool 2026-07-16; used by 3D-scene autocal.
Chart scenes use §0's reference mode instead — see §0.3 for why frame-mean autocal
is a biased chart anchor.)*

Normalise autocal on **radiance `L`**, not on `E_raw` (irradiance). This keeps `rad_to_e`
in the result (f-number preserved) while still deriving the absolute scale from the
rendered EXR (immune to pbrt's SPD normalisation).

```diff
--- a/tools/pbrt_spectral_exr_to_electrons.py
+++ b/tools/pbrt_spectral_exr_to_electrons.py
@@ the radiometric_autocalibration block
-            E_scene_mean = np.mean(E_raw, axis=(0, 1))
-            scene_lux = illuminance_lux_from_irradiance(lam, E_scene_mean)
+            # Normalise on RADIANCE, not on E_raw. E_raw = L * rad_to_e * extra_scale and
+            # illuminance_lux_from_irradiance is linear, so dividing by lux(E_raw) cancels
+            # rad_to_e = pi/(4 N^2) exactly -- which is why autocal made exposure
+            # independent of f-number. Normalising on L leaves rad_to_e in the result.
+            L_scene_mean = np.mean(L.astype(np.float64), axis=(0, 1))
+            scene_lux = illuminance_lux_from_irradiance(lam, L_scene_mean)
```

With `irradiance_scale_W_m2nm_per_unit = rho_mean/pi` (= 0.06366 for ColorChecker,
ρ_mean ≈ 0.20), no `illuminant_override_csv`, autocal on:

| check | result | |
|---|---|---|
| linear in lux | 10.0× / 50.0× | ✓ |
| f-number (aperture 8.756→4.378 mm) | **4.00×** | ✓ |
| colorimetry vs analytic | 1.01 / 1.00 / 1.01 | ✓ |
| absolute @1000 lux | 738 e⁻ (est. ~700–1000, full well 9500) | ✓ |

Electron counts, ColorChecker, iPhone 8 Bayer:

| illum | target | R | G | B |
|---|---|---|---|---|
| CWF | 1000 | 375.1 | 738.5 | 264.2 |
| A | 1000 | 563.0 | 700.1 | 207.1 |

A correctly reads **redder** than CWF (R 563 vs 375) for a 2856 K source. The pre-fix data
had this inverted.

### Useful invariant

`irr_scale = rho_mean/pi = Lambda/E_scene_exr`, where `Lambda = lux(mean radiance)`. The
`Lambda` cancels:

```
E_e = L * rad_to_e * target / E_scene_exr
```

`E_scene_exr ≈ 4867` for this rig (light `scale=1.0`, pbrt normalisation, cosθ=0.98). It is
a property of the **light**, not of scene content — so this formulation needs no per-scene
ρ_mean at all. Derivation: `E_scene_exr = Lambda/irr_scale = 309.83/0.06366`.

---

## 4. Open decision (Munsell) — *settled by §0: option 2, with the reference derived
per run from the surround probe instead of a stored constant*

`irr_scale = rho_mean/pi` is scene-dependent, and each Munsell hue family is its own scene
with different chips → ~40 different values. Under scene-illuminance semantics that is
*correct* (every scene lit at exactly 1000 lux), but it needs either:

1. **Derive per scene automatically** — `irr_scale = lux(mean EXR radiance)/4867`. No
   hand-maintained table; algebraically identical to `target = scene illuminance`.
   Requires validating that 4867 holds for the Munsell light rig.
2. **Add a calibration mode** — `scene_illuminance_reference_exr: 4867` with
   `photometry_scale = target/reference`, autocal off. Explicit, one constant, no ρ, no
   autocal. Slightly larger change to a shared tool.

Also worth deciding: whether per-hue ρ_mean is even desirable. It lights every hue at
exactly 1000 lux (physically right) but makes absolute electron levels differ per hue by
design. If hues must be directly comparable frame-to-frame, that is a different choice.

### Unresolved, flagged for accuracy — *settled, see §0.2*

**Does `rad_to_e` double-count the aperture?** The `realistic` camera traces rays through
the actual lens, so the EXR already contains vignetting (10.4% of pixels are exactly zero,
outside the image circle; centre-to-edge falls ~500→200). `rad_to_e = π/(4N²)` is then
applied *on top*. **Resolution (2026-07-16):** the camera's weight
(`cos⁴θ·A_pupil/(pdf·z²)`, cameras.cpp:946) does make the EXR film-irradiance-like, but
under §0's per-EXR anchoring the global part of that weight is inside the measured E₀ and
cancels; only *our* `rad_to_e` sets the absolute conversion, exactly once. The zero-pixel
frame-mean bias likewise disappeared with autocal (§0.3).

---

## 5. Scope of the bug

Models setting **both** keys (→ lux² on any chart render):

- `config/sensor_models/default.yaml` ← inherited by **every** recipe
- `config/camera_models/default.yaml`
- `config/camera_models/iphone_8.yaml` (legacy monolith)

A root-cause code fix (guard branch 1 on autocal, and raise if both are set rather than
silently squaring) would protect every future model, but changes calibration for every
recipe inheriting `default.yaml` — so it needs its own decision.

## 6. Data affected

*(updated 2026-07-16)* `out/dataset_dual/colorchecker/` and
`out/dataset_dual_ryycy/colorchecker/` have been **regenerated on the §0 calibration**
and are valid. The **Munsell halves** still predate this work, carry the original lux²
error, and are marked `INVALID_DO_NOT_USE.md`; regeneration needs ~4–5 h GPU
(2 illuminants × 40 hues, EXRs shared across cameras) via
`scripts/generate_dual_munsell.py`, which is fully wired for the new calibration.
Any other dataset produced through the illuminant-CSV + autocal default pair (§5)
remains suspect until regenerated.

## 7. Reproducing

```bash
# lux applied twice — photometry_scale and autocal_scale each scale with target
python - <<'PY'
import numpy as np
for p,l in [('1000lux_1x',1000),('100lux_2x',100)]:
    z=np.load(f'out/dataset_dual/colorchecker/{p}/CWF/electrons.npz')
    print(l, float(z['photometry_calibration_scale']), float(z['exr_radiometric_autocalibration_scale']))
PY

# pbrt normalises light SPDs photometrically
grep -n "SpectrumToPhotometric" third_party/pbrt-v4/src/pbrt/lights.cpp

# the two mechanisms are mutually exclusive, with different target semantics
sed -n '1,20p' tools/build_highway_scene.py
```
