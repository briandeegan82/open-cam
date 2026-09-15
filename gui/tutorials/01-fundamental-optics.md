# Tutorial 01 -- Fundamental Optics / Imaging Geometry

**Demo:** `opencam-gui demo geometry`
**Audience:** graduate / advanced undergraduate, no optics background assumed
**Goal:** get from "a lens forms an image" to the four numbers every later
tutorial depends on -- field of view, magnification, depth of field, and the
falloff and distortion a real lens adds on top of the ideal one.

Every number and curve comes from calling `tools/imaging_geometry.py`, which is
pure first-order (Gaussian) optics with no PBRT dependency. The cos^4 falloff
delegates to `sensor_radiometry.cos4_vignetting_from_pinhole` and the distortion
grid uses the same Brown-Conrady coefficients `tools/spectral_sensor_forward.py`
inverts, so the demo and the render path cannot drift apart.

This tutorial deliberately contains no blur. A first-order lens is a perfect
one: every object point maps to exactly one image point. Tutorial 02 takes that
perfect point and spreads it.

---

## 1. Principles

### 1.1 The thin-lens equation

A lens with focal length *f* images an object at distance `s_o` to an image at
distance `s_i`:

```
1/f = 1/s_o + 1/s_i
```

Three consequences worth saying out loud:

- At `s_o = infinity`, `s_i = f`. "Focal length" is the image distance for a
  distant subject, which is why it is the number engraved on the barrel.
- As `s_o` falls toward *f*, `s_i` runs away to infinity. Focusing closer means
  physically extending the lens, and below `s_o = f` no real image forms at all.
- The transverse magnification is `m = -s_i / s_o`, negative because a camera
  image is inverted. `|m| = 1` is life size (1:1 macro).

### 1.2 Field of view is the sensor, not the lens

The lens has no field of view on its own -- it has an image circle. What gets
recorded is whatever the sensor cuts out of it:

```
FOV = 2 atan(sensor_extent / 2 s_i)
```

Using `s_i` rather than *f* means the (slightly narrower) field at close focus
falls out of the same expression that gives the familiar `2 atan(d/2f)` at
infinity.

**Crop factor** is the ratio of the full-frame diagonal (43.27 mm) to this
sensor's diagonal, and the **35 mm equivalent focal length** is the focal length
multiplied by it. This is a framing statement only -- see §1.5.

### 1.3 Circle of confusion and depth of field

An out-of-focus point projects a blur disc. Depth of field is the range of
object distances over which that disc stays below an agreed threshold, the
**circle of confusion** (CoC). Two thresholds are in common use and the demo
offers both:

- **Print criterion**: `CoC = sensor_diagonal / 1440`. What a viewer will not
  resolve in a standard-sized print at normal viewing distance.
- **Pixel criterion**: two pixel pitches. Far stricter on a modern sensor,
  which is why depth of field shrinks the moment you pixel-peep.

From the CoC *c*, aperture *N* and focal length *f*:

```
H     = f^2 / (N c) + f                      (hyperfocal distance)
near  = s (H - f) / (H + s - 2f)
far   = s (H - f) / (H - s)     -> infinity once s >= H
```

Focus at *H* and everything from *H*/2 to infinity is acceptably sharp. That is
the whole of hyperfocal focusing.

### 1.4 The aperture trade-off

Stopping down shrinks the defocus blur disc but grows the Airy disk
(`2.44 N lambda`). Added in quadrature, the total has a minimum:

```
blur_total(N) = sqrt( defocus(N)^2 + (2.44 N lambda)^2 )
```

Past that optimum, stopping down further makes the image worse. This is the
quantitative version of the folklore that lenses go soft past f/11, and it is
the first place in the course where diffraction appears.

### 1.5 Equivalence

Two systems frame identically when `f / crop_factor` matches. They do **not**
have the same depth of field, because depth of field depends on the physical
aperture diameter `f/N`, and halving *f* at fixed *N* halves that diameter.
Micro Four Thirds at 25 mm f/2.8 frames like full frame at 50 mm f/2.8 and has
roughly twice the depth of field. Crop factor scales focal length; it does not
scale the f-number.

### 1.6 Close focus costs light

At magnification *m* the lens is extended, so the same aperture subtends a
smaller solid angle at the sensor:

```
bellows factor = (1 + |m|)^2        N_eff = N (1 + |m|)
```

At 1:1 that is 4x -- two stops. The image-generation pipeline applies the same
`(1+m)^2` term when converting radiance to irradiance at close focus, so this is
not a teaching simplification.

### 1.7 What a real lens adds

Two departures from the ideal model, both geometric rather than blur:

- **Natural (cos^4) vignetting.** Even a mechanically perfect lens loses corner
  light as the fourth power of the cosine of the chief-ray angle. On a wide lens
  over a large sensor that exceeds a stop before any mechanical vignetting.
- **Brown-Conrady distortion.** Straight lines map to curves:
  `r_d = r_u (1 + k1 r^2 + k2 r^4)` plus tangential `p1`, `p2` terms. `k1 < 0`
  bows lines outward (barrel), `k1 > 0` inward (pincushion).

---

## 2. UI map

| Panel / control | What it shows |
| --- | --- |
| Thin-lens construction | The three standard construction rays, object and image arrows, focal points. Schematic: object and image sides are scaled independently so both fit |
| Subject footprint | The rectangle of the world the frame covers at the focus distance, in metres |
| FOV vs focal length | Diagonal field of view swept across focal length, for the current format |
| Depth of field | Near and far limits against focus distance, log-log, with the hyperfocal distance marked |
| Defocus blur | Blur-disc diameter against where the object actually is, with the CoC threshold drawn in |
| Aperture trade-off | Defocus blur and diffraction blur against f-number, with their quadrature sum and its minimum |
| cos^4 falloff | Relative illumination centre to corner, plus a frame-shaped brightness preview |
| Distortion grid | A straight-line grid pushed through the forward Brown-Conrady model |
| Status line | Format, crop factor, megapixels, equivalent focal length, FOV, magnification, bellows factor, CoC, hyperfocal, DoF limits, corner falloff, distortion percentage |

**Core sliders:** focal length, f-number, focus distance (log scale), sensor format.
**Advanced:** pixel pitch, pixel-level CoC toggle, subject height, distortion `k1` and `k2`.
**Camera recipe dropdown:** loads focal length, f-number, pixel pitch and the distortion
coefficients from a real `config/camera_recipes/*.yaml`.

---

## 3. Guided walkthrough (lecture path)

### Experiment A -- The normal lens and the crop factor (~8 min)

1. Load **The 'normal' lens: 50 mm on full frame**. Read the diagonal field of
   view: about 47 degrees, near the sensor diagonal.
2. Change **only** the sensor format to APS-C. The lens has not moved, but the
   field narrows and the status line reports a 75 mm equivalent.
3. Switch to the phone format and watch the same 50 mm become a long telephoto.

**Ask the room:** "Which of focal length, field of view and equivalent focal
length is a property of the lens alone?"

### Experiment B -- Where depth of field actually comes from (~10 min)

1. Load **Portrait: long lens, wide aperture, shallow depth of field**.
2. Note the total depth of field in the status line -- a few centimetres at
   85 mm f/1.8 on a head-and-shoulders subject.
3. Stop down one stop at a time and watch the near and far limits pull apart.
   The blur plot's threshold line is the CoC; the DoF limits are exactly where
   the blur curve crosses it.
4. Switch the CoC criterion to pixel-level. The depth of field roughly halves
   without anything physical changing.

### Experiment C -- Hyperfocal focusing and the diffraction wall (~10 min)

1. Load **Landscape: hyperfocal focusing and the diffraction wall**.
2. On the depth-of-field plot, find where the far limit runs vertical. That
   focus distance is the hyperfocal distance, and the status line agrees.
3. Move to the aperture trade-off plot and read the optimum f-number.
4. Push the aperture to f/22 and confirm the total blur is *larger* than at the
   optimum. Stopping down bought depth of field and spent resolution.

### Experiment D -- Equivalence, done honestly (~10 min)

1. Load **Equivalence: same framing, different depth of field** (Micro Four
   Thirds, 25 mm, f/2.8). Record the field of view and the total depth of field.
2. Switch to full frame and set the focal length to 50 mm, leaving f/2.8 alone.
3. The field of view matches. The depth of field does not -- it is roughly half.
4. Now find the f-number on full frame that restores the original depth of
   field. It should land near f/5.6: the crop factor, applied to the aperture.

### Experiment E -- Macro and the two stops you did not budget for (~8 min)

1. Load **Macro at 1:1: magnification costs two stops**.
2. Confirm the magnification is 1.0 and the image distance equals the object
   distance -- the only symmetric case in the set.
3. Read the bellows factor (4x) and the effective f-number: a marked f/2.8
   behaves like f/5.6. That is two stops of exposure the meter has to find.

### Experiment F -- What the lens adds for free (~8 min)

1. Load **Natural vignetting: cos^4 corner falloff** at 16 mm. Read the corner
   falloff in stops and look at the brightness preview.
2. Sweep the focal length to 85 mm and watch the falloff flatten. Long lenses
   have small chief-ray angles, so cos^4 barely bites.
3. Load **Wide-angle barrel distortion** and read the corner distortion
   percentage. Flip `k1` positive to turn barrel into pincushion.

---

## 4. Self-paced lab sheet

| Task | Measurement | Your value |
| --- | --- | --- |
| 50 mm, full frame: diagonal FOV | degrees | |
| Same lens, Micro Four Thirds: diagonal FOV and equivalent focal length | deg / mm | |
| 85 mm f/1.8 at 1.5 m, print CoC: total depth of field | mm | |
| Same, pixel CoC at 5.94 um: total depth of field | mm | |
| 24 mm f/11, print CoC: hyperfocal distance | m | |
| Same: near limit when focused at H | m (expect H/2) | |
| 24 mm focused at 5 m: optimum f-number from the trade-off plot | f/N | |
| 100 mm at 1:1: effective f-number when marked f/2.8 | f/N | |
| 16 mm on full frame: corner falloff | stops | |
| `k1 = -0.18`, `k2 = 0.04`: corner distortion | % | |

---

## 5. Check your understanding

1. A lens is engraved "50 mm". What physical distance is that, and under what
   condition?
2. Why does the demo compute field of view from the image distance rather than
   the focal length? When does the difference matter?
3. Two cameras frame a subject identically and both are set to f/4. One has
   twice the depth of field. What differs, and why is it not the f-number?
4. Why does the aperture trade-off curve have a minimum rather than falling
   monotonically as you stop down?
5. Depth of field is a threshold on a continuous quantity. Name the quantity and
   the threshold, and explain why two photographers can disagree about the depth
   of field of the same photograph without either being wrong.
6. cos^4 falloff and mechanical vignetting both darken corners. How would you
   tell them apart from images alone?

---

## 6. Bridge to the next tutorial

Everything above treats the lens as perfect: one object point, one image point.
Real lenses spread that point into a disc even in perfect focus, from
diffraction at the aperture and from residual aberration in the glass. Tutorial
02 picks the story up exactly there, at the point-spread function -- and the
Airy disk you met on the aperture trade-off plot is where it starts.
