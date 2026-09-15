# Tutorial 06 -- Colour and the ISP

**Demo:** `opencam-gui demo isp`
**Audience:** graduate / advanced undergraduate; Tutorials 04 and 05 assumed
**Goal:** understand why raw camera RGB is not a colour space, what each ISP
stage costs and buys, and why some colour errors survive every stage no matter
how well you fit the matrix.

The colour science is `tools/colour_science.py`: the CIE 1931 2-degree observer,
spectral integration, XYZ/sRGB/Lab, Bradford chromatic adaptation and both
delta-E metrics. The pipeline stages are `tools/apply_emva_noise.py`'s own
`bayer_sample_rgb`, `bilinear_demosaic`, `malvar_demosaic`, `gray_world_gains`,
`white_patch_gains`, `fit_ccm_lstsq`, `apply_ccm` and `linear_to_srgb`.

The scene is synthesised from the real X-Rite ColorChecker reflectance spectra
rather than from RGB values. That is what makes the illuminant sweep meaningful:
under F11 the patches genuinely change colour because the light genuinely
changes, not because a constant was swapped.

---

## 1. Principles

### 1.1 Colour is an integral, three times

A surface has a reflectance spectrum `rho(lambda)`, a lamp has a power spectrum
`S(lambda)`, and a detector has a response `R(lambda)`. What the detector
records is their product, integrated:

```
response = integral  S(lambda) rho(lambda) R(lambda) dlambda
```

Change *R* and you change the answer. For the human eye, *R* is the three CIE
1931 colour-matching functions and the answer is called **XYZ**. For a camera,
*R* is the three QE curves and the answer is **camera RGB**. Neither is more
correct; they are answers to different questions.

### 1.2 Metamerism and the Luther condition

Two surfaces with different spectra can produce the *same* three numbers. That
is metamerism, and it is unavoidable: you are projecting an infinite-dimensional
spectrum onto three numbers.

The trouble is that a camera and an eye project onto **different** three
dimensions. If the camera's QE curves were an exact linear combination of the
CMFs -- the **Luther condition** -- then a single 3x3 matrix would convert camera
RGB to XYZ perfectly, and any two spectra that matched for the eye would match
for the camera. Real QE curves are not: they overlap more than the CMFs do and
have the wrong shape in the blue.

The demo reports the **Luther error**, the residual of the best least-squares fit
of the CMFs by the QE curves. It is the floor on what any 3x3 can achieve, and it
is why colour correction is a fit rather than a conversion.

### 1.3 Delta E: measuring a colour error

RGB distances are meaningless as error metrics because the space is not
perceptually uniform. CIELAB was designed so that Euclidean distance roughly
tracks perceived difference:

- **CIE76** is that Euclidean distance. Simple, and wrong about saturated blues
  by roughly a factor of two.
- **CIEDE2000** adds lightness, chroma and hue weightings plus a blue-region
  rotation term. It is what the demo reports.

Rules of thumb: delta-E around 1 is the just-noticeable difference for adjacent
patches, under 2 is good camera colour, and double digits is visibly wrong.

**Exposure is not colour.** Being a third of a stop dark would register as a
large delta-E on every patch and swamp what you are trying to measure, so the
demo normalises exposure on the neutral ladder before computing delta-E. Keeping
the two separate is what lets a colour error be read as a colour error.

### 1.4 The CFA throws away two thirds of the colour

A Bayer array puts one filter over each pixel, so each pixel records one channel
and the other two have to be interpolated:

- **Bilinear** averages the nearest same-colour neighbours. It ignores the other
  channels entirely.
- **Malvar-He-Cutler** adds a gradient correction from the channel that *was*
  measured at that pixel. It exploits the fact that in natural images luminance
  carries the fine detail while chroma varies slowly.

That last clause is a statement about images, not about arithmetic. On a target
with fine detail at slowly varying hue, Malvar beats bilinear roughly two to one
on edges. On saturated, independently varying channels the assumption is false
and Malvar *loses*. The demo shows both regimes, because the lesson is about
when a prior holds.

### 1.5 White balance: gains, and their limits

An illuminant scales the three channels unequally. Two classic estimators:

- **Gray world** assumes the scene averages to grey and divides each channel by
  its own mean.
- **White patch** assumes the brightest thing is white and divides by the maximum.

On a ColorChecker, gray world fails in a specific and instructive way: the chart
does *not* average to grey, so the estimate over-corrects and leaves a blue cast
on the neutral ladder. White patch has an actual white square to lock onto and
gets it right.

Either way, a per-channel gain can only fix *neutrals*. It cannot reshape a
channel's spectral response, so the saturated patches stay wrong. That is what
the next stage is for.

### 1.6 The colour correction matrix

A 3x3 fitted by least squares against the colorimetric reference:

```
[R' G' B']^T = M [R G B]^T
```

A well-fitted CCM has a dominant diagonal with mostly negative off-diagonals,
because camera QE curves overlap more than the CMFs do and the correction is to
subtract the neighbouring channels -- effectively sharpening each channel's
response. That sharpening amplifies noise, which is the real cost of an
aggressive CCM.

What it cannot do is beat the Luther error. Whatever residual delta-E survives a
correctly fitted 3x3 is the camera failing to be a linear transform of the
observer, and no 3x3 will remove it.

### 1.7 sRGB encoding

The last stage is not a colour transform but a coding one: the sRGB transfer
function spends code values where the eye can tell them apart, which is in the
shadows. Linear light shown directly looks far too dark down there. This is
gamma, and it is about bit allocation rather than about display physics.

### 1.8 Why the illuminant matters so much

The demo carries 18 illuminant SPDs, and they are not interchangeable:

- **D65 / D50** -- smooth daylight. The easy case.
- **A** -- tungsten at 2856 K, overwhelmingly red. The blue channel is starved,
  so white balance must amplify it hard, and amplifying a weak channel amplifies
  its noise with it.
- **F11** -- three narrow mercury lines. Surfaces that match under daylight come
  apart, because the camera and the eye sample those spikes with different curves.
- **RGB LED** -- three narrow peaks that look white to the eye while carrying
  almost no power between them. Colour rendering collapses, and no gain or matrix
  recovers information the light never delivered.

---

## 2. UI map

| Panel / control | What it shows |
| --- | --- |
| Stage strip | Every stage in order, with the image as it leaves that stage and a note on what it did |
| Colorimetric reference | What the chart should look like, adapted to D65 |
| Fitted colour correction matrix | The actual 3x3, live |
| Per-patch colour error | CIEDE2000 for all 24 patches, with the worst named |
| Demosaic comparison | Bilinear and Malvar against known ground truth, with a difference map and RMSE overall and on edges |
| Illuminant, reflectance and QE | The three curves that decide a patch's colour, each normalised |
| Their product | What each channel actually collects -- a channel only responds where all three overlap |
| Status line | Illuminant and its CCT, Luther error, white-balance gains, neutral cast, mean and max delta-E |

**Core controls:** illuminant, one checkbox per ISP stage, demosaic method, white balance method.
**Advanced:** CFA pattern, spectral-overlay patch index.
**Camera recipe dropdown:** loads the camera's own QE curves, which changes the Luther error
and therefore the achievable colour accuracy.

---

## 3. Guided walkthrough (lecture path)

### Experiment A -- Raw is not a colour space (~8 min)

1. Load **Raw is not a colour space**: mosaic, demosaic and sRGB only.
2. Read the mean delta-E. It should be in the teens, and the image should look
   obviously wrong next to the colorimetric reference.
3. Emphasise that nothing is broken. These are real measurements -- just
   measurements in the camera's own basis.
4. Turn white balance on, then the CCM, reading delta-E after each. It should
   fall from the teens to about one.

### Experiment B -- What the CFA costs (~8 min)

1. Load **The CFA alone: two thirds of the data thrown away** and look at the
   mosaic stage: every pixel is now one number.
2. Turn demosaic on and off and watch the colour reappear -- inferred, not
   measured.
3. Move to the demosaic tab. Compare the two error maps and note where the
   differences live: on edges, which is exactly where the eye looks.

### Experiment C -- When a prior holds and when it does not (~10 min)

1. Still on the demosaic tab, read the four RMSE numbers. Malvar should win
   overall and win by more on edges.
2. Explain why: fine luminance detail at slowly varying hue is the assumption,
   and the test target is built to have it.
3. The counter-case is worth stating out loud even though the demo's target does
   not show it -- on saturated, independently varying channels the assumption
   fails and bilinear wins. `gui/tests/test_isp_engine.py` pins both directions.

**Ask the room:** "Is Malvar a better algorithm, or a better-matched prior?"

### Experiment D -- Two white-balance estimators, one chart (~10 min)

1. Load **Gray world versus white patch** and watch the *neutral cast* readout
   rather than the image.
2. Gray world leaves the neutrals visibly blue -- the estimate over-corrects
   because a ColorChecker does not average to grey.
3. Switch to white patch. The cast goes to (1, 1, 1).
4. Note what white balance did *not* fix: with the CCM still off, the saturated
   patches are as wrong as before. Gains fix neutrals, nothing else.

### Experiment E -- The matrix, and its ceiling (~10 min)

1. Load **White balanced but uncorrected**: delta-E around six with the neutral
   ladder already correct.
2. Turn the CCM on and read the fitted 3x3. Point out the dominant diagonal and
   the negative off-diagonals, and connect the sign to overlapping QE curves.
3. Read the Luther error in the status line, then the residual max delta-E. The
   error that survives is the camera failing the Luther condition.
4. Load a different camera recipe and watch the Luther error and the achievable
   delta-E move together.

### Experiment F -- Tungsten and the starved channel (~8 min)

1. Load **Tungsten: illuminant A at 2856 K**. Check the CCT readout.
2. Compare the blue white-balance gain against the D65 scenario.
3. Look at the illuminant SPD to see where that gain comes from: there is simply
   very little blue power in the lamp.
4. Draw the practical conclusion: that gain multiplies the blue channel's noise
   too, which is why tungsten shots are noisy in the shadows.

### Experiment G -- Spiky illuminants and metamerism (~10 min)

1. Load **F11: a spiky illuminant and metamerism** and go to the spectra tab.
2. Where the lamp has no power, neither the eye nor the sensor can see the
   surface at all -- the product curve is flat there regardless of reflectance.
3. Load **RGB LED: the hardest illuminant of all** and compare delta-E against
   D65 with identical settings. It should be the worst in the set even with the
   full pipeline running.
4. Make the point that this is an information problem, not a processing one.

### Experiment H -- Alternative CFAs: RCCB and CMY (~8 min)

1. Load **RCCB: cyan in place of a green**. The camera recipe dropdown should
   show `default_rccb`, and the green QE curve on the spectra tab is now cyan.
2. Compare the Luther error against the D65 Bayer scenario. A worse spectral
   basis raises the floor on what any 3x3 can achieve.
3. Load **CMY: complementary filters**. Every channel is now broad, so more
   photons arrive -- and the curves overlap more.
4. Read delta-E. More light did not buy more colour accuracy, because the
   Luther condition is about the *shape* of the curves, not their height.

---

## 4. Self-paced lab sheet

| Task | Measurement | Your value |
| --- | --- | --- |
| D65, mosaic + demosaic only: mean delta-E | dimensionless | |
| Same plus white patch: mean delta-E | dimensionless | |
| Same plus CCM: mean delta-E | dimensionless | |
| Full pipeline: worst patch name and its delta-E | name, value | |
| Gray world on D65: neutral cast (R, G, B) | triple | |
| White patch on D65: neutral cast (R, G, B) | triple | |
| Illuminant A: blue white-balance gain, vs D65 | ratio | |
| D65 vs A vs F11 vs LED_RGB1, full pipeline: mean delta-E | four values | |
| Camera's Luther error, and its full-pipeline mean delta-E | two values | |
| RCCB vs Bayer: Luther error and mean delta-E | four values | |
| CMY vs Bayer: Luther error and mean delta-E | four values | |
| Demosaic tab: bilinear and Malvar edge RMSE | two values | |
| CCM diagonal and off-diagonal signs | describe | |

---

## 5. Check your understanding

1. Two paint chips look identical to you under daylight and different under a
   fluorescent lamp. Nothing about the paint changed. What did?
2. Why can a 3x3 matrix not fix a camera that fails the Luther condition, no
   matter how it is fitted?
3. Gray world and white patch disagree on this chart. Construct a scene where
   the opposite happens and gray world is the better estimator.
4. White balance fixes the neutrals and leaves the saturated patches wrong.
   Explain in terms of what a per-channel gain can and cannot do to a spectral
   response.
5. A CCM with large negative off-diagonals gives excellent chart accuracy and
   noisy images. Why are those the same fact?
6. Why does the demo normalise exposure on the neutral patches before computing
   delta-E? What would the numbers look like if it did not?
7. Malvar beats bilinear on natural images and loses on some synthetic ones.
   What does that tell you about benchmarking demosaic algorithms?
8. An RGB LED lamp has a colour rendering index in the 20s but looks white. How
   are both of those true at once?
9. A CMY sensor collects more photons than a Bayer sensor of the same well
   capacity. Why might its colour still be worse?

---

## 6. Bridge to the next tutorial

Six tutorials, six pieces: geometry places the image, optics blurs it, MTF
measures the blur, the sensor adds noise, exposure and defects distort the
electrons, and the ISP turns them into colour. The last tutorial runs all of it
at once. Pick a scene, a light source, an illuminance and a real camera, and
generate an actual image through the same `tools/*.py` scripts you have been
calling one at a time -- with every parameter you have met along the way now
acting on the same frame.
