# Tutorial 03 -- Resolution and MTF

**Demo:** `opencam-gui demo mtf`
**Audience:** graduate / advanced undergraduate; Tutorials 01 and 02 assumed
**Goal:** turn the point-spread function from a picture into a number. Measure
MTF from a slanted edge the way ISO 12233 does, compare it against diffraction
theory, and find out what happens to detail finer than the pixel grid.

The measurement chain is `tools/sfr_analysis.py` -- edge angle estimation,
4x-oversampled edge spread function, line spread function, FFT, MTF50. The blur
being measured comes from `tools/apply_spectral_psf.py`, the same functions
Tutorial 02 explored, so the number you read here describes the optics model the
render pipeline actually uses.

---

## 1. Principles

### 1.1 Why MTF and not "resolution"

"Resolves 60 line pairs per millimetre" is a threshold on a continuum, and the
threshold is someone's opinion. The **modulation transfer function** reports the
whole curve instead: for a sinusoidal pattern at spatial frequency *f*, MTF(*f*)
is the ratio of output contrast to input contrast. MTF(0) = 1 by construction,
and it falls monotonically for any physically realisable system.

Two conventional readouts:

- **MTF50**, the frequency where contrast has fallen to half. It correlates well
  with perceived sharpness.
- **MTF at Nyquist** (0.5 cycles/pixel), which predicts aliasing rather than
  sharpness -- see §1.5.

Units matter. **Cycles per pixel** describes the lens-and-sensor pair; **cycles
per millimetre** describes the lens alone. The demo reports both, and the only
thing connecting them is the pixel pitch.

### 1.2 The slanted-edge method (ISO 12233)

A perfectly vertical edge gives you samples at exactly one phase within the
pixel, which is not enough to see detail finer than a pixel. Slanting the edge a
few degrees means each row crosses it at a slightly different sub-pixel offset.
Pooling the rows therefore *super-samples* the edge:

1. **Find the angle.** A per-row centroid of the gradient, fitted with a line.
2. **Project.** Re-reference every pixel to its perpendicular distance from that
   line and bin at 1/4-pixel spacing. This is the **edge spread function** (ESF).
3. **Differentiate.** `LSF = d(ESF)/dx` -- the **line spread function**, which is
   the PSF integrated along the edge direction.
4. **Transform.** `MTF = |FFT(LSF)|`, normalised so MTF(0) = 1.

The slant is a sampling trick, not a property of the system, so the answer must
not depend on the angle chosen. Experiment E checks exactly that.

One practical trap the implementation has to avoid: convolving an edge with
`mode="same"` zero-pads the border, leaving a second, artificial step at the ROI
boundary. A whole-row gradient centroid will happily lock onto it instead of the
real edge. The demo blurs an oversized image and crops back, and the edge
locator ignores the outer margin.

### 1.3 Diffraction-limited MTF

For a circular aperture, the diffraction-limited MTF has a closed form and a
hard cutoff:

```
MTF_diff(v) = (2/pi) (acos(v) - v sqrt(1 - v^2)),    v = f / f_cutoff
f_cutoff    = 1 / (lambda N)     cycles per mm
```

Above `f_cutoff` a diffraction-limited lens transmits *exactly zero* contrast.
At f/16 and 550 nm that is about 114 cycles/mm, regardless of what the sensor
behind it does. This is the only part of an MTF curve that is pure physics, and
it is the ceiling every real lens sits under.

### 1.4 The pixel is part of the system

A square pixel integrates light over its own area, which is a box filter, whose
transfer function is a sinc:

```
MTF_pixel(f) = |sinc(f * pitch)|
```

First zero at one cycle per pixel; at Nyquist it is `2/pi = 0.64`. The system
MTF is the product of the stages, so the sensor can only ever make the lens
worse:

```
MTF_system = MTF_diffraction * MTF_pixel
```

### 1.5 Nyquist, and what happens past it

Sampling at pitch *p* can represent frequencies up to `1/(2p)` -- 0.5 cycles per
pixel, the **Nyquist limit**. Content above it does not disappear. It *folds
back*, reappearing as a lower frequency that was never in the scene. On a
Siemens star this is unmistakable: toward the centre, where the spokes converge
past Nyquist, the fine spokes give way to coarse moire running the wrong way.

Note that the slanted-edge method measures the **pre-sampling** MTF: the slant
supplies sub-pixel phases, so the measurement legitimately reports frequencies
above Nyquist. Contrast still sitting above 50% at 0.5 cycles/pixel is not a
measurement error. It is precisely the condition for aliasing.

Two ways out, and both cost something:

- An **optical low-pass filter** blurs before sampling, destroying the offending
  frequencies. You lose real detail near Nyquist and gain an image that does not
  lie.
- **Diffraction** does the same job for free when the cutoff falls below Nyquist,
  which is the normal state of affairs on a phone camera.

---

## 2. UI map

| Panel / control | What it shows |
| --- | --- |
| Edge ROI under test | The slanted edge being measured, synthetic or loaded from `out/` |
| Edge spread function | The 4x-oversampled edge profile built from all rows |
| Line spread function | Its derivative -- the PSF along the edge normal |
| Modulation transfer function | The measured curve, the diffraction and pixel-aperture theory curves, their product, MTF50 and the Nyquist line |
| Siemens star (reference) | The target at full resolution |
| Point-sampled on a coarse grid | The same star sampled every N pixels, with the Nyquist radius marked |
| Status line | Edge angle, MTF50 in cycles/px and cycles/mm, MTF10, MTF at Nyquist, diffraction cutoff |

**Core controls:** edge source (synthetic / rendered), PSF mode, f-number, pixel pitch,
geometric aberration sigma, star spokes, sampling factor, OLPF prefilter sigma.
**Advanced:** edge slant angle.
**Camera recipe dropdown:** loads f-number, pixel pitch and PSF settings from a real recipe.

The synthetic edge source needs no PBRT build and has known ground truth. The
rendered source reads slanted-edge targets from `out/` if any exist -- generate
them with `tools/build_image_quality_targets.py`.

---

## 3. Guided walkthrough (lecture path)

### Experiment A -- Read the chain end to end (~8 min)

1. Load **A sharp system: MTF50 near Nyquist**.
2. Walk the three plots left to right: the ROI is an edge, the ESF is that edge
   averaged over all rows at 4x sampling, the LSF is its slope.
3. Note that the LSF is the PSF again -- the same shape Tutorial 02 plotted, now
   arrived at by measurement rather than by evaluating a formula.
4. Read MTF50 in both units. Only the cycles/mm figure changes when you move the
   pixel-pitch slider.

### Experiment B -- Blur is a number (~8 min)

1. Load **A soft lens: aberration-limited** and compare its LSF width against
   the sharp reference.
2. Sweep the geometric aberration sigma from 0.3 to 2.0 px and watch MTF50 fall
   roughly in proportion.
3. For a Gaussian blur of width sigma, theory says
   `MTF50 = sqrt(ln2 / 2) / (pi sigma) = 0.1874 / sigma` cycles/pixel. Check two
   slider positions against that.

**Ask the room:** "The picture of the blur and the number describing it are the
same fact. Which one would you put in a specification, and why?"

### Experiment C -- Diffraction is a hard ceiling (~10 min)

1. Load **Stopped down to f/16: diffraction sets the cutoff** (`airy_disk` mode,
   negligible aberration).
2. The measured curve should hug the theoretical diffraction curve down to the
   marked cutoff and reach zero there -- not before, not after.
3. Convert the cutoff to cycles/mm and check it against `1/(lambda N)` by hand.
4. Sweep the f-number from 4 to 16 and watch the cutoff march left. This is the
   diffraction wall from Tutorial 01, now measured rather than predicted.

### Experiment D -- Why phones never stop down (~8 min)

1. Load **Tiny pixels: why phones never stop down** (f/11, 1.4 um pixel).
2. The diffraction cutoff now sits *below* the Nyquist line. The lens gives up
   before the sensor does.
3. Conclude the trade explicitly: on this camera, stopping down for depth of
   field can only cost resolution the sensor was built to capture.

### Experiment E -- The slant is a trick, not a variable (~6 min)

1. Return to the sharp reference and open the advanced controls.
2. Step the edge slant through 3, 5, 8 and 12 degrees, recording MTF50 each time.
   The values should agree to within a few percent.
3. Now set the slant to 0. The method degrades: with every row sampling the same
   sub-pixel phase there is nothing left to super-sample with.

### Experiment F -- Past Nyquist, on the star (~10 min)

1. Switch to the **Aliasing past Nyquist** tab and load **Past Nyquist: aliasing
   invents detail**.
2. Find the Nyquist radius marker. Outside it the spokes are honest; inside, the
   sampled image shows coarse patterns that are not in the reference.
3. Load **The optical low-pass filter trade** and compare. The moire is gone; so
   is genuine detail near the marker.
4. Sweep the prefilter sigma from 0 to 2 px and find the smallest value that
   suppresses the moire. That is the OLPF design problem in one slider.

---

## 4. Self-paced lab sheet

| Task | Measurement | Your value |
| --- | --- | --- |
| Sharp reference: MTF50 | cycles/px | |
| Same, at 4.3 um pitch | cycles/mm | |
| Same optics, pitch changed to 1.4 um: MTF50 in cycles/px and cycles/mm | both | |
| Gaussian sigma = 1.5 px: measured MTF50 vs `0.1874/sigma` | % difference | |
| f/16, 550 nm: diffraction cutoff | cycles/mm (expect ~114) | |
| Same, expressed at 4.3 um pitch | cycles/px | |
| Pixel-aperture MTF at Nyquist | dimensionless (expect 0.64) | |
| f/11 at 1.4 um: cutoff above or below Nyquist? | circle one | |
| Slant 3 / 5 / 8 / 12 deg: spread in MTF50 | % | |
| Smallest prefilter sigma that kills the star moire | px | |

---

## 5. Check your understanding

1. Why does the slanted-edge method need a slant at all? What breaks at zero
   degrees?
2. The measured MTF reports contrast above 0.5 cycles/pixel. Is that a bug?
   Explain what the method is measuring.
3. A lens is quoted at "60 lp/mm". What extra number do you need before you can
   say whether it out-resolves a given camera?
4. Two systems have identical MTF50 but different MTF at Nyquist. Which will
   alias more, and which will look sharper?
5. Why is the system MTF the *product* of the stage MTFs rather than, say, their
   quadrature sum the way blur sigmas combine?
6. You measure MTF from a rendered target and get a suspiciously low MTF50 that
   gets worse as you enlarge the ROI. What artefact should you suspect first?
7. An OLPF and a small aperture both suppress aliasing. Give one reason a
   manufacturer would choose the OLPF anyway.

---

## 6. Bridge to the next tutorial

MTF describes what happens to *contrast*. It says nothing about whether the
contrast survives the noise sitting on top of it -- a 5% modulation is useless if
the noise floor is 8%. Tutorial 04 builds the other half: the EMVA1288 model of
how photons become electrons and electrons become noisy digital numbers, and the
photon transfer curve that tells you how much signal you need before contrast is
worth measuring.
