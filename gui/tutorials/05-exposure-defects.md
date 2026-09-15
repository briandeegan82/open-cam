# Tutorial 05 -- Exposure and Sensor Defects

**Demo:** `opencam-gui demo exposure`
**Audience:** graduate / advanced undergraduate; Tutorial 04 assumed
**Goal:** connect the photographer's exposure triangle to the electron counts
Tutorial 04 plotted, establish what ISO actually does, and meet the defects that
sit underneath the Gaussian-and-Poisson model -- the ones no noise statistic
describes.

The photometric chain is `tools/sensor_radiometry.py`: the camera equation,
lux to photons via luminous efficacy, and the EV relations. Every defect comes
from `tools/apply_emva_noise.py` -- the same `apply_blooming`,
`apply_hot_stuck_pixel_model`, `ktc_sigma_e`, `row_column_fpn_offsets`,
`flicker_row_offsets` and ADC INL/DNL functions the render pipeline calls. The
demo drives them; it does not reimplement them.

---

## 1. Principles

### 1.1 From scene luminance to electrons

The chain has four links, and each is a plain multiplication:

```
E_image = (pi/4) T L R / N^2               lux at the sensor  (camera equation)
phi     = (E_image / eta) A_pixel / E_ph   photons per second per pixel
mu_e    = phi * t_int * QE                 electrons collected
```

*L* is scene luminance (cd/m^2), *N* the f-number, *T* lens transmission, *R*
relative illumination (the cos^4 term from Tutorial 01), `A_pixel` the pixel
area times fill factor, `t_int` the integration time and QE the quantum
efficiency.

The middle line is the one worth pausing on. Lux is a *photometric* unit -- the
eye's response is already folded into it -- so getting back to photons needs
`eta`, the luminous efficacy of the particular spectrum (lm/W, about 240 for
daylight), not a universal constant. `E_ph = hc/lambda` then converts watts to
photons per second.

The two facts worth extracting: electrons are **linear in time** and **inverse
square in f-number**. Everything the exposure triangle says follows from that.

### 1.2 Exposure value

EV compresses aperture and shutter into one number:

```
EV = log2(N^2 / t)
```

f/1 for one second is EV 0 by definition, and each stop adds one. A line of
constant EV in the (N, t) plane is the reciprocity trade: every point on it
delivers the same electrons, and the sensor genuinely cannot tell which control
you moved. Only motion blur and depth of field can.

Scene brightness gets its own scale:

```
EV100 = log2(L * S / K)        S = 100, K = 12.5 (the metering constant)
```

**Sunny 16** -- ISO 100, f/16, shutter = 1/ISO in direct sun -- is the statement
that these two agree for a scene around 4000 cd/m^2. The demo checks it to
within a third of a stop.

A caution the demo makes visible: the engraved stops are rounded. f/2.8 is
really 2.828 and 1/125 is really 1/128, so a "one stop" move is a couple of
percent off. This is one reason cameras meter rather than compute.

### 1.3 ISO is gain, not sensitivity

Raising ISO does not collect one extra photon. It turns up an analog amplifier
sitting after the photodiode, which rescales the ADC path:

```
K_eff        = K / G          (electrons per DN, so a code is worth fewer electrons)
full_well_eff = full_well / G  (clipping arrives proportionally sooner)
```

So every stop of ISO is a stop of highlight headroom spent, and the electron
count -- the thing that actually sets shot noise -- never moves. Dynamic range
falls monotonically as you push.

> **Note on the amplifier noise model.** `apply_emva_noise` treats `sigma_amp_e`
> as a floor that scales with the gain,
> `sigma_read = sqrt(sigma_d^2 + (sigma_amp * G)^2)`, so in this pipeline the
> electron-referred read noise *rises* with ISO. Many real sensors behave the
> other way (input-referred read noise falls with gain until it floors out,
> which is what "ISO invariance" discussions are about). The demo shows what the
> render path does; treat the direction of this particular curve as a property of
> the model rather than a measured fact about silicon.

### 1.4 Reading the photon transfer curve backwards

Tutorial 04 plotted variance against signal. Here the same curve is used as a
map: given an exposure, which regime are you in?

- **Read-noise limited**: `sqrt(mu_e) < sigma_read`, i.e. `mu_e < sigma_read^2`.
  For a 2.3 e- sensor that is about five electrons. The branch is far narrower
  than students expect -- one stop of aperture is usually enough to leave it.
- **Shot-noise limited**: the normal case. SNR goes as `sqrt(mu_e)`, so four
  times the light is 6 dB.
- **Clipped**: `mu_e >= full_well_eff`. Information already destroyed; no
  processing recovers it.

A well-designed sensor has its full well landing near the top ADC code. A well
that overruns the top code moves the clipping point somewhere the exposure
readout cannot explain; one that falls short wastes bits.

### 1.5 The defects underneath

Everything above is temporal noise with a clean statistical description. These
are not:

| Defect | Physics | Signature |
| --- | --- | --- |
| **Blooming** | Charge past full well spills to cardinal neighbours | A blown highlight grows beyond the pixels that saw the light |
| **Hot / stuck pixels** | Dark current varies enormously pixel to pixel | Fixed bright specks, same place every frame -- which is why dark-frame subtraction works |
| **kTC reset noise** | Resetting the sense node leaves `sqrt(kTC)/q` electrons of random charge | A raised noise floor, growing only as `sqrt(T)`. Cancelled entirely by correlated double sampling, which is why every modern CMOS sensor does CDS |
| **Row / column FPN** | Per-row and per-column readout offsets | Banding that survives averaging along one axis and vanishes along the other |
| **1/f flicker** | Amplifier drift with a pink spectrum, read out row by row | Row banding too, but *smooth*: neighbouring rows track each other |
| **ADC INL** | Integral non-linearity, a quadratic bow across the range | A tone-curve error, peaking mid-scale, zero at both endpoints |
| **ADC DNL** | Per-code width errors | Fixed per-code jitter. Unlike read noise it does **not** average away over frames |

The diagnostic tool for the banding family is the row and column profile:
collapsing a row knocks its read noise down by `sqrt(width)` while leaving a row
offset untouched, so banding far below the per-pixel noise floor stands out. The
demo takes the **median** rather than the mean so a blown highlight crossing a
few rows does not masquerade as banding.

---

## 2. UI map

| Panel / control | What it shows |
| --- | --- |
| Iso-exposure lines | The (f-number, shutter) nomogram, one line per EV, with the current setting marked |
| ISO gain: headroom and noise floor | Effective full well and read noise against ISO gain |
| Dynamic range and SNR against ISO | Both derived from the same sweep |
| Flat field with one blown highlight | The shared base frame with whichever defects are enabled |
| Row profile / Column profile | Median DN per row and per column -- the banding diagnostic |
| ADC transfer-curve deviation | Departure from the ideal ramp, in LSB: the smooth INL bow plus the DNL jitter |
| Status line | EV, scene EV100, image-plane lux, signal electrons, effective well and K, shot / read / total noise, SNR, saturation fraction, and the regime label |

**Core sliders:** scene luminance, f-number, shutter denominator, ISO gain, plus one
checkbox per defect.
**Advanced:** quantum efficiency, pixel pitch, amplifier noise at 1x, sensor temperature,
row and column FPN sigma, flicker sigma, ADC INL fraction, ADC DNL sigma.
**Camera recipe dropdown:** loads gain, full well, read noise, black level, bit depth,
temperature and pixel pitch from a real recipe.

Every defect is off unless ticked, and the random seeds are fixed, so toggling
one changes only that one's contribution. That is what makes the signatures
comparable.

---

## 3. Guided walkthrough (lecture path)

### Experiment A -- Sunny 16 and reciprocity (~8 min)

1. Load **Sunny 16: the reference exposure**. Compare the EV readout against the
   scene EV100 -- they should agree to within a third of a stop.
2. Load **Reciprocity: the same exposure three ways**. Slide the f-number back
   and forth along the iso-exposure line and watch the signal-electron readout
   stay put.
3. Note where the rounding bites: f/2.8 instead of 4/sqrt(2) is about 2% of
   light, which is why the exposure is metered rather than computed.

**Ask the room:** "If the sensor cannot tell which control you moved, what can?"

### Experiment B -- Finding the read-noise branch (~10 min)

1. Load **Underexposed: down in the read noise**. Read the regime label and the
   signal electrons -- single digits.
2. Confirm the rule: the label says read-noise limited exactly while the signal
   is below `sigma_read^2`.
3. Open the aperture one stop. It flips to shot-noise limited. The branch really
   is that narrow.
4. Now sweep the scene luminance from 0.01 to 100000 cd/m^2 and watch the label
   pass through all three regimes exactly once.

### Experiment C -- ISO is gain (~10 min)

1. Load **ISO is gain, not sensitivity**.
2. Push the ISO gain from 1x to 8x. The signal-electron readout does not move --
   not by one electron.
3. What does move: effective full well (down by 8x), dynamic range (down), and
   the noise floor.
4. Raise the scene luminance until the frame clips, then push ISO. Clipping
   arrives sooner at every step. Every stop of ISO is a stop of headroom.

### Experiment D -- Blooming (~6 min)

1. Load **Clipped: blooming out of a saturated highlight**, then toggle blooming
   off and on against the same frame.
2. The highlight's footprint grows; its centre stays clipped either way.
3. Note what did *not* happen: no pixel exceeds full well. Blooming moves charge,
   it does not manufacture it.

### Experiment E -- Fixed patterns and dark-frame subtraction (~8 min)

1. Load **Long exposure, hot sensor**. Find the bright specks.
2. Re-render (nudge any slider and back). The specks do not move. That is the
   entire justification for dark-frame subtraction.
3. Raise the temperature to 60 C and note the specks brighten -- dark current
   roughly doubling every 6 degrees.

### Experiment F -- Telling the three banding artefacts apart (~10 min)

1. Load **Readout banding: row, column and 1/f**, then enable exactly one defect
   at a time and watch the two profile plots.
2. Row FPN moves the row profile and leaves the column profile alone. Column FPN
   does the reverse. This is unambiguous, and the image itself barely changes.
3. Enable flicker on its own. It also bands rows -- but smoothly. Neighbouring
   rows track each other because the spectrum is 1/f and a rolling shutter reads
   rows in time order.
4. Note that the profiles reveal banding far below the per-pixel noise floor,
   because collapsing a row averages read noise down by `sqrt(width)`.

### Experiment G -- The converter's own errors (~8 min)

1. Load **ADC INL and DNL** and work from the transfer-curve deviation plot, not
   the image.
2. INL alone: a smooth bow, zero at both endpoints, peaking mid-scale. For a
   nominal fraction *f* the bow peaks at `f/4` of full scale, because `x(1-x)`
   maxes at a quarter.
3. DNL alone: jitter that changes sign constantly. Re-render and confirm it is
   *identical*. That is the point -- DNL is fixed per code, so stacking a hundred
   frames removes read noise and leaves DNL exactly where it was.

### Experiment H -- kTC and why CDS exists (~6 min)

1. Load **kTC reset noise: the sensor without CDS** and watch the noise floor
   jump.
2. Swing the temperature from -20 C to 80 C. The noise grows as `sqrt(T)` in
   kelvin, so 100 degrees buys only about 20% -- cooling is a poor defence.
3. Conclude: this is why CDS is universal and why this switch is normally off.

---

## 4. Self-paced lab sheet

| Task | Measurement | Your value |
| --- | --- | --- |
| f/16 at 1/128 s: EV | dimensionless (expect 15) | |
| 4000 cd/m^2: scene EV100 | dimensionless | |
| f/4, 1/500, 100 cd/m^2: signal electrons | e- | |
| Same, one stop of aperture and one of shutter traded: signal electrons | e- | |
| `sigma_read` = 2.3 e-: signal at the read/shot crossover | e- | |
| ISO 1x -> 8x: signal electrons before and after | e-, e- | |
| Same: effective full well before and after | e-, e- | |
| Nikon Z6 recipe: full well expressed in DN, vs the top ADC code | DN, DN | |
| INL fraction 0.04 on 12 bits: peak deviation | LSB (expect ~41) | |
| Row FPN 25 e- only: row profile std / column profile std | ratio | |
| kTC at -20 C and 80 C: ratio | dimensionless (expect ~1.18) | |

---

## 5. Check your understanding

1. Two exposures sit on the same EV line. What is identical about them, and what
   is not?
2. Why is "ISO 3200" not a sensitivity? What physical quantity does it change,
   and what does it leave alone?
3. A colleague says "underexposing and lifting in post is the same as exposing
   correctly". Under what condition is that nearly true, and where does it fail?
4. The read-noise-limited branch of the PTC covers only a handful of electrons on
   a modern sensor. Why do people nonetheless talk about it so much?
5. Row FPN and 1/f flicker both produce horizontal banding. Give two ways to tell
   them apart from a single frame.
6. You stack 100 frames of a static scene. Which defects in §1.5 are reduced,
   which are unchanged, and why?
7. kTC noise scales as `sqrt(kTC)`. Given that, would you rather halve the
   temperature or halve the sense-node capacitance? What does CDS do that beats
   both?
8. A sensor's full well corresponds to 9400 DN on a 12-bit converter. What goes
   wrong, and what would you change?

---

## 6. Bridge to the next tutorial

You now have electrons, and digital numbers with every defect the silicon adds.
What you do not have is colour: every pixel so far has been a single number. The
next tutorial puts a colour filter array in front of the sensor, which throws
away two thirds of the colour information, and then walks the ISP stages that
guess it back and turn camera RGB into a colour anyone else can interpret.
