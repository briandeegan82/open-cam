# Wavelength-dependent carrier diffusion

`cfa.spatial_crosstalk` (`bayer.spatial_crosstalk` in recipes) has two models:

- `model: gaussian` (default, unchanged): a fixed Gaussian `sigma_pixels`, optionally per channel.
- `model: silicon_diffusion` (`tools/si_diffusion.py`): kernels derived from where each photon is
  absorbed in silicon and how far the carrier diffuses before it is collected.

```yaml
bayer:
  spatial_crosstalk:
    enabled: true
    model: silicon_diffusion
    architecture: bsi        # or fsi
    epi_thickness_um: 3.0
    depletion_depth_um: 2.5  # measured from the junction side
    radius_px: 4
```

The pixel pitch comes from `sensor.pixel_pitch_um`. It needs a spectral EXR (`integrate_qe`); HDR
captures are not supported yet.

## Model

1. **Absorption depth.** α(λ) = 4πk/λ from Green 2008 (`spectra/silicon/si_green2008_nk.csv`). The
   generation depth is exponential, truncated to the epi layer and sampled in 64 equal-probability
   bins. 1/α is about 0.4 µm at 450 nm, 1.5 µm at 550 nm, 3.4 µm at 650 nm and 18 µm at 850 nm.
2. **Drift or diffuse.** Carriers generated in the depletion region drift to their own photodiode.
   Carriers generated in the field-free layer, at height h above the depletion edge, diffuse:
   - FSI (junction on the lit side): h = z − w, so red spreads more.
   - BSI (junction on the far side): h = d − w − z, so blue spreads more.
3. **Lateral spread.** Pure diffusion (no recombination) in a slab: the depletion edge absorbs and the
   far surface reflects (passivated back surface for BSI, p/p+ epi–substrate junction for FSI). The
   collection point follows a method-of-images series of 2-D Poisson kernels
   `h / (2π (r² + h²)^{3/2})`. Each term is integrated in closed form over every neighbour pixel and
   averaged over generation positions inside the source pixel.
4. **Channel kernel.** Each wavelength's kernel is weighted by that channel's share of electrons at
   that wavelength (QE × IRCF × the image-mean spectrum). The resulting channel kernel spreads
   charge by *source* site, and charge is conserved.

Neighbour leak (one edge-adjacent pixel) and total charge leaving the pixel, at 1.4 µm pitch:

| Stack | 450 nm | 550 nm | 650 nm | 850 nm |
|---|---|---|---|---|
| BSI, d = 3 µm, w = 2.5 µm | 5.0 % / 25 % | 2.1 % / 11 % | 1.5 % / 7 % | 1.1 % / 6 % |
| FSI, d = 5 µm, w = 1 µm | 0.5 % / 4 % | 3.0 % / 34 % | 3.8 % / 50 % | 4.2 % / 61 % |

## Assumptions and limits

- The stack geometry defaults are generic, not taken from a specific sensor.
- Deep-trench isolation, microlens and optical crosstalk, chief-ray angle, recombination and the
  electric field in the field-free layer are not modelled. Without DTI the kernels are an upper
  bound for a modern BSI sensor.
- The image-mean spectrum sets each channel's wavelength mix, so local colour does not change the kernel.

## Validation

`tests/test_si_diffusion.py` covers:
- absorption depths against Green 2008;
- the rectangle integral of the Poisson kernel against the Cauchy strip marginal, (2/π) atan(a/h);
- full collection with a reflecting far side;
- the h = 0 delta;
- FSI red > blue and BSI blue > red;
- smaller pixels leak more;
- spectral mixing and charge conservation;
- source-site routing.
