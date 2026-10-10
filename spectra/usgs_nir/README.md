# USGS NIR material chart

Measured 350-1100 nm reflectances (1 nm) of 24 materials from the USGS Spectral Library v7
(splib07a, ASD FieldSpec, absolute reflectance): Kokaly, R.F. et al., 2017, USGS Spectral Library
Version 7: U.S. Geological Survey Data Series 1035, doi:10.5066/F7RR1WDJ. Public domain.

`NN_<name>.csv` (wavelength_nm, reflectance), row-major in the ColorChecker 6 x 4 layout. The splib07a
record for each patch is listed in `tools/nir_materials.py` (`NIR_CHART`); rerun
`python tools/nir_materials.py` to regenerate. See `docs/CFA_NIR.md`.
