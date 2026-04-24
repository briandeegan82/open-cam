# Illuminant Spectra

This folder contains spectral power distributions (SPDs) used by the sensor-forward calibration path.

## Files

Interpolated CSV files live in `spectra/illuminant/interpolated/` and use:

- Wavelength range: `380-700 nm`
- Step: `1 nm`
- Format: `wavelength_nm,value` (no header)
- Scale: relative SPD (normalized values)

Available presets:

- `D50.csv` - Daylight, ~5000 K (print/viewing workflows)
- `D55.csv` - Daylight, ~5500 K (current default in many models)
- `D65.csv` - Daylight, ~6500 K (common reference daylight)
- `D75.csv` - Daylight, cooler blue daylight
- `A.csv` - Tungsten/incandescent (~2856 K)
- `C.csv` - Legacy daylight illuminant C
- `F2_CWF.csv` - Cool White Fluorescent (CWF / F2)
- `F7.csv` - Broad fluorescent near daylight appearance
- `F11.csv` - Narrow tri-band fluorescent (~4000 K)

CIE typical LED illuminants:

- `LED_B1.csv`
- `LED_B2.csv`
- `LED_B3.csv`
- `LED_B4.csv`
- `LED_B5.csv`
- `LED_BH1.csv`
- `LED_RGB1.csv`
- `LED_V1.csv`
- `LED_V2.csv`

## How to Use

Set the illuminant in your sensor model (or full camera model):

```yaml
sensor_forward:
  model:
    calibration:
      illuminant_override_csv: spectra/illuminant/interpolated/D65.csv
```

You can switch to any other file in the list above, for example:

- `.../A.csv` for tungsten scenes
- `.../F2_CWF.csv` for office-style fluorescent light
- `.../LED_B3.csv` for typical phosphor-converted white LED behavior
- `.../LED_RGB1.csv` for RGB-mixed LED behavior

## Notes

- Existing configs that point to `D55.csv` continue to work unchanged.
- These SPDs are intended for forward-model calibration and illumination targeting behavior.
- Keep all custom illuminants in this same CSV format for compatibility.
