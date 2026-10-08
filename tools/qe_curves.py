"""Shared CSV spectral-curve and QE loaders used by the sensor forward models."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np


def read_csv_curve(path: Path, *, strict_wavelength_axis: bool = False) -> tuple[np.ndarray, np.ndarray]:
    wl: list[float] = []
    val: list[float] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r",\s*", line, maxsplit=1)
        if len(parts) != 2:
            continue
        wl.append(float(parts[0]))
        val.append(float(parts[1]))
    if not wl:
        raise ValueError(f"no data in CSV curve: {path}")
    w = np.asarray(wl, dtype=np.float64)
    v = np.asarray(val, dtype=np.float64)

    ok = np.isfinite(w) & np.isfinite(v)
    w = w[ok]
    v = v[ok]
    if w.size == 0:
        raise ValueError(f"no finite samples in CSV curve: {path}")

    # Some imported camera QE CSVs are normalized-domain traces (0..1-ish) rather than nm.
    # Map them into a visible wavelength domain so interpolation onto spectral buckets is valid.
    if float(np.max(w)) <= 10.0:
        wmin = float(np.min(w))
        wmax = float(np.max(w))
        if wmax - wmin <= 1e-12:
            raise ValueError(f"invalid wavelength axis in CSV curve: {path}")
        if strict_wavelength_axis:
            raise ValueError(
                "strict QE validation: normalized wavelength axis detected "
                f"in {path}; provide explicit wavelength-in-nm CSV"
            )
        w = 380.0 + (w - wmin) * (450.0 / (wmax - wmin))
        print(f"warning: mapped normalized wavelength axis to 380..830 nm for {path}", file=sys.stderr)

    idx = np.argsort(w)
    return w[idx], v[idx]


def load_qe_curves_rgb(
    repo: Path,
    qe_cfg: dict,
    *,
    strict_qe_validation: bool = False,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Load QE curves and auto-correct likely imported red/blue inversion."""
    r = read_csv_curve((repo / qe_cfg["red_csv"]).resolve(), strict_wavelength_axis=strict_qe_validation)
    g = read_csv_curve((repo / qe_cfg["green_csv"]).resolve(), strict_wavelength_axis=strict_qe_validation)
    b = read_csv_curve((repo / qe_cfg["blue_csv"]).resolve(), strict_wavelength_axis=strict_qe_validation)
    r_peak = float(r[0][int(np.argmax(r[1]))])
    g_peak = float(g[0][int(np.argmax(g[1]))])
    b_peak = float(b[0][int(np.argmax(b[1]))])
    if r_peak < g_peak and b_peak > g_peak:
        if strict_qe_validation:
            raise ValueError(
                "strict QE validation: detected likely QE red/blue inversion; "
                "fix channel assignments in QE CSVs"
            )
        print(
            "warning: detected likely QE red/blue inversion; swapping channels "
            f"(red_peak={r_peak:.1f}nm, green_peak={g_peak:.1f}nm, blue_peak={b_peak:.1f}nm)",
            file=sys.stderr,
        )
        r, b = b, r
    # Warn if any QE curve's measured range ends below 780 nm with non-negligible signal.
    # Above the curve's max wavelength, np.interp returns right=0.0, silently dropping
    # any spectral energy in that band.  If the last non-zero sample is before 780 nm and
    # the IRCF is not responsible for cutting it (IRCF check is left to the caller), the
    # channel response is likely truncated.
    _COVERAGE_WARN_NM = 780.0
    for _ch, (_wl, _v) in (("red", r), ("green", g), ("blue", b)):
        _last_nonzero = float(_wl[_v > 1e-4][-1]) if np.any(_v > 1e-4) else 0.0
        if _last_nonzero > 0 and _last_nonzero < _COVERAGE_WARN_NM:
            print(
                f"warning: {_ch} QE curve last non-zero value at {_last_nonzero:.0f} nm "
                f"(< {_COVERAGE_WARN_NM:.0f} nm); spectral energy above this wavelength "
                "is set to zero by extrapolation — verify IRCF covers the gap or extend the QE CSV.",
                file=sys.stderr,
            )
    return r, g, b
