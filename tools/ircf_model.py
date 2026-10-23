#!/usr/bin/env python3
"""Hybrid IR-cut filter model: IR-absorbing blue glass x interference edge coating.

Phone and automotive IR-cut filters are usually a thin IR-absorbing glass carrying an interference
long-wave-cut coating. The glass gives a gradual roll-off from ~600 nm and keeps attenuating to
1100 nm, and its absorption does not shift with incidence angle. The coating gives the steep edge,
blocks the near-IR only to a finite optical density, and its edge moves to shorter wavelengths with
angle (``pixel_angular_response`` ``ircf_angle_shift``). The total transmittance is

    T(lambda) = Ti_glass(lambda)^(d / d_ref) * T_coat(lambda)
    T_coat(lambda) = t_pass * U(lambda) * (floor + (1 - floor) / (1 + exp((lambda - lambda_c) / w)))

``Ti_glass`` is the SCHOTT catalogue internal transmittance (BG60 by default; the faces are
coated, so no bare-glass Fresnel factor). ``U`` is the short-wave (UV) edge of the bundled
``QE_IRCF.csv``. ``lambda_c`` is solved so that T falls to half its 450-600 nm peak at
``t50_nm`` (650 nm, the bundled curve's half-power point). Thickness, edge width, pass level and
the blocking floor are illustrative, not a datasheet: replace them with the filter's data.

Writes ``QE_IRCF_hybrid.csv`` (the sensor's ``ircf_csv``) and ``QE_IRCF_hybrid_coating.csv`` (the
interference part only: ``ircf_coating_csv``, used for the angle shift and as the reflecting face
in the lens-ghost sensor stack).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "spectra/QE/ircf"
GLASS_CSV = OUT / "schott_BG60_taui_1mm.csv"
LEGACY_CSV = REPO / "spectra/QE/interpolated/QE_IRCF.csv"

DEFAULTS = {
    "glass_csv": GLASS_CSV,
    "glass_thickness_mm": 0.21,
    "glass_ref_thickness_mm": 1.0,
    "t50_nm": 650.0,
    "edge_width_nm": 2.5,
    "t_pass": 0.97,
    "blocking_floor": 0.01,
}


def _curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    a = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
    return a[:, 0], a[:, 1]


def glass_transmittance(lam, glass_csv=GLASS_CSV, thickness_mm=0.21, ref_thickness_mm=1.0) -> np.ndarray:
    w, ti = _curve(Path(glass_csv))
    ti = np.clip(np.interp(lam, w, ti), 1e-30, 1.0)
    return ti ** (thickness_mm / ref_thickness_mm)


def uv_edge(lam) -> np.ndarray:
    """Short-wave edge of the bundled IR-cut curve, normalised to 1 above 550 nm."""
    w, t = _curve(LEGACY_CSV)
    peak = t[(w >= 450) & (w <= 600)].max()
    u = np.clip(np.interp(lam, w, t, left=0.0) / peak, 0.0, 1.0)
    return np.where(np.asarray(lam) >= 550.0, 1.0, u)


def coating_transmittance(lam, lambda_c, edge_width_nm=2.5, t_pass=0.97, blocking_floor=0.01) -> np.ndarray:
    lam = np.asarray(lam, dtype=np.float64)
    edge = 1.0 / (1.0 + np.exp(np.clip((lam - lambda_c) / edge_width_nm, -60, 60)))
    return t_pass * uv_edge(lam) * (blocking_floor + (1.0 - blocking_floor) * edge)


def hybrid(lam, **kw) -> tuple[np.ndarray, np.ndarray, float]:
    """(total T, coating T, lambda_c) on ``lam``; lambda_c puts T at half its 450-600 nm peak at t50_nm."""
    p = {**DEFAULTS, **kw}
    g = lambda x: glass_transmittance(x, p["glass_csv"], p["glass_thickness_mm"], p["glass_ref_thickness_mm"])  # noqa: E731
    ref = np.arange(450.0, 601.0)

    def rel50(lc):
        c = lambda x: coating_transmittance(x, lc, p["edge_width_nm"], p["t_pass"], p["blocking_floor"])  # noqa: E731
        return float((g([p["t50_nm"]]) * c([p["t50_nm"]]))[0] / (g(ref) * c(ref)).max()) - 0.5

    lo, hi = p["t50_nm"] - 50.0, p["t50_nm"] + 80.0
    if rel50(hi) < 0:
        raise ValueError("glass alone is below half power at t50_nm: use thinner glass")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if rel50(mid) < 0 else (lo, mid)
    lc = 0.5 * (lo + hi)
    coat = coating_transmittance(lam, lc, p["edge_width_nm"], p["t_pass"], p["blocking_floor"])
    return g(lam) * coat, coat, lc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--glass-thickness-mm", type=float, default=DEFAULTS["glass_thickness_mm"])
    ap.add_argument("--t50-nm", type=float, default=DEFAULTS["t50_nm"])
    ap.add_argument("--edge-width-nm", type=float, default=DEFAULTS["edge_width_nm"])
    ap.add_argument("--blocking-floor", type=float, default=DEFAULTS["blocking_floor"])
    a = ap.parse_args()
    lam = np.arange(300.0, 1101.0, 1.0)
    t, c, lc = hybrid(
        lam,
        glass_thickness_mm=a.glass_thickness_mm,
        t50_nm=a.t50_nm,
        edge_width_nm=a.edge_width_nm,
        blocking_floor=a.blocking_floor,
    )
    a.out_dir.mkdir(parents=True, exist_ok=True)
    common = (
        f"SCHOTT BG60 {a.glass_thickness_mm:g} mm x interference edge (lambda_c {lc:.2f} nm, width {a.edge_width_nm:g} nm, "
        f"floor {a.blocking_floor:g}); T50 {a.t50_nm:g} nm; tools/ircf_model.py; illustrative, not a datasheet"
    )
    np.savetxt(
        a.out_dir / "QE_IRCF_hybrid.csv",
        np.c_[lam, t],
        delimiter=",",
        fmt=["%.1f", "%.6g"],
        header="hybrid IR-cut filter: " + common,
    )
    np.savetxt(
        a.out_dir / "QE_IRCF_hybrid_coating.csv",
        np.c_[lam, c],
        delimiter=",",
        fmt=["%.1f", "%.6g"],
        header="coating of the hybrid IR-cut filter: " + common,
    )
    print(f"lambda_c = {lc:.2f} nm -> {a.out_dir}")


if __name__ == "__main__":
    main()
