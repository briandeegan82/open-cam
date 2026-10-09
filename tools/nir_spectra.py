"""NIR / no-IRCF spectral data: silicon photodiode QE, IR-transparent CFA dyes, NIR LED SPDs.

The repo's measured QE curves (``spectra/QE/interpolated``) stop near 700 nm, so a sensor
without an IR-cut filter cannot be modelled from them alone.  This module builds extended
curves to 1100 nm from first principles plus clearly stated assumptions:

* **Silicon absorption** -- intrinsic c-Si at 300 K, n,k from M. A. Green, "Self-consistent
  optical parameters of intrinsic silicon at 300 K including temperature coefficients",
  Sol. Energ. Mat. Sol. Cells 92, 1305-1310 (2008) (``spectra/silicon/si_green2008_nk.csv``);
  absorption coefficient ``alpha = 4 pi k / lambda``.
* **Photodiode QE** -- ``QE(lambda) = (1 - R) (1 - exp(-alpha d)) * eta_c``: a 1-D absorber of
  collection depth ``d`` (Beer-Lambert), front reflectance ``R`` (0 = ideal AR coating, or the
  bare-Si Fresnel value ``((n-1)^2 + k^2) / ((n+1)^2 + k^2)``) and a wavelength-independent
  collection efficiency ``eta_c``.  No diffusion-length, light-trapping or DTI effects.
* **Colour filters in the NIR** -- *assumption*: organic pigment filters become transparent in
  the NIR, so filter transmission ``T_c = QE_c / QE_si`` at the join wavelength rises linearly
  to 1 at ``transparent_nm`` (default 800 nm).  Replace with measured filter curves when known.
* **NIR LEDs** -- Gaussian SPDs, parameters (peak, FWHM) are user inputs; take them from the
  emitter datasheet.  The defaults written by :func:`main` (850 nm / 30 nm FWHM and
  940 nm / 40 nm FWHM) are illustrative.

These curves are an engineering model, not measurements; treat NIR results accordingly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SI_NK_CSV = "spectra/silicon/si_green2008_nk.csv"
GRID_NM = np.arange(360.0, 1101.0, 1.0)


def silicon_nk(lambdas_nm: np.ndarray, repo: Path = REPO) -> tuple[np.ndarray, np.ndarray]:
    with open(Path(repo) / SI_NK_CSV, encoding="utf-8") as fh:  # skips "#" comments and header row
        d = np.loadtxt([ln for ln in fh if ln[:1].isdigit()], delimiter=",")
    lam = np.asarray(lambdas_nm, dtype=np.float64)
    n = np.interp(lam, d[:, 0], d[:, 1])
    # k spans 13 decades: interpolate log k.
    k = np.exp(np.interp(lam, d[:, 0], np.log(np.maximum(d[:, 2], 1e-30))))
    return n, k


def silicon_alpha_per_um(lambdas_nm: np.ndarray, repo: Path = REPO) -> np.ndarray:
    """Absorption coefficient [1/um] = 4 pi k / lambda."""
    _, k = silicon_nk(lambdas_nm, repo)
    return 4.0 * np.pi * k / (np.asarray(lambdas_nm, dtype=np.float64) * 1e-3)


def silicon_qe(
    lambdas_nm: np.ndarray,
    *,
    thickness_um: float = 3.0,
    reflectance: float | str = 0.0,
    collection_efficiency: float = 1.0,
    repo: Path = REPO,
) -> np.ndarray:
    """1-D photodiode internal QE ``(1-R)(1-exp(-alpha d)) eta_c``; ``reflectance='fresnel'`` for bare Si."""
    lam = np.asarray(lambdas_nm, dtype=np.float64)
    if reflectance == "fresnel":
        n, k = silicon_nk(lam, repo)
        r = ((n - 1.0) ** 2 + k**2) / ((n + 1.0) ** 2 + k**2)
    else:
        r = float(reflectance)
    absorbed = 1.0 - np.exp(-silicon_alpha_per_um(lam, repo) * float(thickness_um))
    return np.clip((1.0 - r) * absorbed * float(collection_efficiency), 0.0, 1.0)


def extend_qe_to_nir(
    vis_wl: np.ndarray,
    vis_qe: np.ndarray,
    si_qe: np.ndarray,
    grid_nm: np.ndarray = GRID_NM,
    *,
    join_nm: float = 650.0,
    transparent_nm: float = 800.0,
) -> np.ndarray:
    """Visible measured QE up to ``join_nm``, then ``QE_si * T`` with T ramping to 1 at ``transparent_nm``."""
    g = np.asarray(grid_nm, dtype=np.float64)
    vis = np.interp(g, vis_wl, vis_qe, left=0.0, right=0.0)
    q_join = float(np.interp(join_nm, vis_wl, vis_qe))
    si_join = float(np.interp(join_nm, g, si_qe))
    t_join = np.clip(q_join / max(si_join, 1e-9), 0.0, 1.0)
    ramp = np.clip((g - join_nm) / max(transparent_nm - join_nm, 1e-9), 0.0, 1.0)
    t = t_join + (1.0 - t_join) * ramp
    return np.where(g <= join_nm, vis, si_qe * t)


def gaussian_led_spd(grid_nm: np.ndarray, peak_nm: float, fwhm_nm: float) -> np.ndarray:
    """Unit-peak Gaussian emitter SPD."""
    sigma = float(fwhm_nm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    spd = np.exp(-0.5 * ((np.asarray(grid_nm, dtype=np.float64) - float(peak_nm)) / sigma) ** 2)
    return np.where(spd < 1e-9, 0.0, spd)


def _write(path: Path, wl: np.ndarray, v: np.ndarray, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {header}\n" + "".join(f"{w:.1f},{x:.6g}\n" for w, x in zip(wl, v, strict=True)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", type=Path, default=REPO)
    ap.add_argument("--thickness-um", type=float, default=3.0, help="Si collection depth (BSI epi ~3 um)")
    ap.add_argument("--join-nm", type=float, default=650.0)
    ap.add_argument("--transparent-nm", type=float, default=800.0)
    args = ap.parse_args()
    repo = args.repo_root
    from qe_curves import read_csv_curve  # noqa: PLC0415

    si = silicon_qe(GRID_NM, thickness_um=args.thickness_um, repo=repo)
    note = f"d={args.thickness_um} um, ideal AR, Green 2008 Si; tools/nir_spectra.py"
    _write(repo / "spectra/QE/nir/QE_si_bsi.csv", GRID_NM, si, f"Si photodiode QE model ({note})")
    for name in ("red", "green", "blue", "mono", "cyan", "yellow", "magenta"):
        wl, q = read_csv_curve(repo / f"spectra/QE/interpolated/QE_{name}.csv")
        ext = extend_qe_to_nir(wl, q, si, join_nm=args.join_nm, transparent_nm=args.transparent_nm)
        _write(
            repo / f"spectra/QE/nir/QE_{name}_noircf.csv",
            GRID_NM,
            ext,
            f"QE_{name} <= {args.join_nm:g} nm, Si QE x filter ramp to T=1 at {args.transparent_nm:g} nm ({note})",
        )
    for peak, fwhm in ((850.0, 30.0), (940.0, 40.0)):
        _write(
            repo / f"spectra/illuminant/nir/LED_{peak:.0f}nm.csv",
            GRID_NM,
            gaussian_led_spd(GRID_NM, peak, fwhm),
            f"Gaussian NIR LED, peak {peak:g} nm, FWHM {fwhm:g} nm (illustrative; use datasheet values)",
        )


if __name__ == "__main__":
    main()
