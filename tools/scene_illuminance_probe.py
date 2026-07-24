#!/usr/bin/env python3
"""Derive the chart-scene illuminance anchor from a surround-only probe render.

Chart scenes built by build_colorchecker_scene.py / build_munsell_scenes.py place a
large Lambertian "neutral surround" of known flat reflectance behind the chart.
Because the distant light illuminates the chart/surround plane *uniformly* and both
planes face the camera with nothing in front of them (no interreflection can reach
them), the on-axis radiance of that surround measures the scene illuminance directly:

    E_scene_exr = photopic_lux( pi * L_surround(r -> 0) / rho_surround )

in raw-EXR units.  This is the number ``calibration.scene_illuminance_reference_exr``
expects (see tools/pbrt_spectral_exr_to_electrons.py).  Anchoring on it makes the
electron conversion  E_e = L * rad_to_e * target_lux / reference,  which

  - keeps the f-number/aperture dependence: ``rad_to_e`` enters exactly once; the
    camera weight pbrt bakes into the EXR is part of the measured reference and cancels;
  - is illuminant-independent: pbrt normalises light SPDs photometrically
    (``sc /= SpectrumToPhotometric(L)``, src/pbrt/lights.cpp), and so is the reference
    (measured A vs F2_CWF: 7170.7 vs 7164.2 raw-EXR lux, 0.09% apart);
  - is scene-content independent: measured constant to <0.1% across the ColorChecker
    scene and six Munsell hue scenes (2026-07-16, per unit light scale).

The chart itself occludes the surround centre, so the probe renders the chart scene
*minus the chart*: every patch block is stripped from the generated .pbrt, leaving
camera, film, light and surround untouched.  The centre value is extrapolated from a
quadratic fit in r^2 over r < 0.25 (the lens relative-illumination profile is smooth
and near-quadratic there); statistical precision is ~0.1% at 256 spp.

``radial_profile()`` additionally supports verifying a *chart* EXR against the probe:
outside the chart, surround pixels of the chart render must reproduce the probe's
profile bin-for-bin (measured agreement ~0.05%); a mismatch means the chart render's
rig (light scale, lens, illuminant handling...) differs from the probe's.

CLI (reproduction):
    python tools/scene_illuminance_probe.py --scene scenes/generated/colorchecker.pbrt
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from exr_multispectral import spectral_buckets_from_exr  # noqa: E402
from pbrt_spectral_exr_to_electrons import illuminance_lux_from_irradiance  # noqa: E402

DEFAULT_PBRT = TOOLS_DIR.parent / "third_party" / "pbrt-v4" / "build" / "pbrt"

# Central rectangle (fraction of W, H) that the chart may occupy in chart EXRs.
# Generous for both chart types at the standard cam_dist=6 framing; verified against
# the rendered board extents (CC: 0.19/0.19, Munsell: 0.10/0.23 half-fractions).
CHART_EXCLUDE_FRAC = (0.25, 0.30)


@dataclass
class ProbeResult:
    e0_lux_exr: float          # on-axis scene illuminance, raw-EXR photopic lux
    rho_surround: float        # surround reflectance parsed from the scene
    profile: list              # [(r_mid, E_lux_exr, npix), ...] radial surround profile
    probe_exr: Path


def make_probe_scene(scene_pbrt: Path, probe_pbrt: Path, probe_exr: Path,
                     spp: int = 256) -> float:
    """Strip the chart from a generated scene, keeping the neutral surround.

    Returns the surround reflectance parsed from the kept block.  Relative .spd/.dat
    resource paths are absolutised against the source scene's directory so the probe
    can live anywhere.
    """
    src = scene_pbrt.read_text()
    blocks = re.split(r"(AttributeBegin.*?AttributeEnd\n)", src, flags=re.S)
    rho = None
    kept = []
    for b in blocks:
        if b.startswith("AttributeBegin"):
            m = re.search(
                r'"rgb reflectance"\s*\[\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*\]', b
            )
            if m and len({m.group(1), m.group(2), m.group(3)}) == 1:
                if rho is not None:
                    raise ValueError(f"{scene_pbrt}: more than one neutral-surround block")
                rho = float(m.group(1))
            else:
                continue  # a chart patch (or other geometry): drop it
        kept.append(b)
    if rho is None:
        raise ValueError(
            f"{scene_pbrt}: no neutral-surround block (flat rgb reflectance) found; "
            "cannot build an illuminance probe"
        )
    txt = "".join(kept)

    scene_dir = scene_pbrt.resolve().parent

    def absolutise(m: re.Match) -> str:
        p = m.group(1)
        if Path(p).is_absolute():
            return m.group(0)
        return f'"{(scene_dir / p).resolve()}"'

    txt = re.sub(r'"([^"]+\.(?:spd|dat|csv))"', absolutise, txt)
    txt = re.sub(r'"integer pixelsamples"\s*\[\s*\d+\s*\]',
                 f'"integer pixelsamples" [{int(spp)}]', txt)
    txt, nsub = re.subn(r'"string filename"\s*\[\s*"[^"]*"\s*\]',
                        f'"string filename" ["{probe_exr.resolve()}"]', txt)
    if nsub != 1:
        raise ValueError(f"{scene_pbrt}: expected exactly one Film filename, found {nsub}")
    probe_pbrt.parent.mkdir(parents=True, exist_ok=True)
    probe_pbrt.write_text(txt)
    return rho


def render(pbrt_scene: Path, pbrt_bin: Path = DEFAULT_PBRT, gpu: bool = True) -> None:
    cmd = [str(pbrt_bin)] + (["--gpu"] if gpu else []) + [str(pbrt_scene)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"pbrt probe render failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")


def radial_profile(exr_path: Path, rho: float, *,
                   exclude_chart: bool = False,
                   r_lo: float = 0.0, r_hi: float = 1.3, r_step: float = 0.05,
                   min_pix: int = 200) -> list:
    """Radius-binned E(r) = lux(pi * L / rho) over surround pixels.

    Radius unit: half the image height (r=1.0 at the top/bottom edge centre).
    With ``exclude_chart`` a central rectangle (CHART_EXCLUDE_FRAC) is masked so the
    same function works on chart EXRs for verification.
    """
    cube, lam = spectral_buckets_from_exr(exr_path)
    cube = cube.astype(np.float64)
    H, W, _ = cube.shape
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(xx - W / 2, yy - H / 2) / (H / 2)
    mask = np.ones((H, W), dtype=bool)
    if exclude_chart:
        fx, fy = CHART_EXCLUDE_FRAC
        mask &= ~((np.abs(xx - W / 2) < fx * W) & (np.abs(yy - H / 2) < fy * H))
    out = []
    for b0 in np.arange(r_lo, r_hi, r_step):
        sel = mask & (r >= b0) & (r < b0 + r_step)
        n = int(sel.sum())
        if n < min_pix:
            continue
        E = np.pi * cube[sel].mean(axis=0) / rho
        out.append((b0 + r_step / 2, float(illuminance_lux_from_irradiance(lam, E)), n))
    return out


def measure_e0(probe_exr: Path, rho: float, r_max: float = 0.25) -> float:
    """On-axis illuminance: quadratic-in-r^2 fit of fine radial bins, extrapolated to r=0."""
    prof = radial_profile(probe_exr, rho, r_hi=r_max, r_step=0.025, min_pix=50)
    if len(prof) < 4:
        raise RuntimeError(f"{probe_exr}: too few central bins for the r->0 fit")
    rm = np.array([p[0] for p in prof])
    Em = np.array([p[1] for p in prof])
    n = np.array([p[2] for p in prof], dtype=np.float64)
    A = np.vstack([np.ones_like(rm), rm ** 2, rm ** 4]).T * np.sqrt(n)[:, None]
    coef, *_ = np.linalg.lstsq(A, Em * np.sqrt(n), rcond=None)
    return float(coef[0])


def derive_reference(scene_pbrt: Path, workdir: Path, *,
                     pbrt_bin: Path = DEFAULT_PBRT, spp: int = 256,
                     gpu: bool = True) -> ProbeResult:
    """Build + render + measure the probe for a generated chart scene."""
    scene_pbrt = Path(scene_pbrt).resolve()
    workdir = Path(workdir).resolve()
    probe_pbrt = workdir / f"probe_{scene_pbrt.stem}.pbrt"
    probe_exr = workdir / f"probe_{scene_pbrt.stem}.exr"
    rho = make_probe_scene(scene_pbrt, probe_pbrt, probe_exr, spp=spp)
    render(probe_pbrt, pbrt_bin=pbrt_bin, gpu=gpu)
    e0 = measure_e0(probe_exr, rho)
    prof = radial_profile(probe_exr, rho)
    return ProbeResult(e0_lux_exr=e0, rho_surround=rho, profile=prof, probe_exr=probe_exr)


def verify_chart_exr(chart_exr: Path, probe: ProbeResult, *,
                     r_lo: float = 0.65, r_hi: float = 1.25,
                     tol: float = 0.01) -> float:
    """Check a chart render's surround against the probe profile.

    Returns the worst |ratio - 1| over the compared bins; raises if it exceeds ``tol``.
    """
    chart = radial_profile(chart_exr, probe.rho_surround, exclude_chart=True,
                           r_lo=r_lo, r_hi=r_hi)
    ref = {round(rm, 4): E for rm, E, _ in probe.profile}
    worst = 0.0
    for rm, E, _ in chart:
        Eref = ref.get(round(rm, 4))
        if Eref is None:
            continue
        worst = max(worst, abs(E / Eref - 1.0))
    if worst > tol:
        raise RuntimeError(
            f"{chart_exr}: surround illuminance deviates {100*worst:.2f}% from the probe "
            f"(tol {100*tol:.1f}%); the chart render's rig does not match the probe "
            "(light scale / lens / scene changed?)"
        )
    return worst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path, required=True,
                    help="Generated chart scene .pbrt (colorchecker or munsell hue)")
    ap.add_argument("--workdir", type=Path, default=Path("out/illuminance_probe"))
    ap.add_argument("--pbrt-bin", type=Path, default=DEFAULT_PBRT)
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--cpu", action="store_true", help="Render the probe on CPU")
    args = ap.parse_args()

    res = derive_reference(args.scene, args.workdir, pbrt_bin=args.pbrt_bin,
                           spp=args.spp, gpu=not args.cpu)
    print(f"scene:                {args.scene}")
    print(f"surround reflectance: {res.rho_surround}")
    print(f"E_scene_exr (r->0):   {res.e0_lux_exr:.1f}  [raw-EXR photopic lux]")
    print("radial profile (r_mid, E, npix):")
    for rm, E, n in res.profile:
        print(f"  {rm:5.3f}  {E:9.1f}  {n:7d}")


if __name__ == "__main__":
    main()
