"""Validate glass dispersion (tools/lens_glass.py + third_party/patches/0003) against a Python ray trace.

A small flat-spectrum emitter is imaged at film (FX, 0) through a sidecar lens (2 um pixels on a
12000 x 12000 film, crop window around the image). pbrt renders it with and without
``"float dispersion"``. Per spectral bucket the PSF centroid gives the lateral colour and the PSF
RMS radius the axial colour (defocus). The prediction is a real-ray trace of the same lens
(``lens_diffraction_psf.trace``) at each bucket's centre wavelength with the sidecar's n(lambda),
focused at the n_d film distance as pbrt does. Cases: D-Gauss f/2 on axis (axial colour),
D-Gauss f/5.6 at 9 mm (a symmetric design: lateral colour ~0) and the retrofocus 22 mm at 9 mm.

Run from the repository root (needs third_party/pbrt-v4/build/pbrt with the patches; ~1 min):
    venv/bin/python scripts/validate_lens_dispersion.py
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import lens_diffraction_psf as ldp  # noqa: E402
from exr_multispectral import parse_s0_wavelength_nm, read_separate_exr_channels  # noqa: E402
from lens_glass import LAMBDA_D_NM, pbrt_camera_lines  # noqa: E402

LENSES = REPO / "config" / "lenses"
CASES = {  # name: (lens file, aperture diameter mm, film x of the image mm)
    "dgauss_f2_axis": ("dgauss.50mm.dat", 25.0, 0.0),
    "dgauss_f5.6_9mm": ("dgauss.50mm.dat", 6.194, 9.0),
    "wide22_9mm": ("wide_22mm.dat", 5.0, 9.0),
}
PBRT = REPO / "third_party" / "pbrt-v4" / "build" / "pbrt"
FOCUS_M = 3.2
PITCH_UM = 2.0
RES = 12000
CROP = 160
NBUCKETS, LMIN, LMAX = 12, 400.0, 700.0  # 25 nm buckets; one is centred on 587.5 nm
DIAG_MM = PITCH_UM * 1e-3 * RES * math.sqrt(2)
SOURCE_HALF_M = 1e-4


def scene_text(lens, ap_mm, X, Y, bounds, spp, out_exr, dispersion: bool) -> str:
    disp = "\n".join(pbrt_camera_lines(lens, dispersion))
    s = SOURCE_HALF_M
    pts = " ".join(f"{X + dx:.8f} {Y + dy:.8f} {FOCUS_M}" for dx, dy in ((-s, -s), (s, -s), (-s, s), (s, s)))
    return f"""Option "seed" 1
ColorSpace "srgb"
Sampler "zsobol" "integer pixelsamples" [{spp}]
Integrator "path" "integer maxdepth" [1]
PixelFilter "box"
Film "spectral" "string filename" ["{out_exr}"] "integer xresolution" [{RES}] "integer yresolution" [{RES}]
    "float diagonal" [{DIAG_MM:.6f}] "bool savefp16" false
    "integer nbuckets" [{NBUCKETS}] "float lambdamin" [{LMIN}] "float lambdamax" [{LMAX}]
    "integer pixelbounds" [{bounds[0]} {bounds[1]} {bounds[2]} {bounds[3]}]
LookAt 0 0 0  0 0 1  0 1 0
Camera "realistic" "string lensfile" ["{lens}"] "float aperturediameter" [{ap_mm}]
    "float focusdistance" [{FOCUS_M}]
{disp}
WorldBegin
AttributeBegin
  AreaLightSource "diffuse" "spectrum L" [300 1 900 1] "bool twosided" true
  Shape "bilinearmesh" "point3 P" [{pts}]
AttributeEnd
"""


def pbrt_psf_stats(exr: Path, bounds, src_side_mm: float) -> dict[float, tuple[float, float, float]]:
    """Bucket centre (nm) -> (centroid x, y in film mm, RMS radius in um without pixel/source size)."""
    out = {}
    for name, img in read_separate_exr_channels(exr).items():
        lam = parse_s0_wavelength_nm(name)
        if lam is None:
            continue
        img = np.clip(np.asarray(img, dtype=np.float64), 0.0, None)
        rows, cols = np.indices(img.shape)
        w = img.sum()
        col = bounds[0] + (cols * img).sum() / w + 0.5
        row = bounds[2] + (rows * img).sum() / w + 0.5
        var = ((cols + bounds[0] + 0.5 - col) ** 2 + (rows + bounds[2] + 0.5 - row) ** 2) * img
        var_um2 = var.sum() / w * PITCH_UM**2 - 2 * (PITCH_UM**2 + (1e3 * src_side_mm) ** 2) / 12
        out[float(lam)] = (
            -(col - RES / 2) * PITCH_UM * 1e-3,
            (row - RES / 2) * PITCH_UM * 1e-3,
            math.sqrt(max(var_um2, 0.0)),
        )
    return out


def traced_stats(model: ldp.LensPsfModel, obj_mm, lam_nm: float, n: int = 241) -> tuple[float, float, float]:
    """Centroid (mm) and RMS radius (um) on the film of a uniform ray grid over the front element."""
    lens = model.lens_at(lam_nm)
    g = np.linspace(-1.0, 1.0, n) * lens.semi_aperture[0]
    u, v = np.meshgrid(g, g)
    keep = u**2 + v**2 <= lens.semi_aperture[0] ** 2
    tgt = np.stack([u[keep], v[keep], np.zeros(keep.sum())], axis=1)
    o = np.broadcast_to(np.asarray(obj_mm, dtype=np.float64), tgt.shape)
    r = ldp.trace(lens, o, tgt - o)
    ok = r["ok"]
    p = r["o"][ok] + ((model.z_film - r["o"][ok, 2]) / r["d"][ok, 2])[:, None] * r["d"][ok]
    cx, cy = p[:, 0].mean(), p[:, 1].mean()
    return float(cx), float(cy), 1e3 * float(np.sqrt(np.mean((p[:, 0] - cx) ** 2 + (p[:, 1] - cy) ** 2)))


def run_case(name: str, lens_name: str, ap_mm: float, fx: float, spp: int, od: Path) -> dict:
    lens = LENSES / lens_name
    cfg = {
        "lensfile": str(lens),
        "aperture_diameter_mm": ap_mm,
        "focus_distance_m": FOCUS_M,
        "pixel_pitch_um": PITCH_UM,
    }
    model = ldp.model_from_config(cfg, {}, {}, repo=REPO, film_res=(RES, RES))
    assert model.dispersive_lens is not None, f"{lens_name} has no glass sidecar"
    y0 = ldp.object_height_for_image_height(model.lens, model.z_obj, model.z_film, fx) if fx else 0.0
    X, Y = -y0 * 1e-3, 0.0  # camera-space metres (inverted image)
    h_ref = fx if fx else 0.1
    mag = h_ref / ldp.object_height_for_image_height(model.lens, model.z_obj, model.z_film, h_ref)
    src_side_mm = 2e3 * SOURCE_HALF_M * abs(mag)
    col = RES / 2 - fx / (PITCH_UM * 1e-3)
    x0, y0r = int(round(col)) - CROP // 2, RES // 2 - CROP // 2
    bounds = (x0, x0 + CROP, y0r, y0r + CROP)
    pb = {}
    for disp in (True, False):
        tag = f"{name}_{'dispersion' if disp else 'nd_only'}"
        sc = od / f"{tag}.pbrt"
        sc.write_text(scene_text(lens, ap_mm, X, Y, bounds, spp, od / f"{tag}.exr", disp))
        subprocess.run([str(PBRT), "--quiet", str(sc)], check=True, cwd=od)
        pb[disp] = pbrt_psf_stats(od / f"{tag}.exr", bounds, src_side_mm)
    lams = sorted(pb[True])
    ref = min(lams, key=lambda lam: abs(lam - LAMBDA_D_NM))
    py = {lam: traced_stats(model, (X * 1e3, Y * 1e3, model.z_obj), lam) for lam in lams}
    rows = [
        {
            "lambda_nm": lam,
            "lateral_pbrt_um": 1e3 * (pb[True][lam][0] - pb[True][ref][0]),
            "lateral_trace_um": 1e3 * (py[lam][0] - py[ref][0]),
            "lateral_pbrt_nd_only_um": 1e3 * (pb[False][lam][0] - pb[False][ref][0]),
            "rms_pbrt_um": pb[True][lam][2],
            "rms_trace_um": py[lam][2],
            "rms_pbrt_nd_only_um": pb[False][lam][2],
        }
        for lam in lams
    ]
    print(f"\n{name}: {lens_name} aperture {ap_mm} mm, field {fx} mm (um; lateral = x shift vs {ref} nm)")
    print(" lambda | lateral pbrt  trace  n_d-only | RMS radius pbrt  trace  n_d-only")
    for r in rows:
        print(
            f"{r['lambda_nm']:7.1f} | {r['lateral_pbrt_um']:8.2f} {r['lateral_trace_um']:7.2f} "
            f"{r['lateral_pbrt_nd_only_um']:7.2f} | {r['rms_pbrt_um']:10.2f} {r['rms_trace_um']:7.2f} "
            f"{r['rms_pbrt_nd_only_um']:7.2f}"
        )
    return {"lens": lens_name, "aperture_mm": ap_mm, "field_mm": fx, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spp", type=int, default=1024)
    ap.add_argument("--cases", nargs="*", default=list(CASES))
    ap.add_argument("--out-dir", type=Path, default=REPO / "out" / "lens_dispersion")
    args = ap.parse_args()
    od = args.out_dir.resolve()
    od.mkdir(parents=True, exist_ok=True)
    res = {c: run_case(c, *CASES[c], args.spp, od) for c in args.cases}
    (od / "lens_dispersion_validation.json").write_text(json.dumps({"spp": args.spp, "cases": res}, indent=2))


if __name__ == "__main__":
    main()
