"""Validate tools/lens_diffraction_psf.py with pbrt renders through config/lenses/dgauss.50mm.dat.

For f/2, f/5.6 and f/11 (stop diameters solved with lens_prescription.traced_f_number) and five
field points (on axis; 6.5 and 10.9 mm image height along the film x axis -> radial MTF, and
along the film y axis -> azimuthal MTF), a near-vertical (5 deg) black/white edge is rendered
with pbrt's realistic camera (box pixel filter, 2 um pixels, crop windows of a 12000 x 12000 film,
20 nm spectral buckets). The edge MTF of the 550 nm bucket is measured with the noise-robust
slanted-edge method of paper/ei2027/scripts/fig_mtf.py (edge_sfr) before and after
``combine: diffraction_only`` and compared with
  * the lens-design wave-optics MTF: |OTF| of the full traced pupil P exp(ikW) (Goodman 2005,
    ch. 6), and
  * the geometric (ray) OTF of the traced spot, x the diffraction-only MTF,
both multiplied by the pixel-aperture MTF |sinc(f p)| that pbrt's box filter imposes.

Run from the repository root (needs third_party/pbrt-v4/build/pbrt; ~10-20 min on 8 cores):
    venv/bin/python scripts/validate_lens_diffraction.py [--spp 2048] [--out-dir out/lens_diffraction]
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
sys.path.insert(0, str(REPO / "paper" / "ei2027" / "scripts"))

import lens_diffraction_psf as ldp  # noqa: E402
from exr_multispectral import parse_s0_wavelength_nm, read_separate_exr_channels  # noqa: E402
from fig_mtf import edge_sfr  # noqa: E402

LENS = REPO / "config" / "lenses" / "dgauss.50mm.dat"
PBRT = REPO / "third_party" / "pbrt-v4" / "build" / "pbrt"
FOCUS_M = 3.2
PITCH_UM = 2.0
RES = 12000
TILES = 11
CROP = 200
LAM = 550.0
APERTURES = {"f/2": 25.0, "f/5.6": 6.194, "f/11": 3.155}
TILE_STEP_MM = RES / TILES * PITCH_UM * 1e-3
FIELDS = {  # name: (film x mm, film y mm, MTF direction label)
    "axis": (0.0, 0.0, "on-axis"),
    "x3": (3 * TILE_STEP_MM, 0.0, "radial"),
    "x5": (5 * TILE_STEP_MM, 0.0, "radial"),
    "y3": (0.0, 3 * TILE_STEP_MM, "azimuthal"),
    "y5": (0.0, 5 * TILE_STEP_MM, "azimuthal"),
}
DIAG_MM = PITCH_UM * 1e-3 * RES * math.sqrt(2)


def raster_of_film(fx: float, fy: float) -> tuple[float, float]:
    """Film mm -> raster (col, row); the convention used by LensPsfModel.tile_kernel."""
    return RES / 2 - fx / (PITCH_UM * 1e-3), RES / 2 + fy / (PITCH_UM * 1e-3)


def object_point(lens, z_film, z_obj, fx, fy) -> tuple[float, float]:
    """Camera-space object (X, Y) in metres imaged at film (fx, fy) (inverted image)."""
    h = math.hypot(fx, fy)
    if h == 0:
        return 0.0, 0.0
    y0 = ldp.object_height_for_image_height(lens, z_obj, z_film, h)
    return -fx / h * y0 * 1e-3, -fy / h * y0 * 1e-3


def scene_text(ap_mm, xres, yres, bounds, spp, rects, out_exr, nbuckets=15) -> str:
    pb = "" if bounds is None else f'    "integer pixelbounds" [{bounds[0]} {bounds[1]} {bounds[2]} {bounds[3]}]\n'
    s = f"""Option "seed" 1
ColorSpace "srgb"
Sampler "zsobol" "integer pixelsamples" [{spp}]
Integrator "path" "integer maxdepth" [1]
PixelFilter "box"
Film "spectral" "string filename" ["{out_exr}"] "integer xresolution" [{xres}] "integer yresolution" [{yres}]
    "float diagonal" [{DIAG_MM:.6f}] "bool savefp16" false
    "integer nbuckets" [{nbuckets}] "float lambdamin" [400] "float lambdamax" [700]
{pb}LookAt 0 0 0  0 0 1  0 1 0
Camera "realistic" "string lensfile" ["{LENS}"] "float aperturediameter" [{ap_mm}]
    "float focusdistance" [{FOCUS_M}]
WorldBegin
AttributeBegin
  AreaLightSource "diffuse" "spectrum L" [300 1 900 1] "bool twosided" true
  Material "diffuse" "rgb reflectance" [0 0 0]
  Shape "bilinearmesh" "point3 P" [-2 -2 {FOCUS_M + 0.001} 2 -2 {FOCUS_M + 0.001} -2 2 {FOCUS_M + 0.001} 2 2 {FOCUS_M + 0.001}]
AttributeEnd
Material "diffuse" "rgb reflectance" [0 0 0]
"""
    for quad in rects:
        pts = " ".join(f"{x:.7f} {y:.7f} {FOCUS_M:.6f}" for x, y in quad)
        s += f'Shape "bilinearmesh" "point3 P" [{pts}]\n'
    return s


def edge_rect(x, y, black_side: float, size=0.08, angle_deg=5.0):
    """Rectangle whose near-vertical edge (tilted angle_deg) passes through (x, y)."""
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    pts = []
    for u, v in ((0, -1), (black_side, -1), (0, 1), (black_side, 1)):  # bilinear patch order
        du, dv = u * size, v * size / 2
        pts.append((x + c * du - s * dv, y + s * du + c * dv))
    return pts


def bucket(chans: dict, lam_nm: float) -> tuple[str, np.ndarray]:
    for k, v in chans.items():
        if parse_s0_wavelength_nm(k) == lam_nm:
            return k, v
    raise KeyError(f"no S0 bucket at {lam_nm} nm in {list(chans)}")


def render(scene: Path, cwd: Path) -> None:
    subprocess.run([str(PBRT), "--quiet", str(scene)], check=True, cwd=cwd)


def measure(ch: np.ndarray) -> dict:
    band = ch[20:-20]
    prof = np.abs(np.diff(band.mean(axis=0)))
    c = int(np.argmax(prof[45:-45])) + 45
    return edge_sfr(band[:, c - 40 : c + 40])


def predictions(model: ldp.LensPsfModel, fx, fy, freqs_cy_mm):
    """Wave-optics and geometric x diffraction-only MTF along film x at the field point."""
    h = math.hypot(fx, fy)
    fp = model.pupil(h)
    # trace_field_pupil images at (0, -h): radial = film y (axis 0), azimuthal = film x (axis 1).
    radial = fx != 0.0
    os_ = ldp.oversampling_for(fp, LAM, 0.25)
    dx = 0.25 / os_
    n = 2 ** int(math.ceil(math.log2(max(1024, 160.0 / (dx * fp.s_extent / (LAM * 1e-3))))))
    n = max(n, int(round(4 * 1e3 * float(np.max(np.hypot(*fp.spot_mm.T), initial=0.0)) / dx)) + 1024)
    out = {}
    for key, ab in (("wave", True), ("diff", False)):
        psf = ldp.fine_psf(fp, LAM, dx, n, aberrations=ab)
        f, mx, my = ldp.mtf_from_psf(psf, dx)
        out[key] = np.interp(freqs_cy_mm, f, my if radial else mx)
    geo_angle = 0.5 * math.pi if radial else 0.0
    out["geom"] = ldp.geometric_otf(fp, freqs_cy_mm, geo_angle)
    pix = np.abs(np.sinc(freqs_cy_mm * PITCH_UM * 1e-3))
    return {
        "wave": out["wave"] * pix,
        "geom_x_diff": out["geom"] * out["diff"] * pix,
        "geom": out["geom"] * pix,
        "diff_only": out["diff"],
    }


def synthetic_edge_mtf(lsf: np.ndarray, x_um: np.ndarray, angle_deg: float = 5.0) -> dict:
    """Noise-free slanted edge blurred by a 1-D LSF (along the edge normal), measured like the renders.

    Pixel value = ESF(d) = sum_j lsf_j R(d - x_j / p), R = box-pixel-integrated step, d = signed
    distance to the edge in pixels. Passing predictions through the same edge_sfr ROI and
    flat-fielding removes window/normalisation bias from the render-vs-prediction comparison.
    """
    rows, cols = np.mgrid[0:CROP, 0:CROP].astype(np.float64)
    t = math.tan(math.radians(angle_deg))
    d = (cols - (CROP / 2 + t * (rows - CROP / 2))) * math.cos(math.radians(angle_deg))
    w = lsf / lsf.sum()
    u = x_um / PITCH_UM
    keep = w > 1e-7 * w.max()
    w, u = w[keep], u[keep]
    img = np.zeros_like(d)
    for wj, uj in zip(w, u, strict=True):
        img += wj * np.clip(0.5 - (d - uj), 0.0, 1.0)
    return measure(0.02 + img)


def lsfs(model: ldp.LensPsfModel, fx, fy) -> dict:
    """1-D LSFs (along the MTF direction) of the wave, diffraction-only, geometric, geom x diff PSFs."""
    h = math.hypot(fx, fy)
    fp = model.pupil(h)
    radial = fx != 0.0
    os_ = ldp.oversampling_for(fp, LAM, 0.25)
    dx = 0.25 / os_
    spot = 1e3 * float(np.max(np.hypot(*fp.spot_mm.T), initial=0.0))
    n = max(1024, int(round(4 * spot / dx)) + 1024)
    n += n % 2
    x = (np.arange(n) - n // 2) * dx
    out = {}
    for key, ab in (("wave", True), ("diff_only", False)):
        psf = ldp.fine_psf(fp, LAM, dx, n, aberrations=ab)
        out[key] = psf.sum(axis=1) if radial else psf.sum(axis=0)
    proj = 1e3 * (fp.spot_mm[:, 1] if radial else fp.spot_mm[:, 0])
    geom = np.bincount(np.clip(np.round(proj / dx).astype(int) + n // 2, 0, n - 1), fp.spot_weight, minlength=n)
    out["geom"] = geom
    out["geom_x_diff"] = np.convolve(geom, out["diff_only"], mode="same")
    return {k: (v, x) for k, v in out.items()}


def mtf50(f, m):
    i = np.flatnonzero(m < 0.5)
    if i.size == 0 or i[0] == 0:
        return float("nan")
    j = i[0]
    return float(f[j - 1] + (m[j - 1] - 0.5) * (f[j] - f[j - 1]) / (m[j - 1] - m[j]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spp", type=int, default=2048)
    ap.add_argument("--out-dir", type=Path, default=REPO / "out" / "lens_diffraction")
    ap.add_argument("--apertures", nargs="*", default=list(APERTURES))
    args = ap.parse_args()
    od = args.out_dir.resolve()
    od.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in args.apertures:
        ap_mm = APERTURES[name]
        cfg = {
            "lensfile": str(LENS),
            "aperture_diameter_mm": ap_mm,
            "focus_distance_m": FOCUS_M,
            "pixel_pitch_um": PITCH_UM,
            "tile_grid": [TILES, TILES],
        }
        model = ldp.model_from_config(cfg, {}, {}, repo=REPO, film_res=(RES, RES))
        assert abs(model.z_film - model.z_obj - 1e3 * FOCUS_M) < 1e-9
        rects, crops = [], {}
        for fname, (fx, fy, _lab) in FIELDS.items():
            X, Y = object_point(model.lens, model.z_film, model.z_obj, fx, fy)
            rects.append(edge_rect(X, Y, black_side=-1.0))
            col, row = raster_of_film(fx, fy)
            x0, y0 = int(round(col)) - CROP // 2, int(round(row)) - CROP // 2
            crops[fname] = (x0, x0 + CROP, y0, y0 + CROP)
        # Probe: full film at low resolution; check the raster convention puts each edge where expected.
        if name == args.apertures[0]:
            probe_res = 600
            sc = od / "probe.pbrt"
            sc.write_text(scene_text(ap_mm, probe_res, probe_res, None, 16, rects, od / "probe.exr", nbuckets=3))
            render(sc, od)
            img = bucket(read_separate_exr_channels(od / "probe.exr"), LAM)[1]
            scale = probe_res / RES
            for fname, (fx, fy, _lab) in FIELDS.items():
                col, row = raster_of_film(fx, fy)
                c, r = int(col * scale), int(row * scale)
                strip = img[r, c - 8 : c + 9]
                assert strip[:4].mean() < 0.2 * strip[-4:].mean() or strip[-4:].mean() < 0.2 * strip[:4].mean(), (
                    f"raster convention check failed at {fname}: {strip}"
                )
        res_ap = {}
        for fname, (fx, fy, lab) in FIELDS.items():
            exr = od / f"edge_{name.replace('/', '')}_{fname}.exr"
            if not exr.is_file():
                sc = od / f"edge_{name.replace('/', '')}_{fname}.pbrt"
                sc.write_text(scene_text(ap_mm, RES, RES, crops[fname], args.spp, rects, exr))
                render(sc, od)
            full, crop0 = ldp.exr_windows(exr)
            assert full == (RES, RES) and crop0 == (crops[fname][0], crops[fname][2]), (full, crop0)
            chans = {k: v for k, v in read_separate_exr_channels(exr).items() if k.startswith("S0.")}
            key, plane = bucket(chans, LAM)
            diffr = ldp.apply_lens_psf({key: plane}, model, crop_origin=crop0)
            m0 = measure(plane)
            m1 = measure(diffr[key])
            f_mm = m0["frequency"] / (PITCH_UM * 1e-3)
            keep = f_mm <= 0.5 / (PITCH_UM * 1e-3)
            f_mm = f_mm[keep]
            pr = predictions(model, fx, fy, f_mm)
            sim = {}
            for k, (lsf, x) in lsfs(model, fx, fy).items():
                ms = synthetic_edge_mtf(lsf, x)
                sim[k] = np.interp(f_mm, ms["frequency"] / (PITCH_UM * 1e-3), ms["mtf"])
            r0 = m0["mtf"][keep]
            r1 = np.interp(f_mm, m1["frequency"] / (PITCH_UM * 1e-3), m1["mtf"])
            res_ap[fname] = {
                "label": lab,
                "image_height_mm": math.hypot(fx, fy),
                "pupil_area_rel_stop": model.pupil(math.hypot(fx, fy)).pupil_area_rel_stop,
                "working_f_number": model.pupil(math.hypot(fx, fy)).working_f_number,
                "edge_angle_deg": m0["angle_deg"],
                "freq_cy_mm": f_mm.tolist(),
                "mtf_pbrt": r0.tolist(),
                "mtf_pbrt_diffraction": r1.tolist(),
                **{f"pred_{k}": v.tolist() for k, v in pr.items()},
                **{f"sim_{k}": v.tolist() for k, v in sim.items()},
                **{f"mtf50_sim_{k}": mtf50(f_mm, v) for k, v in sim.items()},
                "rms_pbrt_diffraction_vs_sim_wave": float(np.sqrt(np.mean((r1 - sim["wave"]) ** 2))),
                "rms_pbrt_diffraction_vs_sim_geom_x_diff": float(np.sqrt(np.mean((r1 - sim["geom_x_diff"]) ** 2))),
                "rms_pbrt_vs_sim_geom": float(np.sqrt(np.mean((r0 - sim["geom"]) ** 2))),
                "mtf50_pbrt": mtf50(f_mm, r0),
                "mtf50_pbrt_diffraction": mtf50(f_mm, r1),
                "mtf50_pred_wave": mtf50(f_mm, pr["wave"]),
                "mtf50_pred_geom_x_diff": mtf50(f_mm, pr["geom_x_diff"]),
                "mtf50_pred_geom": mtf50(f_mm, pr["geom"]),
                "rms_pbrt_diffraction_vs_wave": float(np.sqrt(np.mean((r1 - pr["wave"]) ** 2))),
                "rms_pbrt_diffraction_vs_geom_x_diff": float(np.sqrt(np.mean((r1 - pr["geom_x_diff"]) ** 2))),
                "rms_pbrt_vs_geom": float(np.sqrt(np.mean((r0 - pr["geom"]) ** 2))),
            }
            d = res_ap[fname]
            print(
                f"{name:6s} {fname:5s} h={d['image_height_mm']:5.2f} mm MTF50 [cy/mm] pbrt {d['mtf50_pbrt']:6.1f} "
                f"+diff {d['mtf50_pbrt_diffraction']:6.1f} | wave {d['mtf50_pred_wave']:6.1f} "
                f"geomxdiff {d['mtf50_pred_geom_x_diff']:6.1f} geom {d['mtf50_pred_geom']:6.1f} | "
                f"rms(+diff - wave) {d['rms_pbrt_diffraction_vs_wave']:.3f}\n"
                f"{'':14s} same-ROI predictions: wave {d['mtf50_sim_wave']:6.1f} geomxdiff "
                f"{d['mtf50_sim_geom_x_diff']:6.1f} geom {d['mtf50_sim_geom']:6.1f} | rms(+diff - wave) "
                f"{d['rms_pbrt_diffraction_vs_sim_wave']:.3f} rms(+diff - geomxdiff) "
                f"{d['rms_pbrt_diffraction_vs_sim_geom_x_diff']:.3f} rms(pbrt - geom) {d['rms_pbrt_vs_sim_geom']:.3f}",
                flush=True,
            )
        results[name] = res_ap
    (od / "results.json").write_text(json.dumps(results, indent=1))
    plot(results, od)


def plot(results: dict, od: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    cols = ["axis", "x5", "y5"]
    fig, axs = plt.subplots(len(names) + 1, 3, figsize=(13, 3.2 * (len(names) + 1)), squeeze=False)
    for i, name in enumerate(names):
        for j, fname in enumerate(cols):
            d, ax = results[name][fname], axs[i, j]
            f = np.array(d["freq_cy_mm"])
            ax.plot(f, d["mtf_pbrt"], color="0.6", lw=1.5, label="pbrt (geometric)")
            ax.plot(f, d["mtf_pbrt_diffraction"], color="C0", lw=2, label="pbrt + diffraction_only")
            ax.plot(f, d["sim_wave"], "k--", lw=1.4, label="lens-design wave-optics MTF")
            ax.plot(f, d["sim_geom_x_diff"], color="C3", ls=":", lw=1.6, label="geometric x diffraction (predicted)")
            ax.plot(f, d["sim_geom"], color="0.3", ls=":", lw=1, label="geometric spot (predicted)")
            ax.plot(
                f, d["pred_diff_only"], color="C2", ls="-.", lw=1, label="diffraction only, analytic (traced pupil)"
            )
            ax.set_ylim(0, 1.02)
            ax.set_xlim(0, f.max())
            ax.grid(alpha=0.3)
            ax.set_title(f"{name}, {d['label']} h={d['image_height_mm']:.1f} mm (N_w={d['working_f_number']:.2f})")
            if j == 0:
                ax.set_ylabel("MTF (550 nm, incl. pixel)")
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="upper right")
    for j, (key, lab) in enumerate((("x", "radial"), ("y", "azimuthal"))):
        ax = axs[-1, j]
        for k, name in enumerate(names):
            pts = [("axis", 0.0)] + [(f"{key}{n}", n * TILE_STEP_MM) for n in (3, 5)]
            h = [p[1] for p in pts]
            ax.plot(
                h,
                [results[name][p[0]]["mtf50_pbrt_diffraction"] for p in pts],
                "o-",
                color=f"C{k}",
                label=f"{name} pbrt+diff",
            )
            ax.plot(
                h,
                [results[name][p[0]]["mtf50_pbrt"] for p in pts],
                "s:",
                color=f"C{k}",
                alpha=0.5,
                label=f"{name} pbrt",
            )
            ax.plot(h, [results[name][p[0]]["mtf50_sim_wave"] for p in pts], "k^", ms=5, mfc="none")
        ax.set_xlabel("image height [mm]")
        ax.set_ylabel("MTF50 [cy/mm]")
        ax.set_title(f"{lab} MTF50 vs field (\u25b3 wave-optics)")
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=7)
    axs[-1, 2].axis("off")
    for ax in axs[-2]:
        ax.set_xlabel("spatial frequency [cy/mm]")
    fig.suptitle(
        "dgauss.50mm.dat at 3.2 m, 2 um pixels, 550 nm: pbrt slanted-edge MTF with / without traced-lens diffraction\n"
        "(predictions passed through the same edge ROI / SFR as the renders; incl. box-pixel MTF)"
    )
    fig.tight_layout()
    fig.savefig(od / "lens_diffraction_validation.png", dpi=110)
    print(f"wrote {od / 'lens_diffraction_validation.png'}")


if __name__ == "__main__":
    main()
