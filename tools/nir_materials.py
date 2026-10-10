#!/usr/bin/env python3
"""Measured 350-2500 nm reflectances for a 24-patch NIR material chart (USGS splib07a).

Writes ``spectra/usgs_nir/NN_<name>.csv`` (wavelength_nm, reflectance; 350-1100 nm, 1 nm) from the
USGS Spectral Library v7 ASCII archive (Kokaly et al. 2017, USGS Data Series 1035,
doi:10.5066/F7RR1WDJ; public domain). All records are ASD FieldSpec absolute reflectance (AREF).
The chart uses the ColorChecker layout, so ``build_colorchecker_scene.py --patch-dir spectra/usgs_nir``
renders it. Unlike ``spectra/xrite`` (380-730 nm), these curves are measured through 1100 nm.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
from highway_materials import USGS_WAVELENGTHS, USGS_ZIP_SHA256, USGS_ZIP_URL, cache_dir

REPO = Path(__file__).resolve().parents[1]
OUT_REL = "spectra/usgs_nir"
_A = "ASCIIdata_splib07a/ChapterA_ArtificialMaterials/splib07a_"
_V = "ASCIIdata_splib07a/ChapterV_Vegetation/splib07a_"

#: chart patch name -> splib07a member (row-major, 6 x 4, same order as the ColorChecker slots).
NIR_CHART = {
    "oak_leaf_fresh": _V + "Oak_Oak-Leaf-1_fresh_ASDFRa_AREF.txt",
    "aspen_leaf_green": _V + "Aspen_Aspen-1_green-top_ASDFRa_AREF.txt",
    "pine_needles": _V + "Lodgepole-Pine_LP-Needles-1_ASDFRa_AREF.txt",
    "grass_dry_golden": _V + "Grass_Golden_Dry_GDS480_ASDFRa_AREF.txt",
    "plywood_pine": _A + "Plywood_GDS365_Fresh_Pine_ASDFRa_AREF.txt",
    "burlap_brown": _A + "Burlap_Fabric_GDS430_Brown_ASDFRa_AREF.txt",
    "nylon_red": _A + "Nylon_Fabric_GDS431_Red_RpSt_ASDFRa_AREF.txt",
    "nylon_green": _A + "Nylon_Fabric_GDS432_Grn_RpSt_ASDFRa_AREF.txt",
    "nylon_blue": _A + "Nylon_Fabric_GDS433_Blu_RpSt_ASDFRa_AREF.txt",
    "nylon_black_coated": _A + "Nylon_Fabric_GDS436_BlkCoatd_ASDFRa_AREF.txt",
    "cotton_white": _A + "Cotton_Fabric_GDS437_White_ASDFRa_AREF.txt",
    "tarp_green": _A + "Plastic_Tarp_GDS339_Green_ASDFRa_AREF.txt",
    "vinyl_red": _A + "Plastic_Vinyl_GDS398_Red_Toy_ASDFRa_AREF.txt",
    "hdpe_yellow": _A + "Plastic_HDPE_GDS386_Yelopaqu_ASDFRa_AREF.txt",
    "pipe_blue": _A + "Plastic_Pipe_GDS344_Blue_ASDFRa_AREF.txt",
    "hdpe_black": _A + "Plastic_HDPE_GDS351_BlkSheet_ASDFRa_AREF.txt",
    "pvc_white": _A + "Plastic_PVC_GDS338_White_ASDFRa_AREF.txt",
    "painted_aluminium_green": _A + "Painted_Aluminum_GDS333_LgGr_ASDFRa_AREF.txt",
    "brick_red": _A + "Brick_GDS350_Dk_Red_Building_ASDFRa_AREF.txt",
    "brick_grey": _A + "Brick_GDS354_Building_Lt_Gry_ASDFRa_AREF.txt",
    "concrete_road": _A + "Concrete_GDS375_Lt_Gry_Road_ASDFRa_AREF.txt",
    "asphalt_road": _A + "Asphalt_GDS376_Blck_Road_old_ASDFRa_AREF.txt",
    "tar_paper_black": _A + "Tar_Paper_GDS285B_Weath_Blk_ASDFRa_AREF.txt",
    "spectralon_white": _A + "Spectralon99WhiteRef_LSPHERE_ASDFRa_AREF.txt",
}


def usgs_zip(repo: Path = REPO) -> Path:
    """The sha256-checked splib07a ASCII archive in the gitignored cache (downloaded if missing)."""
    from fetch_highway_assets import _download

    z = cache_dir(repo) / "usgs" / "splib07a_ASCII.zip"
    if not z.is_file():
        _download(USGS_ZIP_URL, z, USGS_ZIP_SHA256)
    return z


def resample(lam_nm: np.ndarray, val: np.ndarray, wl: np.ndarray) -> np.ndarray:
    """Drop USGS bad channels (-1.23e34) and interpolate onto ``wl``."""
    ok = (val > -1.0) & (val < 2.0)
    return np.interp(wl, lam_nm[ok], val[ok])


def write_chart(repo: Path = REPO, lo: float = 350.0, hi: float = 1100.0) -> list[Path]:
    wl = np.arange(lo, hi + 0.5, 1.0)
    out = Path(repo) / OUT_REL
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    with zipfile.ZipFile(usgs_zip(repo)) as z:
        lam = np.loadtxt(z.open(USGS_WAVELENGTHS), skiprows=1) * 1000.0
        for i, (name, member) in enumerate(NIR_CHART.items(), start=1):
            r = resample(lam, np.loadtxt(z.open(member), skiprows=1), wl)
            p = out / f"{i:02d}_{name}.csv"
            np.savetxt(p, np.column_stack([wl, r]), fmt=["%.0f", "%.5f"], delimiter=",")
            paths.append(p)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo-root", type=Path, default=REPO)
    args = ap.parse_args()
    for p in write_chart(args.repo_root):
        print(p.relative_to(args.repo_root))


if __name__ == "__main__":
    main()
