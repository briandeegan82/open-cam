"""Calibrate pbrt coateddiffuse so the rendered reflectance matches a measured reflectance.

pbrt's ``coateddiffuse`` takes the reflectance of the substrate under the coat, not the reflectance of
the finished surface. Light trapped under the coat and the energy a rough GGX interface loses (single
scattering only) make the surface much darker than the input: at roughness 0.35 and eta 1.5 an input
of 0.12 renders 0.069, and 0.8 renders 0.38. A measured spectrum is the reflectance of the finished
surface (near-normal incidence, hemispherical), so the builders pass it through ``base_reflectance``.

``config/coat_albedo.json`` holds the rendered directional-hemispherical reflectance at normal incidence,
by eta, roughness and substrate reflectance: a plate under a uniform environment, viewed head-on.
Regenerate it with ``venv/bin/python tools/coat_calibration.py``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TABLE = REPO / "config" / "coat_albedo.json"
ETAS = (1.333, 1.5)
ROUGHNESS = (0.0, 0.001, 0.01, 0.038, 0.08, 0.15, 0.3, 0.35, 0.4, 0.45)
SUBSTRATE = (0.0, 0.03, 0.06, 0.12, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0)
# Measured finished-surface spectra -> nominal coat roughness in the highway scenes. Not the car paints:
# their spectra are pigment bodies under a separate clear coat (carpaint_black is below the coat's own 0.04).
COAT_ROUGHNESS = {
    "asphalt_aged": 0.35,
    "asphalt_new": 0.35,
    "asphalt_patch": 0.35,
    "crack_sealant": 0.08,
    "paint_road_white": 0.35,
    "paint_road_yellow": 0.35,
    "galvanized": 0.3,
    "rubber": 0.45,
}


@lru_cache(maxsize=1)
def _table() -> dict:
    return json.loads(TABLE.read_text())


def rendered_reflectance(substrate, eta: float, roughness: float) -> np.ndarray:
    """Rendered reflectance of a coateddiffuse with this substrate (bilinear in roughness and substrate)."""
    t = _table()
    tab = np.asarray(t["reflectance"][str(eta)], dtype=np.float64)
    r = np.asarray(t["roughness"], dtype=np.float64)
    if not r[0] <= roughness <= r[-1]:
        raise ValueError(f"roughness {roughness} outside the calibrated range")
    row = np.array([np.interp(roughness, r, tab[:, j]) for j in range(tab.shape[1])])
    return np.interp(substrate, t["substrate"], row)


def max_reflectance(eta: float, roughness: float) -> float:
    """The brightest surface this coat can render (substrate reflectance 1)."""
    return float(rendered_reflectance(1.0, eta, roughness))


def base_reflectance(target, eta: float = 1.5, roughness: float = 0.0) -> np.ndarray:
    """Substrate reflectance that renders ``target``; clipped to [0, 1] where the coat can't reach it."""
    s = np.asarray(_table()["substrate"], dtype=np.float64)
    return np.interp(target, rendered_reflectance(s, eta, roughness), s)


def calibrate_scene(lines: list[str], spd_dir: Path, write_spd) -> tuple[list[str], dict]:
    """Point the coated uses of the measured spectra at their calibrated substrate spectra.

    Runs on the finished scene: a statement counts if it is a Texture (they feed the coated road) or a
    coateddiffuse; the Lambertian tires and lamp housings keep the measured spectrum. Returns the new
    lines and, per spectrum, the coat used and the fraction of the spectrum the coat cannot reach.
    """
    meta = {}
    for name, rough in COAT_ROUGHNESS.items():
        src = spd_dir / f"{name}.spd"
        if src.is_file():
            d = np.loadtxt(src)
            write_spd(spd_dir / f"coat_{name}.spd", d[:, 0], base_reflectance(d[:, 1], 1.5, rough))
            over = d[:, 1] > max_reflectance(1.5, rough)
            meta[name] = {"roughness": rough, "unreachable_fraction": round(float(np.mean(over)), 3)}
    spd_re = re.compile(r'"spd/(' + "|".join(meta) + r')\.spd"') if meta else None
    out, coated = [], False
    for ln in lines:
        if not ln[:1].isspace():
            coated = ln.startswith("Texture ") or "coateddiffuse" in ln
        elif "coateddiffuse" in ln:
            coated = True
        out.append(spd_re.sub(r'"spd/coat_\1.spd"', ln) if coated and spd_re else ln)
    return out, meta


def _furnace(pbrt: str, tmp: Path, material: str) -> float:
    from exr_multispectral import read_separate_exr_channels  # noqa: PLC0415

    (tmp / "plate.pbrt").write_text(
        "LookAt 0 0 1  0 0 0  0 1 0\n"
        'Camera "orthographic" "float screenwindow" [-0.5 0.5 -0.5 0.5]\n'
        'Sampler "independent" "integer pixelsamples" [4096]\n'
        'Integrator "path" "integer maxdepth" [32]\n'
        f'Film "rgb" "integer xresolution" [8] "integer yresolution" [8] "string filename" "{tmp}/plate.exr"\n'
        'WorldBegin\nLightSource "infinite" "spectrum L" [300 1 800 1]\n'
        f'{material}\nShape "disk" "float radius" [100]\n'
    )
    subprocess.run([pbrt, "--quiet", str(tmp / "plate.pbrt")], check=True, stdout=subprocess.DEVNULL)
    c = read_separate_exr_channels(tmp / "plate.exr")
    return float(np.mean([c[k].mean() for k in ("R", "G", "B")]))


def generate(pbrt: str) -> dict:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        white = _furnace(pbrt, tmp, 'Material "diffuse" "spectrum reflectance" [300 1 800 1]')
        table = {}
        for eta in ETAS:
            rows = []
            for r in ROUGHNESS:
                row = [
                    _furnace(
                        pbrt,
                        tmp,
                        f'Material "coateddiffuse" "spectrum reflectance" [300 {a} 800 {a}]'
                        f' "float roughness" [{r}] "float eta" [{eta}] "float thickness" [0.0001]',
                    )
                    / white
                    for a in SUBSTRATE
                ]
                rows.append([round(v, 5) for v in row])
                print(eta, r, rows[-1], flush=True)
            table[str(eta)] = rows
    return {"roughness": ROUGHNESS, "substrate": SUBSTRATE, "reflectance": table}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pbrt", default=str(REPO / "third_party" / "pbrt-v4" / "build" / "pbrt"))
    args = ap.parse_args()
    TABLE.write_text(json.dumps(generate(args.pbrt), indent=1) + "\n")


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "tools"))
    main()
