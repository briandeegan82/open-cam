"""Validate the even-asphere pbrt patch (third_party/patches/0004) against the Python ray trace.

Renders the US 8,390,940 B2 phone lens (config/lenses/phone_6p_2.97mm.dat, aspheric on all 12
surfaces, plastic dispersion) imaging a small flat-spectrum emitter on axis and at 1.5 mm image
height (0.67 field), with the machinery of scripts/validate_lens_dispersion.py at a 0.25 um film
pitch (4 mm film), and compares per spectral bucket the PSF centroid (lateral colour, distortion)
and RMS radius with ``lens_diffraction_psf.trace`` of the same aspheric surfaces.

Run from the repository root (needs third_party/pbrt-v4/build/pbrt with the patches; ~1 min):
    venv/bin/python scripts/validate_lens_asphere.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_lens_dispersion as vd  # noqa: E402

vd.PITCH_UM = 0.25
vd.RES = 16000
vd.DIAG_MM = vd.PITCH_UM * 1e-3 * vd.RES * math.sqrt(2)
CASES = {"phone_axis": ("phone_6p_2.97mm.dat", 1.212, 0.0), "phone_1.5mm": ("phone_6p_2.97mm.dat", 1.212, 1.5)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spp", type=int, default=1024)
    ap.add_argument("--out", type=Path, default=None, help="write the comparison as JSON")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as td:
        res = {name: vd.run_case(name, *case, args.spp, Path(td)) for name, case in CASES.items()}
    if args.out:
        args.out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
