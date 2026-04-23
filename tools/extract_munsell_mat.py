#!/usr/bin/env python3
"""Extract reflectance spectra from the Joensuu Munsell matte MATLAB file.

Expected variables (see ``spectra/munsell/README.txt``):

- ``munsell`` — (421, 1269) reflectance, 380–800 nm, 1 nm; one column per chip.
- ``S`` — length-1269 array of Unicode labels (e.g. ``2.5R 9/2``).
- ``C`` — (16, 1269) colorimetry under D65 (optional metadata in manifest).

Default input: ``spectra/munsell/munsell380_800_1.mat`` (MATLAB v4 or newer ``.mat``).

Requires: ``pip install scipy`` (listed in ``requirements.txt``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from munsell_mat import load_joensuu_mat, sanitize_filename


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mat",
        type=Path,
        default=None,
        help="Path to munsell380_800_1.mat (default: <repo>/spectra/munsell/munsell380_800_1.mat)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for per-chip CSV files (default: <repo>/spectra/munsell/csv)",
    )
    ap.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="Optional path to write combined archive: wavelength_nm, reflectance (421x1269), labels.",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="JSON manifest of chips (default: <out-dir>/manifest.json)",
    )
    ap.add_argument(
        "--max-chips",
        type=int,
        default=None,
        help="Export only the first N chips (for quick tests).",
    )
    args = ap.parse_args()

    mat_path = (args.mat or (root / "spectra" / "munsell" / "munsell380_800_1.mat")).resolve()
    bundle = load_joensuu_mat(mat_path)

    spec = bundle.reflectance
    wl = bundle.wavelength_nm
    str_labels = bundle.labels
    C = bundle.C_D65

    n = spec.shape[1]
    if args.max_chips is not None:
        n = min(n, max(0, args.max_chips))
        spec = spec[:, :n]
        str_labels = str_labels[:n]
        if C is not None:
            C = C[:, :n]

    out_dir = (args.out_dir or (root / "spectra" / "munsell" / "csv")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    for i in range(n):
        lab = str_labels[i]
        base = f"{i + 1:04d}_{sanitize_filename(lab)}"
        csv_name = f"{base}.csv"
        csv_path = out_dir / csv_name
        lines_csv = ["# wavelength_nm,value (Joensuu Munsell matte; see spectra/munsell/README.txt)"]
        lines_csv.extend(f"{float(w):.6g},{float(v):.12g}" for w, v in zip(wl, spec[:, i]))
        csv_path.write_text("\n".join(lines_csv) + "\n")
        entry: dict = {"index": i + 1, "label": lab, "csv": csv_name}
        if C is not None:
            entry["C_D65_xyY"] = [float(C[j, i]) for j in range(min(3, C.shape[0]))]
        manifest.append(entry)

    man_path = (args.manifest or (out_dir / "manifest.json")).resolve()
    man_path.write_text(json.dumps({"source_mat": str(mat_path), "chips": manifest}, indent=2) + "\n")

    if args.npz is not None:
        npz_path = args.npz if args.npz.is_absolute() else (root / args.npz).resolve()
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict = {
            "wavelength_nm": wl.astype(np.float32),
            "reflectance": spec.astype(np.float32),
            "labels": np.array(str_labels, dtype=object),
        }
        if C is not None:
            payload["C_D65"] = C.astype(np.float32)
        np.savez_compressed(npz_path, **payload)

    print(f"read {mat_path}", file=sys.stderr)
    print(f"wrote {n} CSV files under {out_dir}", file=sys.stderr)
    print(f"wrote {man_path}", file=sys.stderr)
    if args.npz:
        print(f"wrote {npz_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
