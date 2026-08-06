#!/usr/bin/env python3
"""Add processed sRGB (D65 white balance + gamma) images to an existing patch-pair
EMVA dataset, from the linear16 files already present — no re-rendering.

Writes clean_srgb8.png / noisy_srgb8.png per level and a per-pair contact.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import importlib.util

spec = importlib.util.spec_from_file_location("gpp", REPO / "scripts" / "generate_patch_pair_dataset.py")
gpp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpp)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=REPO / "out" / "patch_pairs_emva")
    args = ap.parse_args()
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    lux_levels = manifest["lux_levels"]
    for entry in manifest["pairs"]:
        pair_dir = args.dataset / entry["dir"]
        gpp.process_pair_srgb(pair_dir, lux_levels)
        print(f"  {entry['dir']}: wrote clean_srgb8/noisy_srgb8 + contact", flush=True)
    print(f"Done: {len(manifest['pairs'])} pairs -> {args.dataset}", flush=True)


if __name__ == "__main__":
    main()
