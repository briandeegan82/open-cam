#!/usr/bin/env python3
"""Assign a physics-appropriate lens model to every camera recipe.

Assignment logic (highest priority first):
  1. sensor_class from the sensor model's ``source.sensor_class`` field.
  2. pixel_pitch_um from the merged sensor model (inheriting sensor_models/default.yaml).

Sensor class → lens model mapping
----------------------------------
  phone      → phone_wide_f18         (wide_22mm, f/1.8, 12.2 mm aperture)
  compact    → compact_wide_f28       (wide_22mm, f/2.8, 7.9 mm aperture)
  mft / apsc → normal_dgauss_50mm_f28 (dgauss.50mm, f/2.8, 17.9 mm aperture)
  fullframe / old / (≥4.5 µm) → normal_dgauss_50mm_f20 (dgauss.50mm, f/2.0, 25.0 mm aperture)

Recipes whose stem starts with ``research_``, ``study_``, or ``default`` are
skipped — they use deliberately chosen lens types for controlled experiments.

Usage
-----
  # preview changes without writing
  python tools/assign_lens_models.py --dry-run

  # apply
  python tools/assign_lens_models.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent

# Lens model name by sensor class label.
_CLASS_TO_LENS: dict[str, str] = {
    "phone": "phone_wide_f18",
    "compact": "compact_wide_f28",
    "mft": "normal_dgauss_50mm_f28",
    "apsc": "normal_dgauss_50mm_f28",
    "fullframe": "normal_dgauss_50mm_f20",
    "old": "normal_dgauss_50mm_f20",
}

# Recipe stems that should never be rewritten.
_SKIP_PREFIXES = ("research_", "study_", "default")


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _load_sensor_model(name: str) -> dict:
    sensor_dir = REPO / "config" / "sensor_models"
    default_path = sensor_dir / "default.yaml"
    base: dict = yaml.safe_load(default_path.read_text()) or {} if default_path.is_file() else {}
    path = sensor_dir / f"{name}.yaml"
    if not path.is_file():
        return base
    override: dict = yaml.safe_load(path.read_text()) or {}
    return _deep_merge(base, override)


def _pitch_to_lens(pitch_um: float) -> str:
    if pitch_um < 1.5:
        return "phone_wide_f18"
    if pitch_um < 3.0:
        return "compact_wide_f28"
    if pitch_um < 4.0:
        # APS-C and Micro Four Thirds sensors top out around 3.9 µm.
        # Canon 5DsR is full-frame at 4.14 µm, so the cutoff is 4.0 µm.
        return "normal_dgauss_50mm_f28"
    return "normal_dgauss_50mm_f20"


def _assign(sensor_model_name: str) -> str:
    cfg = _load_sensor_model(sensor_model_name)
    sensor_class = (cfg.get("source") or {}).get("sensor_class", "").lower().strip()
    if sensor_class in _CLASS_TO_LENS:
        return _CLASS_TO_LENS[sensor_class]
    pitch = (cfg.get("sensor") or {}).get("pixel_pitch_um")
    if pitch is not None:
        return _pitch_to_lens(float(pitch))
    # Fallback: no class, no explicit pitch — assume default sensor (1.4 µm phone-like).
    return "phone_wide_f18"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="Print changes without writing files.")
    ap.add_argument(
        "--skip",
        nargs="*",
        default=[],
        metavar="STEM",
        help="Additional recipe stems to skip (on top of research_/study_/default prefixes).",
    )
    args = ap.parse_args()

    recipe_dir = REPO / "config" / "camera_recipes"
    updated = 0
    skipped = 0
    already_ok = 0

    for recipe_path in sorted(recipe_dir.glob("*.yaml")):
        stem = recipe_path.stem
        if any(stem.startswith(p) for p in _SKIP_PREFIXES) or stem in args.skip:
            skipped += 1
            continue

        text = recipe_path.read_text()
        cfg = yaml.safe_load(text) or {}
        sensor_name = cfg.get("sensor_model", stem)
        new_lens = _assign(sensor_name)
        old_lens = cfg.get("lens_model", "")

        if old_lens == new_lens:
            print(f"  ok  {stem:<42}  {old_lens}")
            already_ok += 1
            continue

        marker = ">>" if not args.dry_run else "--"
        print(f"  {marker}  {stem:<42}  {old_lens!r}  →  {new_lens!r}")

        if not args.dry_run:
            new_text = re.sub(
                r"^(lens_model:\s*).*$",
                f"lens_model: {new_lens}",
                text,
                flags=re.MULTILINE,
            )
            recipe_path.write_text(new_text)
            updated += 1

    print()
    if args.dry_run:
        print(f"Dry run: {already_ok} already correct, {skipped} skipped.")
    else:
        print(f"Updated {updated} recipes, {already_ok} already correct, {skipped} skipped.")


if __name__ == "__main__":
    main()
