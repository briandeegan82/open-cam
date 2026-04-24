#!/usr/bin/env python3
"""Split legacy camera models into lens models, sensor models, and camera recipes."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml


def _stable_hash(data: dict) -> str:
    text = yaml.safe_dump(data, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_yaml(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text()) or {}
    if not isinstance(cfg, dict):
        raise TypeError(f"Expected YAML mapping in {path}")
    return cfg


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _write_index(path: Path, title: str, names: list[str]) -> None:
    lines = [f"# {title}", ""]
    for name in sorted(names):
        lines.append(f"- {name}")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    repo_default = Path(__file__).resolve().parent.parent
    ap.add_argument("--repo-root", type=Path, default=repo_default)
    ap.add_argument("--src-dir", type=Path, default=None, help="Defaults to config/camera_models")
    ap.add_argument("--lens-dir", type=Path, default=None, help="Defaults to config/lens_models")
    ap.add_argument("--sensor-dir", type=Path, default=None, help="Defaults to config/sensor_models")
    ap.add_argument("--recipes-dir", type=Path, default=None, help="Defaults to config/camera_recipes")
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    src = (args.src_dir or (repo / "config" / "camera_models")).resolve()
    lens_dir = (args.lens_dir or (repo / "config" / "lens_models")).resolve()
    sensor_dir = (args.sensor_dir or (repo / "config" / "sensor_models")).resolve()
    recipes_dir = (args.recipes_dir or (repo / "config" / "camera_recipes")).resolve()

    lens_by_hash: dict[str, str] = {}
    sensor_by_hash: dict[str, str] = {}

    created_lens = 0
    created_sensor = 0
    created_recipes = 0
    lens_names: set[str] = set()
    sensor_names: set[str] = set()
    recipe_names: set[str] = set()

    for model_path in sorted(src.glob("*.yaml")):
        name = model_path.stem
        cfg = _read_yaml(model_path)
        if "lens" not in cfg:
            continue
        if not all(k in cfg for k in ("sensor", "noise", "cfa", "sensor_forward")):
            continue

        lens_payload = {"schema_version": 1, "lens": cfg["lens"]}
        sensor_payload = {
            "schema_version": 1,
            "sensor": cfg["sensor"],
            "noise": cfg["noise"],
            "cfa": cfg["cfa"],
            "sensor_forward": cfg["sensor_forward"],
        }
        if "validation" in cfg:
            sensor_payload["validation"] = cfg["validation"]
        if "source" in cfg:
            sensor_payload["source"] = cfg["source"]
        if "model" in cfg:
            sensor_payload["model"] = cfg["model"]

        lens_hash = _stable_hash(lens_payload)
        sensor_hash = _stable_hash(sensor_payload)

        lens_name = lens_by_hash.get(lens_hash)
        if lens_name is None:
            lens_name = name
            lens_by_hash[lens_hash] = lens_name
            _write_yaml(lens_dir / f"{lens_name}.yaml", lens_payload)
            created_lens += 1
        lens_names.add(lens_name)

        sensor_name = sensor_by_hash.get(sensor_hash)
        if sensor_name is None:
            sensor_name = name
            sensor_by_hash[sensor_hash] = sensor_name
            _write_yaml(sensor_dir / f"{sensor_name}.yaml", sensor_payload)
            created_sensor += 1
        sensor_names.add(sensor_name)

        recipe_payload = {
            "schema_version": 1,
            "model": cfg.get("model", {"name": name, "display_name": name}),
            "lens_model": lens_name,
            "sensor_model": sensor_name,
            "source": cfg.get("source", {"migrated_from": f"config/camera_models/{name}.yaml"}),
        }
        _write_yaml(recipes_dir / f"{name}.yaml", recipe_payload)
        created_recipes += 1
        recipe_names.add(name)

    _write_index(lens_dir / "INDEX.md", "Lens Models", sorted(lens_names))
    _write_index(sensor_dir / "INDEX.md", "Sensor Models", sorted(sensor_names))
    _write_index(recipes_dir / "INDEX.md", "Camera Recipes", sorted(recipe_names))

    print(f"Created lens models: {created_lens}")
    print(f"Created sensor models: {created_sensor}")
    print(f"Created camera recipes: {created_recipes}")


if __name__ == "__main__":
    main()
