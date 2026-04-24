#!/usr/bin/env python3
"""Helpers for loading and projecting camera model configs."""

from __future__ import annotations

from pathlib import Path

import yaml


def _load_yaml_mapping(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text()) or {}
    if not isinstance(cfg, dict):
        raise TypeError(f"YAML file must be a mapping: {path}")
    return cfg


def _require_sections(cfg: dict, path: Path, sections: tuple[str, ...]) -> None:
    for key in sections:
        if key not in cfg:
            raise KeyError(f"missing required section '{key}' in {path}")


def _resolve_model_ref(repo: Path, root: Path, subdir: str, ref: str) -> Path:
    raw = Path(str(ref))
    if raw.is_absolute():
        return raw.resolve()
    if "/" in str(ref):
        return (root / raw).resolve()
    return (repo / "config" / subdir / f"{ref}.yaml").resolve()


def load_camera_model(path: Path) -> dict:
    cfg = _load_yaml_mapping(path)
    required_full = ("lens", "sensor", "noise", "cfa", "sensor_forward")
    if all(key in cfg for key in required_full):
        return cfg

    # Recipe mode: compose a full in-memory camera model from split files.
    if "lens_model" not in cfg or "sensor_model" not in cfg:
        missing = [k for k in required_full if k not in cfg]
        raise KeyError(f"camera model missing required section(s): {', '.join(missing)}")

    root = path.parent
    repo = root.parent.parent if root.parent.name == "config" else root
    lens_path = _resolve_model_ref(repo, root, "lens_models", str(cfg["lens_model"]))
    sensor_path = _resolve_model_ref(repo, root, "sensor_models", str(cfg["sensor_model"]))
    if not lens_path.is_file():
        raise FileNotFoundError(f"missing lens model config: {lens_path}")
    if not sensor_path.is_file():
        raise FileNotFoundError(f"missing sensor model config: {sensor_path}")

    lens_cfg = _load_yaml_mapping(lens_path)
    sensor_cfg = _load_yaml_mapping(sensor_path)
    _require_sections(lens_cfg, lens_path, ("lens",))
    _require_sections(sensor_cfg, sensor_path, ("sensor", "noise", "cfa", "sensor_forward"))

    composed = {
        "schema_version": cfg.get("schema_version", 1),
        "model": cfg.get("model", {}),
        "lens": lens_cfg["lens"],
        "sensor": sensor_cfg["sensor"],
        "noise": sensor_cfg["noise"],
        "cfa": sensor_cfg["cfa"],
        "sensor_forward": sensor_cfg["sensor_forward"],
    }
    if "validation" in sensor_cfg:
        composed["validation"] = sensor_cfg["validation"]
    if "source" in sensor_cfg or "source" in cfg:
        composed["source"] = {}
        if isinstance(sensor_cfg.get("source"), dict):
            composed["source"].update(sensor_cfg["source"])
        if isinstance(cfg.get("source"), dict):
            composed["source"].update(cfg["source"])
    composed["resolved_from"] = {
        "recipe": str(path),
        "lens_model": str(lens_path),
        "sensor_model": str(sensor_path),
    }
    return composed


def noise_config_from_camera_model(camera_model: dict, linear_rgb_in: str, raw_out: str) -> dict:
    return {
        "schema_version": 1,
        "sensor": camera_model.get("sensor", {}),
        "emva": camera_model.get("noise", {}).get("emva", {}),
        "adc": camera_model.get("noise", {}).get("adc", {}),
        "processing": camera_model.get("noise", {}).get("processing", {}),
        "bayer": camera_model.get("cfa", {}),
        "output": {
            "linear_rgb_in": linear_rgb_in,
            "raw_out": raw_out,
        },
    }


def sensor_forward_config_from_camera_model(
    camera_model: dict,
    spectral_reference_npz: str,
    scene_manifest_json: str,
    electrons_npz: str,
) -> dict:
    return {
        "schema_version": 1,
        "inputs": {
            "spectral_reference_npz": spectral_reference_npz,
            "scene_manifest_json": scene_manifest_json,
        },
        "model": camera_model.get("sensor_forward", {}).get("model", {}),
        "output": {"electrons_npz": electrons_npz},
    }

