#!/usr/bin/env python3
"""Helpers for loading and projecting camera model configs."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_camera_model(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text()) or {}
    if not isinstance(cfg, dict):
        raise TypeError(f"camera model must be a YAML mapping: {path}")
    for key in ("lens", "sensor", "noise", "cfa", "sensor_forward"):
        if key not in cfg:
            raise KeyError(f"camera model missing required section: {key}")
    return cfg


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

