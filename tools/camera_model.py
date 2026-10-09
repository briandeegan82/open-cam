#!/usr/bin/env python3
"""Helpers for loading and projecting camera model configs.

Inheritance model
-----------------
Sensor model YAMLs (``config/sensor_models/*.yaml``) and lens model YAMLs
(``config/lens_models/*.yaml``) are loaded with automatic **deep-merge
inheritance** from a ``default.yaml`` base in the same directory.

Deep-merge rules
~~~~~~~~~~~~~~~~
* Dict values are merged recursively; the specific YAML wins on any conflict.
* All other types (scalars, lists, ``null``) in the specific YAML fully replace
  the corresponding value from the base.
* Lists are **not** appended — a list in the specific YAML entirely replaces the
  base list.  This keeps validation level lists (``ptc_mu_e_levels``) predictable.
* ``default.yaml`` itself does **not** inherit from anything (no self-reference).

Writing a new camera model
~~~~~~~~~~~~~~~~~~~~~~~~~~
A minimal sensor model only needs to declare keys that differ from
``config/sensor_models/default.yaml``.  Keys that match the default can be
omitted entirely — they will be filled in from the base at load time.  Existing
full YAMLs continue to work unchanged (merging a complete override on top of the
base is identical to using the override alone).
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from lens_prescription import traced_f_number

_SENSOR_DEFAULTS_NAME = "default.yaml"
_LENS_DEFAULTS_NAME = "default.yaml"
_REPO = Path(__file__).resolve().parents[1]
_LEGACY_REALISTIC_LENSFILE = "scenes/lenses/wide_22mm.dat"
_DEFAULT_REALISTIC_LENSFILE = "config/lenses/wide_22mm.dat"


def _load_yaml_mapping(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text()) or {}
    if not isinstance(cfg, dict):
        raise TypeError(f"YAML file must be a mapping: {path}")
    return cfg


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* on top of *base*.

    Dict values are merged recursively.  All other types (scalars, lists,
    ``null``) in *override* fully replace the corresponding *base* value.
    Returns a new dict; neither input is mutated.
    """
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _load_with_defaults(path: Path, defaults_path: Path) -> dict:
    """Load *path*, deep-merging it on top of *defaults_path* (if it exists).

    If *path* IS the defaults file, or the defaults file does not exist, the
    file is loaded without merging (avoids self-reference loops).
    """
    specific = _load_yaml_mapping(path)
    if not defaults_path.is_file() or path.resolve() == defaults_path.resolve():
        return specific
    base = _load_yaml_mapping(defaults_path)
    return deep_merge(base, specific)


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
        # Full in-file model: deep-merge from defaults if a defaults file lives
        # alongside (edge case: full camera model YAMLs in config/camera_models/).
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

    # Load each model file with defaults inheritance.
    sensor_defaults = sensor_path.parent / _SENSOR_DEFAULTS_NAME
    lens_defaults = lens_path.parent / _LENS_DEFAULTS_NAME

    lens_cfg = _load_with_defaults(lens_path, lens_defaults)
    sensor_cfg = _load_with_defaults(sensor_path, sensor_defaults)

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
        "sensor_defaults": str(sensor_defaults) if sensor_defaults.is_file() else None,
        "lens_defaults": str(lens_defaults) if lens_defaults.is_file() else None,
    }
    return composed


def noise_config_from_camera_model(camera_model: dict, linear_rgb_in: str, raw_out: str) -> dict:
    return {
        "schema_version": 1,
        "sensor": camera_model.get("sensor", {}),
        "emva": camera_model.get("noise", {}).get("emva", {}),
        "adc": camera_model.get("noise", {}).get("adc", {}),
        "processing": camera_model.get("noise", {}).get("processing", {}),
        "hdr": camera_model.get("noise", {}).get("hdr", {}),
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


def effective_f_number(sensor_cfg: dict, lens_cfg: dict | None, *, default: float = 2.8, tag: str = "") -> float:
    """f-number for radiance-to-irradiance conversion.

    For PBRT ``realistic`` cameras the solid angle is set by the traced lens, so the
    f-number is the lens prescription's traced on-axis working f-number at
    ``realistic_aperture_diameter_mm`` (see ``lens_prescription.traced_f_number``),
    not the nominal ``sensor.f_number`` label. Without a readable lens file it falls
    back to ``focal_length_mm / realistic_aperture_diameter_mm``.
    """
    f_number = float(sensor_cfg.get("f_number", default))
    if not lens_cfg or str(lens_cfg.get("camera", "pinhole")).lower() != "realistic":
        return f_number
    fl = lens_cfg.get("focal_length_mm")
    ap = lens_cfg.get("realistic_aperture_diameter_mm")
    lens_file = resolve_lensfile(lens_cfg.get("realistic_lensfile"))
    prefix = f"warning [{tag}]" if tag else "warning"
    if lens_file is not None:
        try:
            return traced_f_number(str(lens_file), None if ap is None else float(ap))
        except ValueError as exc:
            print(f"{prefix}: cannot trace {lens_file} ({exc}); using focal/aperture.", file=sys.stderr)
    if fl is not None and ap is not None and float(ap) > 0:
        return float(fl) / float(ap)
    print(
        f"{prefix}: realistic camera lens model missing focal_length_mm — falling back to sensor.f_number={f_number}.",
        file=sys.stderr,
    )
    return f_number


def resolve_lensfile(lensfile: object) -> Path | None:
    """Absolute path of a lens model's ``realistic_lensfile`` (repo-relative), or None if absent."""
    if not lensfile:
        return None
    rel = str(lensfile)
    if rel == _LEGACY_REALISTIC_LENSFILE:
        rel = _DEFAULT_REALISTIC_LENSFILE
    path = Path(rel)
    if not path.is_absolute():
        path = _REPO / path
    return path if path.is_file() else None
