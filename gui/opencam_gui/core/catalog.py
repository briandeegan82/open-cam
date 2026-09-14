"""Read-only listings of the config/spectra assets shipped with Open Cam.

Nothing here computes camera physics; it only walks ``config/`` and
``spectra/`` so the GUIs can populate dropdowns from what actually exists on
disk instead of a hard-coded list that can drift out of sync.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from opencam_gui.core.repo import config_dir, import_tool, spectra_dir


@dataclass(frozen=True)
class CameraRecipe:
    id: str
    display_name: str
    path: Path


def _yaml_display_name(path: Path, default_id: str) -> str:
    try:
        cfg = yaml.safe_load(path.read_text()) or {}
    except Exception:
        return default_id
    model = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    if isinstance(model, dict) and model.get("display_name"):
        return str(model["display_name"])
    return default_id


def list_camera_recipes() -> list[CameraRecipe]:
    """All ``config/camera_recipes/*.yaml`` (skips ``INDEX.md``)."""
    out: list[CameraRecipe] = []
    d = config_dir() / "camera_recipes"
    for path in sorted(d.glob("*.yaml")):
        rid = path.stem
        out.append(CameraRecipe(id=rid, display_name=_yaml_display_name(path, rid), path=path))
    return out


def find_recipe(recipe_id: str) -> CameraRecipe:
    for r in list_camera_recipes():
        if r.id == recipe_id:
            return r
    raise KeyError(f"unknown camera recipe: {recipe_id!r}")


@dataclass(frozen=True)
class Illuminant:
    id: str
    path: Path
    repo_relative: str


_ILLUMINANT_LABELS = {
    "A": "A  -  incandescent tungsten (2856 K)",
    "C": "C  -  average daylight (obsolete)",
    "D50": "D50  -  horizon light / print viewing (5003 K)",
    "D55": "D55  -  mid-morning/afternoon daylight (5503 K)",
    "D65": "D65  -  noon daylight, sRGB reference (6504 K)",
    "D75": "D75  -  north-sky daylight (7504 K)",
    "F2_CWF": "F2  -  cool white fluorescent",
    "F7": "F7  -  broadband daylight fluorescent",
    "F11": "F11  -  narrow tri-band fluorescent",
    "LED_B1": "LED B1  -  phosphor-converted, blue-rich",
    "LED_B2": "LED B2  -  phosphor-converted",
    "LED_B3": "LED B3  -  phosphor-converted",
    "LED_B4": "LED B4  -  phosphor-converted",
    "LED_B5": "LED B5  -  phosphor-converted, warm",
    "LED_BH1": "LED BH1  -  hybrid phosphor + red",
    "LED_RGB1": "LED RGB1  -  RGB-mixed (narrow lines)",
    "LED_V1": "LED V1  -  violet-pumped phosphor",
    "LED_V2": "LED V2  -  violet-pumped phosphor",
}


def list_illuminants() -> list[Illuminant]:
    """All ``spectra/illuminant/interpolated/*.csv``."""
    d = spectra_dir() / "illuminant" / "interpolated"
    out: list[Illuminant] = []
    for path in sorted(d.glob("*.csv")):
        rid = path.stem
        rel = str(path.relative_to(spectra_dir().parent))
        out.append(Illuminant(id=rid, path=path, repo_relative=rel))
    return out


def illuminant_label(illum: Illuminant) -> str:
    return _ILLUMINANT_LABELS.get(illum.id, illum.id)


def load_spectrum_csv(path: Path) -> tuple[list[float], list[float]]:
    """Wavelength (nm), value  -  delegates to ``tools/apply_emva_noise.read_csv_curve``."""
    apply_emva_noise = import_tool("apply_emva_noise")
    wl, val = apply_emva_noise.read_csv_curve(Path(path))
    return list(wl), list(val)
