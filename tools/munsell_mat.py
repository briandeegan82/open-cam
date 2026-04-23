"""Load Joensuu matte Munsell MATLAB spectra (``spectra/munsell/README.txt``)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class JoensuuMunsellBundle:
    """421×N reflectance (380–800 nm, 1 nm), N chip labels, optional 16×N CIE/D65 table ``C``."""

    wavelength_nm: np.ndarray  # (421,) float64
    reflectance: np.ndarray  # (421, N) float64
    labels: list[str]
    C_D65: np.ndarray | None  # (16, N) float64 or None


def parse_munsell_label(label: str) -> dict[str, Any]:
    """Split ``'2.5R 9/2'`` / ``'5 YR 7 / 1'`` style strings into hue, value, chroma (or nulls)."""
    s = str(label).strip()
    m = re.match(r"^(.+?)\s+([\d.]+)\s*/\s*([\d.]+)\s*$", s)
    if not m:
        return {"hue": None, "value": None, "chroma": None, "label": s}
    hue, v, c = m.group(1).strip(), float(m.group(2)), float(m.group(3))
    return {"hue": hue, "value": v, "chroma": c, "label": s}


def sanitize_filename(label: str) -> str:
    s = str(label).strip()
    s = re.sub(r"\s+", "_", s)
    s = s.replace("/", "_")
    s = re.sub(r"[^0-9A-Za-z._+-]+", "", s)
    return s or "unknown"


def load_joensuu_mat(mat_path: Path) -> JoensuuMunsellBundle:
    try:
        import scipy.io
    except ImportError as exc:
        raise RuntimeError("scipy is required: pip install scipy") from exc

    mat_path = mat_path.resolve()
    if not mat_path.is_file():
        raise FileNotFoundError(mat_path)

    data = scipy.io.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    for key in ("munsell", "S"):
        if key not in data:
            raise KeyError(f"MAT file missing {key!r}; keys: {sorted(k for k in data if not str(k).startswith('__'))!r}")

    spec = np.asarray(data["munsell"], dtype=np.float64)
    if spec.ndim != 2 or spec.shape[0] != 421:
        raise ValueError(f"expected munsell shape (421, N), got {spec.shape}")

    labels = np.asarray(data["S"], dtype=object).ravel()
    if labels.size != spec.shape[1]:
        raise ValueError(f"S length {labels.size} != columns {spec.shape[1]}")

    wl = np.arange(380.0, 801.0, 1.0, dtype=np.float64)
    if wl.size != spec.shape[0]:
        raise RuntimeError(f"wavelength grid {wl.size} != spectrum rows {spec.shape[0]}")

    C = None
    if "C" in data:
        C = np.asarray(data["C"], dtype=np.float64)
        if C.shape[1] != spec.shape[1]:
            raise ValueError(f"C shape {C.shape} vs spectra columns {spec.shape[1]}")

    str_labels = [str(labels[i]).strip() for i in range(labels.size)]
    return JoensuuMunsellBundle(wavelength_nm=wl, reflectance=spec, labels=str_labels, C_D65=C)
