"""Repo-wide locations: repo root, pbrt binary ($OPENCAM_PBRT), output root ($OPENCAM_OUT), pbrt output dirs."""

from __future__ import annotations

import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

_FILM_FILENAME = re.compile(r'"string\s+filename"\s*\[?\s*"([^"]+)"')


def pbrt_path() -> Path:
    """pbrt binary: ``$OPENCAM_PBRT`` if set, else the submodule build."""
    env = os.environ.get("OPENCAM_PBRT")
    return Path(env) if env else REPO / "third_party" / "pbrt-v4" / "build" / "pbrt"


def out_dir(*parts: str) -> Path:
    """``$OPENCAM_OUT/<parts>`` (default ``<repo>/out/<parts>``); not created."""
    return Path(os.environ.get("OPENCAM_OUT") or REPO / "out").joinpath(*parts)


def ensure_pbrt_output_dir(scene: Path, cwd: Path | None = None, outfile: str | Path | None = None) -> Path | None:
    """Create the directory pbrt will write the film to (pbrt silently writes nothing if it is missing).

    ``outfile`` (pbrt ``--outfile``) wins over the scene's ``Film "string filename"``; relative paths are
    resolved against ``cwd`` (default: the scene's directory). Returns the created directory, or None.
    """
    scene = Path(scene)
    base = Path(cwd) if cwd is not None else scene.parent
    name = outfile
    if name is None:
        if not scene.is_absolute():
            scene = base / scene
        m = _FILM_FILENAME.search(scene.read_text(errors="replace")) if scene.is_file() else None
        name = m.group(1) if m else None
    if name is None:
        return None
    d = (base / Path(name)).parent
    d.mkdir(parents=True, exist_ok=True)
    return d
