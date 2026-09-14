"""Locate the Open Cam repo root and expose its ``tools/`` modules for import.

``tools/*.py`` use bare imports (``from camera_model import load_camera_model``)
that assume ``tools/`` itself is on ``sys.path`` — this mirrors how the CLI
scripts are normally invoked (``cwd=tools`` or ``PYTHONPATH=tools``). We do the
same thing here so the GUI can import the pipeline's real functions instead of
re-implementing them.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from importlib import import_module
from pathlib import Path


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Return the Open Cam repo root.

    Override with ``OPENCAM_REPO_ROOT`` (useful for tests); otherwise derived
    from this file's location: ``<repo>/gui/opencam_gui/core/repo.py``.
    """
    env = os.environ.get("OPENCAM_REPO_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def tools_dir() -> Path:
    return repo_root() / "tools"


def ensure_tools_on_path() -> None:
    """Put ``<repo>/tools`` on ``sys.path`` (idempotent) so ``import camera_model`` etc. work."""
    td = str(tools_dir())
    if td not in sys.path:
        sys.path.insert(0, td)


def import_tool(module_name: str):
    """Import a module from ``tools/`` (e.g. ``"camera_model"``, ``"emva_theory"``)."""
    ensure_tools_on_path()
    return import_module(module_name)


def config_dir() -> Path:
    return repo_root() / "config"


def spectra_dir() -> Path:
    return repo_root() / "spectra"


def pbrt_binary() -> Path:
    return repo_root() / "third_party" / "pbrt-v4" / "build" / "pbrt"


def pbrt_available() -> bool:
    return pbrt_binary().is_file()


def python_executable() -> str:
    """Prefer the repo's own venv, matching ``tools/run_pipeline.py``'s own convention."""
    venv_py = repo_root() / "venv" / "bin" / "python"
    if venv_py.is_file():
        return str(venv_py)
    return sys.executable
