"""Session checks shared by the whole suite."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pbrt_patchstamp import stamp_problem

REPO = Path(__file__).resolve().parents[1]


def pytest_sessionstart(session: pytest.Session) -> None:
    """Fail fast when the pbrt binary the tests will use was built from an older patch stack."""
    pbrt = Path(os.environ.get("OPENCAM_PBRT", REPO / "third_party" / "pbrt-v4" / "build" / "pbrt"))
    if not pbrt.is_file() or os.environ.get("OPENCAM_ALLOW_STALE_PBRT") == "1":
        return
    problem = stamp_problem(pbrt)
    if problem:
        raise pytest.UsageError(
            f"stale pbrt binary {pbrt}: {problem}.\n"
            "Rebuild it with tools/build_pbrt.sh (incremental), point OPENCAM_PBRT at a current build, "
            "or set OPENCAM_ALLOW_STALE_PBRT=1 to run anyway."
        )
