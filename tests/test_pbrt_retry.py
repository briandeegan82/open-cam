"""tools/pbrt_retry.py: a crash or hang is run again, a normal error is not."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from pbrt_retry import run_pbrt  # noqa: E402

# fake pbrt: counts its runs in ./n; on run k behaves as argv[k] (ok, abort, segv, hang, error)
FAKE = r"""
import os, signal, sys, time
n = int(open("n").read()) + 1 if os.path.exists("n") else 1
open("n", "w").write(str(n))
what = sys.argv[min(n, len(sys.argv) - 1)]
if what == "abort": os.kill(os.getpid(), signal.SIGABRT)
if what == "segv": os.kill(os.getpid(), signal.SIGSEGV)
if what == "hang": time.sleep(60)
if what == "error": sys.exit(1)
"""


def _fake(tmp_path: Path, *script: str) -> list[str]:
    (tmp_path / "fake.py").write_text(FAKE)
    return [sys.executable, "fake.py", *script]


def _runs(tmp_path: Path) -> int:
    return int((tmp_path / "n").read_text())


def test_crash_then_ok(tmp_path):
    run_pbrt(_fake(tmp_path, "abort", "segv", "ok"), cwd=tmp_path)
    assert _runs(tmp_path) == 3


def test_hang_then_ok(tmp_path):
    run_pbrt(_fake(tmp_path, "hang", "ok"), cwd=tmp_path, timeout=2)
    assert _runs(tmp_path) == 2


def test_error_not_retried(tmp_path):
    with pytest.raises(subprocess.CalledProcessError) as e:
        run_pbrt(_fake(tmp_path, "error", "ok"), cwd=tmp_path)
    assert e.value.returncode == 1 and _runs(tmp_path) == 1


def test_gives_up(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        run_pbrt(_fake(tmp_path, "segv"), cwd=tmp_path, attempts=2)
    assert _runs(tmp_path) == 2


def test_cli(tmp_path):
    cmd = [sys.executable, str(REPO / "tools" / "pbrt_retry.py"), "--cwd", str(tmp_path), "--"]
    assert subprocess.run(cmd + _fake(tmp_path, "abort", "ok")).returncode == 0
    (tmp_path / "n").unlink()
    assert subprocess.run(cmd + _fake(tmp_path, "error")).returncode == 1
