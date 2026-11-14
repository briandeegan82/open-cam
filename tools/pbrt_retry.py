"""Run pbrt and run it again if it crashes or hangs.

The patched pbrt crashes now and then during rendering on the night wet highway (double free, SIGSEGV, or a hang;
3-4 in 64 full renders, not reproduced under AddressSanitizer in 60; docs/HIGHWAY_SCENES.md). The render itself is
fine when it finishes, so a run killed by a signal or past ``timeout`` seconds is started again, up to ``attempts``
times. A normal error exit (bad scene, missing file) is not retried.

    from pbrt_retry import run_pbrt
    run_pbrt([pbrt, "--quiet", "--seed", "3", "scene.pbrt"], cwd=d, timeout=3600)

    venv/bin/python tools/pbrt_retry.py --timeout 3600 --cwd out/x -- third_party/pbrt-v4/build/pbrt --quiet x.pbrt
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

ATTEMPTS = int(os.environ.get("PBRT_ATTEMPTS", "3"))


def _crashed(rc: int) -> bool:
    """Killed by a signal: negative from subprocess, or 128 + signal from a shell or wrapper in between."""
    return rc < 0 or rc > 128


def run_pbrt(cmd: list[str], cwd: str | Path | None = None, *, timeout: float | None = None,
             attempts: int = ATTEMPTS, **kwargs) -> subprocess.CompletedProcess:
    """``subprocess.run(cmd, cwd=cwd, **kwargs)`` with up to ``attempts`` runs while pbrt crashes or hangs.

    Raises ``subprocess.CalledProcessError`` on a normal error exit or when the last attempt crashed,
    ``subprocess.TimeoutExpired`` when the last attempt hung.
    """
    cmd = [str(c) for c in cmd]
    attempts = max(attempts, 1)
    for attempt in range(1, attempts + 1):
        last = attempt == attempts
        try:
            p = subprocess.run(cmd, cwd=cwd, timeout=timeout, **kwargs)
        except subprocess.TimeoutExpired:
            if last:
                raise
            print(f"pbrt_retry: no exit after {timeout:g} s, attempt {attempt + 1}/{attempts}: {' '.join(cmd)}",
                  file=sys.stderr, flush=True)
            continue
        if p.returncode == 0:
            return p
        if not _crashed(p.returncode) or last:
            raise subprocess.CalledProcessError(p.returncode, cmd, p.stdout, p.stderr)
        sig = -p.returncode if p.returncode < 0 else p.returncode - 128
        name = signal.Signals(sig).name if sig in signal.Signals._value2member_map_ else f"signal {sig}"
        print(f"pbrt_retry: {name}, attempt {attempt + 1}/{attempts}: {' '.join(cmd)}", file=sys.stderr, flush=True)
    raise AssertionError("unreachable")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--attempts", type=int, default=ATTEMPTS, help="runs at most (default $PBRT_ATTEMPTS or 3)")
    ap.add_argument("--timeout", type=float, default=float(os.environ.get("PBRT_TIMEOUT_S", "0")) or None,
                    help="seconds before a run counts as hung (default $PBRT_TIMEOUT_S; none)")
    ap.add_argument("--cwd", type=Path, help="directory to run pbrt in")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- pbrt and its arguments")
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("no command")
    try:
        run_pbrt(cmd, cwd=a.cwd, timeout=a.timeout, attempts=a.attempts)
    except subprocess.CalledProcessError as e:
        return e.returncode if e.returncode > 0 else 128 - e.returncode
    except subprocess.TimeoutExpired:
        print(f"pbrt_retry: gave up, no exit after {a.timeout:g} s", file=sys.stderr)
        return 124
    return 0


if __name__ == "__main__":
    sys.exit(main())
