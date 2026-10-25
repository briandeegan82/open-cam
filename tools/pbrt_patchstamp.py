"""Patch-stack stamp for the pbrt binary built by tools/build_pbrt.sh.

The stamp is sha256 over every third_party/patches/*.patch in byte order of the file name
(name, newline, contents). build_pbrt.sh writes it next to the binary as `pbrt.patchstamp`, so a
binary built from an older patch stack can be detected instead of failing tests in confusing ways.
Usage: python tools/pbrt_patchstamp.py [--check PBRT_BIN]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PATCH_DIR = REPO / "third_party" / "patches"
STAMP_NAME = "pbrt.patchstamp"


def patch_stamp(patch_dir: Path = PATCH_DIR) -> str:
    h = hashlib.sha256()
    for p in sorted(patch_dir.glob("*.patch"), key=lambda q: q.name.encode()):
        h.update(p.name.encode() + b"\n")
        h.update(p.read_bytes())
    return h.hexdigest()


def stamp_problem(pbrt_bin: Path, patch_dir: Path = PATCH_DIR) -> str | None:
    """None when pbrt_bin has a stamp matching the current patch stack, else a human-readable reason."""
    stamp = Path(pbrt_bin).parent / STAMP_NAME
    if not stamp.is_file():
        return f"{stamp} is missing (binary predates patch stamping or was not built by tools/build_pbrt.sh)"
    if stamp.read_text().strip() != patch_stamp(patch_dir):
        return f"{stamp} does not match third_party/patches (binary built from an older patch stack)"
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", type=Path, help="pbrt binary to check against the current patch stack")
    args = ap.parse_args(argv)
    if args.check is None:
        print(patch_stamp())
        return 0
    problem = stamp_problem(args.check)
    if problem:
        print(f"stale pbrt: {problem}; rebuild with tools/build_pbrt.sh", file=sys.stderr)
        return 1
    print(f"ok: {args.check} matches the patch stack")
    return 0


if __name__ == "__main__":
    sys.exit(main())
