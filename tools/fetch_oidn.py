"""Download Intel Open Image Denoise (official Linux x86-64 release) into ``third_party/oidn``.

``tools/denoise_spectral_exr.py`` (opt-in ``render.denoise`` stage) calls the ``oidnDenoise`` command-line tool.
The archive is pinned by version and SHA-256. OIDN is Apache-2.0 licensed (https://www.openimagedenoise.org).
Other platforms: install OIDN yourself and point ``$OPENCAM_OIDN`` (or ``--oidn``) at ``oidnDenoise``.

    venv/bin/python tools/fetch_oidn.py
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VERSION = "2.3.3"
URL = f"https://github.com/RenderKit/oidn/releases/download/v{VERSION}/oidn-{VERSION}.x86_64.linux.tar.gz"
SHA256 = "3c385230d9e6f63527ba72f2229594dbac5051674219d72e0044b5d0b841796f"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dest", type=Path, default=REPO / "third_party" / "oidn")
    ap.add_argument("--force", action="store_true", help="re-download even if already installed")
    a = ap.parse_args(argv)
    binary = a.dest / "bin" / "oidnDenoise"
    if binary.is_file() and not a.force:
        print(f"already installed: {binary}")
        return 0
    if sys.platform != "linux" or platform.machine() not in ("x86_64", "AMD64"):
        print("fetch_oidn.py only handles Linux x86-64; install OIDN manually and set $OPENCAM_OIDN.", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as td:
        archive = Path(td) / "oidn.tar.gz"
        print(f"downloading {URL}")
        with urllib.request.urlopen(URL, timeout=120) as r, open(archive, "wb") as f:
            shutil.copyfileobj(r, f)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != SHA256:
            print(f"SHA-256 mismatch: got {digest}, expected {SHA256}", file=sys.stderr)
            return 1
        with tarfile.open(archive) as tf:
            if sys.version_info >= (3, 12):
                tf.extractall(td, filter="data")
            else:
                tf.extractall(td)  # noqa: S202 - pinned, hash-checked official archive
        root = Path(td) / f"oidn-{VERSION}.x86_64.linux"
        if a.dest.exists():
            shutil.rmtree(a.dest)
        a.dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(root), str(a.dest))
    print(f"installed {binary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
