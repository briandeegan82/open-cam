"""Content-keyed cache of derived textures, shared by every scene out-dir.

A derived file (detiled road maps, luminance maps, normal-map PNGs, wear maps) is stored once under
``cache_root()/<key>/`` and hard-linked (or copied) into each out-dir. The key hashes the generating
code (``inspect.getsource`` of the listed functions), the parameters and the bytes of the source
assets, so a code, parameter or asset change gets a new entry instead of silently reusing a stale map.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path

VERSION = 1
REPO = Path(__file__).resolve().parents[1]
_FILE_HASH: dict[tuple, str] = {}


def cache_root() -> Path:
    """``$OPENCAM_DERIVED_CACHE``, else ``scenes/assets/derived`` (git-ignored, next to the asset cache)."""
    env = os.environ.get("OPENCAM_DERIVED_CACHE")
    return Path(env) if env else REPO / "scenes" / "assets" / "derived"


def file_hash(p: Path) -> str:
    st = Path(p).stat()
    k = (str(Path(p).resolve()), st.st_size, st.st_mtime_ns)
    if k not in _FILE_HASH:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        _FILE_HASH[k] = h.hexdigest()
    return _FILE_HASH[k]


def key(kind: str, code: Iterable[Callable], sources: Iterable[Path] = (), **params) -> str:
    """Hex key of ``kind``: source text of ``code``, sha256 of each file in ``sources``, ``params``."""
    blob = json.dumps(
        [VERSION, kind, [inspect.getsource(f) for f in code], [file_hash(s) for s in sources], params],
        sort_keys=True,
        default=str,
    )
    return f"{kind}-{hashlib.sha256(blob.encode()).hexdigest()[:20]}"


def _place(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.exists() and os.path.samefile(src, dst):
            return
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def entry(k: str, names: list[str], make: Callable[[list[Path]], None]) -> Path:
    """Cache directory of ``k`` holding ``names``; on a miss ``make(paths)`` writes them first.

    The entry is written to a temporary directory and renamed into place, so concurrent builds (xdist
    workers) never see a half-written entry."""
    entry = cache_root() / k
    if not all((entry / n).is_file() for n in names):
        entry.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=f".{k}.", dir=entry.parent))
        try:
            make([tmp / n for n in names])
            try:
                os.rename(tmp, entry)
            except OSError:  # another process won the race (or a partial entry exists): replace per file
                entry.mkdir(exist_ok=True)
                for n in names:
                    os.replace(tmp / n, entry / n)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return entry


def materialize(k: str, dsts: list[Path], make: Callable[[list[Path]], None]) -> list[Path]:
    """Files ``dsts`` linked from cache entry ``k`` (``make`` writes them on a miss, same names)."""
    e = entry(k, [d.name for d in dsts], make)
    for d in dsts:
        _place(e / d.name, d)
    return dsts
