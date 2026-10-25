import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import derived_cache  # noqa: E402


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENCAM_DERIVED_CACHE", str(tmp_path / "cache"))
    return tmp_path


def _gen(x):
    return x + 1


def test_key_tracks_code_params_and_source_bytes(cache):
    src = cache / "src.bin"
    src.write_bytes(b"a")
    k = derived_cache.key("t", [_gen], [src], n=1)
    assert derived_cache.key("t", [_gen], [src], n=1) == k
    assert derived_cache.key("t", [_gen], [src], n=2) != k
    assert derived_cache.key("t", [_gen, test_key_tracks_code_params_and_source_bytes], [src], n=1) != k
    src.write_bytes(b"b")
    assert derived_cache.key("t", [_gen], [src], n=1) != k


def test_materialize_shares_one_file_and_replaces_stale_outputs(cache):
    calls = []

    def make(ps):
        calls.append(ps)
        ps[0].write_text("fresh")

    a, b = cache / "outA" / "t.txt", cache / "outB" / "t.txt"
    b.parent.mkdir()
    b.write_text("stale")
    derived_cache.materialize("k1", [a], make)
    derived_cache.materialize("k1", [b], make)
    derived_cache.materialize("k1", [b], make)
    assert len(calls) == 1
    assert a.read_text() == b.read_text() == "fresh"
    assert os.path.samefile(a, b)
