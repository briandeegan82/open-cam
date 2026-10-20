"""--vegetation-pool: extended Poly Haven species, manifest provenance, classic draws unchanged."""

import argparse
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import highway_variety as hv  # noqa: E402

MANIFEST = yaml.safe_load((ROOT / "config" / "highway_assets.yaml").read_text())["assets"]


def _args(**kw) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    hv.add_arguments(ap)
    a = ap.parse_args([])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


@pytest.mark.parametrize("vid", sorted(hv.SPECIES_EXTENDED))
def test_extended_species_manifest(vid):
    e = MANIFEST[vid]
    name = vid.removeprefix("veg_")
    assert e["kind"] == "gltf" and e["license"] == "CC0-1.0"
    assert e["source"] == f"https://polyhaven.com/a/{name}"
    assert e["gltf"] == f"{name}/{name}_1k.gltf"
    paths = [f["path"] for f in e["files"]]
    assert e["gltf"] in paths and any(p.endswith(".bin") for p in paths)
    for f in e["files"]:
        assert f["url"].startswith("https://dl.polyhaven.org/file/ph-assets/Models/")
        assert len(f["sha256"]) == 64 and int(f["sha256"], 16) >= 0
    kind, mult = hv.SPECIES_EXTENDED[vid]
    assert kind in ("tree", "shrub") and 0.1 < mult < 3.0
    assert vid not in hv.SPECIES


def test_classic_pool_unchanged():
    a = _args(seed=11)
    classic = hv.resolve(a, ("car_bmw_m6",), 1)
    assert all(v in hv.SPECIES for v, _ in classic.species)
    assert hv.resolve(_args(seed=None), ("car_bmw_m6",), 1).species == hv.DEFAULT_SPECIES


def test_extended_pool_draws_new_species():
    seen = set()
    for seed in range(40):
        seen |= {v for v, _ in hv.resolve(_args(seed=seed, vegetation_pool="extended"), ("car_bmw_m6",), 1).species}
    assert seen & set(hv.SPECIES_EXTENDED)
    v = hv.resolve(_args(seed=None, vegetation_pool="extended"), ("car_bmw_m6",), 1)
    assert v.species == hv.DEFAULT_SPECIES_EXTENDED
    for vid, _ in v.species:
        assert v.species_scale(vid) > 0
