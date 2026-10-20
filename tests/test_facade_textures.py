"""Opt-in ambientCG facade textures on the urban street lots (--facade-textures)."""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import build_urban_scene as urban  # noqa: E402

BASE = ["--proxy-cars", "--allow-missing-assets", "--xres", "64", "--yres", "36", "--pixelsamples", "1"]
MANIFEST = yaml.safe_load((REPO / "config/highway_assets.yaml").read_text())["assets"]
PREPARED = REPO / "scenes/assets/highway/prepared"
HAVE = all((PREPARED / a / "prepared.json").is_file() for a in urban.FACADE_TEX)


def _build(tmp_path: Path, *extra: str) -> tuple[str, dict]:
    urban.main([*BASE, "--out-dir", str(tmp_path), *extra])
    return (tmp_path / "urban.pbrt").read_text(), json.loads((tmp_path / "urban_manifest.json").read_text())


def test_manifest_entries_are_cc0_ambientcg_with_pinned_maps():
    for aid in urban.FACADE_TEX:
        a = MANIFEST[aid]
        assert a["kind"] == "texture" and a["role"] == "facade"
        assert a["license"] == "CC0-1.0" and a["source"].startswith("https://ambientcg.com/view?id=")
        assert a["spd"] in ("facade_brick", "facade_render", "concrete")
        assert 0.5 < float(a["tile_size_m"]) < 3.0
        (f,) = a["files"]
        assert re.fullmatch(r"[0-9a-f]{64}", f["sha256"]) and f["url"].startswith("https://ambientcg.com/get?file=")
        assert set(a["maps"]) == {"color", "normal", "roughness"}
        assert all(m.startswith(f["extract"] + "/") for m in a["maps"].values())


def test_textured_box_uvs_are_metric_per_tile():
    p, t, uv = urban.textured_box(10.0, 5.0, 14.0, 9.6, 12.0, 2.0)
    assert p.shape == (20, 3) and t.shape == (10, 3) and uv.shape == (20, 2)
    assert np.isclose(p[:, 1].max(), 9.6) and np.isclose(p[:, 0].min(), 3.0) and np.isclose(p[:, 2].max(), 11.0)
    street = p[:4]  # x = x0 face: u runs along z, v up
    assert np.allclose(uv[:4, 0] * 2.0, street[:, 2]) and np.allclose(uv[:4, 1] * 2.0, street[:, 1])
    side = p[8:12]  # z = z0 face: u runs along x
    assert np.allclose(uv[8:12, 0] * 2.0, side[:, 0])


def test_default_and_explicit_none_are_identical(tmp_path):
    a, ma = _build(tmp_path / "a")
    b, mb = _build(tmp_path / "b", "--facade-textures", "none")
    assert a == b
    assert json.dumps(ma).replace(str(tmp_path / "a"), "OUT") == json.dumps(mb).replace(str(tmp_path / "b"), "OUT")
    assert "textures" not in ma["facades"] and "normalmap" not in a


def test_missing_maps_fall_back_to_procedural(tmp_path, monkeypatch):
    monkeypatch.setattr(urban, "FACADE_TEX", ("tex_does_not_exist",))
    a, _ = _build(tmp_path / "a")
    b, mb = _build(tmp_path / "b", "--facade-textures", "mixed")
    assert a == b and "textures" not in mb["facades"]


@pytest.mark.skipif(not HAVE, reason="facade textures not prepared (tools/fetch_highway_assets.py)")
def test_mixed_textures_every_lot_and_keeps_windows(tmp_path):
    a, ma = _build(tmp_path / "a")
    b, mb = _build(tmp_path / "b", "--facade-textures", "mixed")
    tex = mb["facades"]["textures"]
    assert len(tex) == mb["facades"]["lots"] and {t["texture"] for t in tex} <= set(urban.FACADE_TEX)
    assert len({t["texture"] for t in tex}) >= 4
    for aid in {t["texture"] for t in tex}:
        assert f'MakeNamedMaterial "{aid}" "string type" "diffuse"' in b and f'NamedMaterial "{aid}"' in b
    assert b.count('"string normalmap"') == len(urban.FACADE_TEX)
    for k in ("windows_lit", "windows_dark", "lots"):
        assert mb["facades"][k] == ma["facades"][k]
    assert mb["cars"] == ma["cars"] and mb["emitters"] == ma["emitters"]
    assert all(aid in mb["assets"] for aid in {t["texture"] for t in tex})
