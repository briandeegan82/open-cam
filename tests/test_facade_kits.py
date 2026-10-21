"""Opt-in street fronts assembled from the Poly Haven modular facade kits (--facade-style kits)."""

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
import facade_kits as fk  # noqa: E402

BASE = ["--proxy-cars", "--allow-missing-assets", "--xres", "64", "--yres", "36", "--pixelsamples", "1"]
MANIFEST = yaml.safe_load((REPO / "config/highway_assets.yaml").read_text())["assets"]
PREPARED = REPO / "scenes/assets/highway/prepared"
HAVE = all((PREPARED / a / "prepared.json").is_file() for a in fk.KITS)


def _build(tmp_path: Path, *extra: str) -> tuple[str, dict]:
    urban.main([*BASE, "--out-dir", str(tmp_path), *extra])
    return (tmp_path / "urban.pbrt").read_text(), json.loads((tmp_path / "urban_manifest.json").read_text())


def _same_manifest(ma, mb, a, b):
    return json.dumps(ma).replace(str(a), "OUT") == json.dumps(mb).replace(str(b), "OUT")


def test_manifest_entries_are_poly_haven_cc0_with_pinned_files():
    for aid in fk.KITS:
        a = MANIFEST[aid]
        assert a["kind"] == "gltf" and a["license"] == "CC0-1.0"
        assert a["source"].startswith("https://polyhaven.com/a/modular_")
        assert a["gltf"].endswith("_2k.gltf")
        paths = [f["path"] for f in a["files"]]
        assert a["gltf"] in paths and any(p.endswith(".bin") for p in paths)
        for f in a["files"]:
            assert f["url"].startswith("https://dl.polyhaven.org/") and re.fullmatch(r"[0-9a-f]{64}", f["sha256"])


def test_frame_maps_bay_onto_both_street_faces():
    # right side (s=+1): local +z (out of the wall) -> world -x, local x in [-3, 0] -> world z in [zb, zb+wb]
    yaw, za = fk.frame(1, 9.0, 10.0, 3.0)
    assert yaw == -90.0 and za == 13.0
    yaw, za = fk.frame(-1, -9.0, 10.0, 3.0)
    assert yaw == 90.0 and za == 10.0
    part = {"lo": np.array([-2.5, 0.3, 0.0]), "hi": np.array([-0.5, 2.3, 0.1])}
    for s, (_yaw, za) in ((1, fk.frame(1, 9.0, 10.0, 3.0)), (-1, fk.frame(-1, -9.0, 10.0, 3.0))):
        q = fk.room_quad(part, s, s * 9.0, 3.0, za, 1.0)
        assert np.allclose(q[:, 0], s * (9.0 + fk.INTERIOR))
        assert np.isclose(q[:, 2].min(), 10.5) and np.isclose(q[:, 2].max(), 12.5)
        assert np.isclose(q[:, 1].min(), 3.3) and np.isclose(q[:, 1].max(), 5.3)


def test_default_and_explicit_procedural_are_identical(tmp_path):
    a, ma = _build(tmp_path / "a")
    b, mb = _build(tmp_path / "b", "--facade-style", "procedural")
    assert a == b and _same_manifest(ma, mb, tmp_path / "a", tmp_path / "b")
    assert "kits" not in ma["facades"] and "ObjectInstance" not in a.split("# ---- facades")[1].split("# ----")[0]


def test_missing_kits_fall_back_to_procedural(tmp_path, monkeypatch):
    monkeypatch.setattr(fk, "KITS", ("kit_does_not_exist",))
    a, _ = _build(tmp_path / "a")
    b, mb = _build(tmp_path / "b", "--facade-style", "kits")
    assert a == b and "kits" not in mb["facades"]


@pytest.mark.skipif(not HAVE, reason="facade kits not prepared (tools/fetch_highway_assets.py)")
def test_kit_parts_pair_walls_with_inserts():
    for aid in fk.KITS:
        info = json.loads((PREPARED / aid / "prepared.json").read_text())
        P = fk.parts(info, PREPARED / aid)
        win, door = fk.pairs(P, "wall_window_", "window_"), fk.pairs(P, "wall_door_", "door_")
        assert len(win) >= 4 and len(door) >= 2
        for wall, ins in win + door:
            assert np.isclose(P[wall]["hi"][0] - P[wall]["lo"][0], fk.BAY, atol=0.05)
            assert P[wall]["lo"][0] - 0.05 <= P[ins]["lo"][0] and P[ins]["hi"][0] <= P[wall]["hi"][0] + 0.05
            assert P[ins]["hi"][1] <= fk.STOREY + 0.05
        assert fk.trim(P, "cornice") and fk.trim(P, "crown")


@pytest.mark.skipif(not HAVE, reason="facade kits not prepared (tools/fetch_highway_assets.py)")
def test_kits_build_keeps_cars_and_emitters_and_lights_rooms_at_night(tmp_path):
    a, ma = _build(tmp_path / "a")
    b, mb = _build(tmp_path / "b", "--facade-style", "kits")
    kits = mb["facades"]["kits"]
    main = [k for k in kits if k.get("street") != "cross"]
    assert len(main) == mb["facades"]["lots"] and {k["kit"] for k in kits} == set(fk.KITS)
    assert len(kits) - len(main) == len(urban.cross_street_lots(0))  # default --kit-cross-street on
    assert all(abs(k["bay_m"] - fk.BAY) < 0.75 for k in kits)
    assert mb["cars"] == ma["cars"] and mb["emitters"] == ma["emitters"]
    assert mb["facades"]["windows_lit"] > 0 and mb["facades"]["windows_dark"] > 0
    assert all(aid in mb["assets"] for aid in fk.KITS)
    assert b.count("ObjectInstance") > 500 and '"thindielectric"' in b
    for name in re.findall(r'ObjectInstance "([^"]+)"', b):
        assert f'ObjectBegin "{name}"' in b


@pytest.mark.skipif(not HAVE, reason="facade kits not prepared (tools/fetch_highway_assets.py)")
def test_kit_recess_is_not_covered_by_lot_or_back_block_boxes(tmp_path):
    b, _ = _build(tmp_path, "--facade-style", "kits")
    seen = 0
    for seg in re.split(r'\nNamedMaterial "', b)[1:]:
        if not seg.startswith(("facade_render", "facade_brick", "concrete")):
            continue
        for arr in re.findall(r'"point3 P" \[([^\]]*)\]', seg):
            p = np.array(arr.split(), float).reshape(-1, 3)
            ax = np.abs(p[p[:, 1] > 0.5, 0])
            seen += ax.size
            assert not np.any((ax > urban.BUILD_X - 0.01) & (ax < urban.BUILD_X + fk.RECESS - 0.01))
    assert seen > 0


def test_cross_street_lots_face_the_cross_street():
    import math

    for lot in urban.cross_street_lots(7):
        sx, sz = lot["corner"]
        yaw, tz = lot["xform"]
        a = math.radians(yaw)
        for zl in (lot["z0"], lot["z1"]):
            xl = lot["side"] * urban.BUILD_X  # wall plane in the main-street frame
            wx, wz = xl * math.cos(a) + zl * math.sin(a), -xl * math.sin(a) + zl * math.cos(a) + tz
            assert wz == pytest.approx(urban.Z_INT + sz * urban.BUILD_X)
            assert sx * wx >= urban.BUILD_X + urban.CROSS_KIT_Z[0] - 1e-9


def test_lot_height_matches_procedural_facades():
    assert fk.lot_height(4, None) == pytest.approx(12.0)
    assert fk.lot_height(4, fk.MATCH_STOREY) == pytest.approx(0.8 + 3.2 * 4)  # facade_lines' h


@pytest.mark.skipif(not HAVE, reason="facade kits not prepared (tools/fetch_highway_assets.py)")
def test_kit_storeys_and_metals_options(tmp_path):
    new, _ = _build(tmp_path / "n", "--facade-style", "kits")
    old, _ = _build(
        tmp_path / "o",
        "--facade-style",
        "kits",
        "--kit-storeys",
        "kit",
        "--kit-cross-street",
        "off",
        "--kit-metals",
        "diffuse",
    )
    assert re.search(r"Scale [0-9.]+ 1\.0666[67] 1", new) and not re.search(r"Scale [0-9.]+ 1\.0666", old)
    assert '"string type" "conductor"' in new and '"string type" "mix"' in new and "conductor" not in old
    assert new.count("ObjectInstance") > old.count("ObjectInstance")  # cross-street fronts
