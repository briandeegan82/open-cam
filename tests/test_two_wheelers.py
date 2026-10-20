"""Parked two-wheelers (tools/two_wheelers.py; urban --two-wheelers parked)."""

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import build_urban_scene as urban  # noqa: E402
import two_wheelers as tw  # noqa: E402
from fetch_highway_assets import cache_root, load_asset_manifest  # noqa: E402

AM = load_asset_manifest(REPO / "config" / "highway_assets.yaml")
PREP = cache_root(REPO, AM) / "prepared"
needs_assets = pytest.mark.skipif(
    not all((PREP / a / "prepared.json").is_file() for a in tw.MODELS), reason="two-wheeler assets not prepared"
)
BASE = ["--proxy-cars", "--allow-missing-assets", "--xres", "64", "--yres", "36", "--pixelsamples", "1"]


def _build(tmp_path: Path, *extra: str) -> tuple[str, dict]:
    urban.main([*BASE, "--out-dir", str(tmp_path), *extra])
    return (tmp_path / "urban.pbrt").read_text(), json.loads((tmp_path / "urban_manifest.json").read_text())


@pytest.mark.parametrize("aid", sorted(tw.MODELS))
def test_manifest_provenance(aid):
    a = AM["assets"][aid]
    assert a["kind"] == "glb" and a["license"] == "CC-BY-4.0" and a["author"]
    uid = a["glb"][:-4]
    assert a["source"] == f"https://sketchfab.com/3d-models/{uid}" and uid in a["license_note"]
    (f,) = a["files"]
    assert re.fullmatch(r"[0-9a-f]{64}", f["sha256"]) and f["url"].endswith(f"/{uid}.glb")


@pytest.mark.parametrize("front", sorted(tw.FRONT_YAW))
def test_front_yaw_turns_model_front_to_plus_z(front):
    a = math.radians(tw.FRONT_YAW[front])
    v = {"x": np.array([1.0, 0, 0]), "z": np.array([0, 0, 1.0])}[front[1]] * (1 if front[0] == "+" else -1)
    rot = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])  # pbrt Rotate a 0 1 0
    assert np.allclose(rot @ v, [0, 0, 1], atol=1e-12)


def test_default_has_no_two_wheelers(tmp_path):
    a, ma = _build(tmp_path / "a")
    b, _ = _build(tmp_path / "b", "--two-wheelers", "none")
    assert a == b and "two_wheelers" not in ma and "two-wheelers" not in a


@needs_assets
def test_parked_two_wheelers(tmp_path):
    _, a = _build(tmp_path / "a")
    scene, b = _build(tmp_path / "b", "--two-wheelers", "parked")
    assert [(c["id"], c["model"], c["paint"], c["z"]) for c in a["cars"]] == [
        (c["id"], c["model"], c["paint"], c["z"]) for c in b["cars"]
    ]
    items = b["two_wheelers"]["items"]
    kinds = {m["kind"] for m in items}
    assert {"bicycle", "kickscooter"} <= kinds and kinds & {"motorcycle", "moped", "scooter"}
    for m in items:
        lo, hi = np.array(m["box_world"])
        ax = np.abs([lo[0], hi[0]])
        if m["role"] == "pavement":
            assert m["position"][1] == urban.KERB_H
            assert ax.min() > urban.HALF - 0.05 and ax.max() < urban.BUILD_X - 0.5
        else:
            assert m["position"][1] == 0.0
            assert ax.min() > urban.HALF - urban.PARK_W - 0.05 and ax.max() < urban.HALF + 0.05
        assert 0.5 < hi[1] - lo[1] < 2.0
        if m["heading_deg"] % 90 == 0:  # length lies along the heading
            along = (hi - lo)[2 if m["heading_deg"] % 180 == 0 else 0]
            assert along == pytest.approx(m["length_m"], rel=0.03)
    assert scene.count("ObjectInstance") == len(items)


@needs_assets
def test_two_wheelers_avoid_street_furniture(tmp_path):
    _, m = _build(tmp_path, "--two-wheelers", "parked", "--street-furniture", "pavement")
    boxes = [np.array(f["box_world"]) for f in m["street_furniture"]["items"]]
    for t in m["two_wheelers"]["items"]:
        lo, hi = np.array(t["box_world"])
        for b in boxes:
            assert np.any(hi < b[0]) or np.any(lo > b[1]), (t["asset"], b)
