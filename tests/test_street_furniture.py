"""Street furniture and imported signal heads (tools/street_furniture.py; urban --signal-model, --street-furniture, --work-zone)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import build_urban_scene as urban  # noqa: E402
import street_furniture as sf  # noqa: E402
from fetch_highway_assets import cache_root, load_asset_manifest, read_ply_positions  # noqa: E402

MANIFEST = load_asset_manifest(REPO / "config" / "highway_assets.yaml")["assets"]
ASSETS = [*sf.PROPS, *(m[0] for m in sf.SIGNAL_MODELS.values())]
PREP = (
    cache_root(REPO, {"assets": MANIFEST, **load_asset_manifest(REPO / "config" / "highway_assets.yaml")}) / "prepared"
)
needs_assets = pytest.mark.skipif(
    not all((PREP / a / "prepared.json").is_file() for a in ASSETS), reason="street-furniture assets not prepared"
)
BASE = ["--proxy-cars", "--allow-missing-assets", "--xres", "64", "--yres", "36", "--pixelsamples", "1"]


def _build(tmp_path: Path, *extra: str) -> tuple[str, dict]:
    urban.main([*BASE, "--out-dir", str(tmp_path), *extra])
    return (tmp_path / "urban.pbrt").read_text(), json.loads((tmp_path / "urban_manifest.json").read_text())


@pytest.mark.parametrize("aid", ASSETS)
def test_manifest_provenance(aid):
    e = MANIFEST[aid]
    assert e["kind"] in ("gltf", "glb")
    assert e["license"] in ("CC0-1.0", "CC-BY-4.0")
    if e["license"] == "CC-BY-4.0":
        assert e["source"].startswith("https://sketchfab.com/3d-models/")
        assert "Attribution required" in e["license_note"]
        assert "/resolve/556637099bf4fa79ea7b239d0d9c328b8a2e9ac8/" in e["files"][0]["url"]
    else:
        assert e["source"].startswith("https://polyhaven.com/a/")
    assert all(len(f["sha256"]) == 64 for f in e["files"])


@pytest.mark.parametrize("face,expect", [("-z", (0, -1)), ("-x", (-1, 0)), ("+x", (1, 0))])
def test_face_yaw_turns_model_front(face, expect):
    a = np.radians(sf.FACE_YAW[face])  # pbrt Rotate about +y maps +z to (sin a, 0, cos a)
    assert (np.sin(a), np.cos(a)) == pytest.approx(expect, abs=1e-12)


def test_world_box_rotation():
    v = {"lo": np.array([-1.0, 0.0, -0.25]), "hi": np.array([1.0, 1.0, 0.25])}
    lo, hi = sf.world_box(v, (10.0, 0.0, 5.0), 90.0, 1.0, np.zeros(3))
    assert lo == pytest.approx([9.75, 0.0, 4.0]) and hi == pytest.approx([10.25, 1.0, 6.0])


def test_work_zone_taper():
    pts = sf.work_zone_layout(5.3, 3.75, 8.0, 20.0)
    assert pts[0] == pytest.approx((5.3, 8.0)) and pts[-1] == pytest.approx((3.75, 20.0))
    assert all(3.75 - 1e-9 <= x <= 5.3 + 1e-9 for x, _ in pts)


def test_default_has_no_furniture(tmp_path):
    scene, m = _build(tmp_path)
    assert "ObjectInstance" not in scene and "street furniture" not in scene
    assert "street_furniture" not in m and m["signal_model"] == "procedural"


def test_pavement_furniture_keeps_parked_cars(tmp_path):
    _, a = _build(tmp_path / "a")
    _, b = _build(tmp_path / "b", "--street-furniture", "pavement")
    assert [(c["id"], c["z"]) for c in a["cars"]] == [(c["id"], c["z"]) for c in b["cars"]]


def test_work_zone_closes_right_parking(tmp_path):
    _, m = _build(tmp_path, "--work-zone", "cones")
    assert m["street_furniture"]["work_zone_z_m"] == [8.0, 20.0]
    assert not [c for c in m["cars"] if c["lane"] == 9 and c["x"] > 0 and 5.0 < c["z"] < 23.0]
    _, d = _build(tmp_path / "d")
    assert [c for c in d["cars"] if c["lane"] == 9 and c["x"] > 0 and 5.0 < c["z"] < 23.0]

    def outside(cars):
        return [
            (c["id"], c["model"], c["paint"], c["x"], c["z"]) for c in cars if not (c["x"] > 0 and 5.0 < c["z"] < 23.0)
        ]

    assert outside(m["cars"]) == outside(d["cars"])


@needs_assets
def test_row_assets_split_into_variants():
    for aid in ("prop_fire_hydrant", "prop_metal_trash_can"):
        info = json.loads((PREP / aid / "prepared.json").read_text())
        vs = sf.variants(aid, info, PREP / aid)
        assert len(vs) == 2 and all(v["hi"][0] - v["lo"][0] < 0.8 for v in vs)


@needs_assets
@pytest.mark.parametrize("model", sf.SIGNAL_MODELS)
def test_signal_lens_constants_match_mesh(model):
    aid, lx, ly, pitch, lz = sf.SIGNAL_MODELS[model]
    info = json.loads((PREP / aid / "prepared.json").read_text())
    p = np.concatenate([read_ply_positions(PREP / aid / s["ply"]) for s in info["shapes"]])
    for k in (-1, 0, 1):
        near = (abs(p[:, 0] - lx) < 0.03) & (abs(p[:, 1] - (ly + k * pitch)) < 0.03)
        assert p[near, 2].max() == pytest.approx(lz, abs=0.01)  # lens face, in front of nothing else


@needs_assets
def test_signal_model_keeps_emitters(tmp_path):
    _, a = _build(tmp_path / "a")
    scene, b = _build(tmp_path / "b", "--signal-model", "nyc80s")
    assert scene.count('ObjectInstance "signal_head_nyc80s"') == 5 and "signal_nyc80s" in b["assets"]

    def sig(m):
        return sorted(
            (e["aspect"], tuple(e["centre_world"]), e["intensity_cd_nominal"])
            for e in m["emitters"]
            if e["kind"] == "traffic_signal"
        )

    assert sig(a) == sig(b)
    for e in b["emitters"]:
        if e["kind"] == "traffic_signal":
            assert 'Shape "disk"' in (tmp_path / "b" / e["states"]["on"]).read_text()


@needs_assets
def test_furniture_placement(tmp_path):
    scene, m = _build(tmp_path, "--street-furniture", "pavement", "--work-zone", "drums")
    items = m["street_furniture"]["items"]
    pav = [i for i in items if i["role"] in ("kerb", "facade")]
    assert len(pav) > 6
    for i in pav:
        lo, hi = i["box_world"]
        xs = sorted((abs(lo[0]), abs(hi[0])))
        assert xs[0] > urban.HALF - 0.05 and xs[1] < urban.BUILD_X + 0.05
        assert lo[1] == pytest.approx(urban.KERB_H, abs=0.02)
    zone = [i for i in items if i["role"] in ("drum", "barrier")]
    assert {i["role"] for i in zone} == {"drum", "barrier"}
    for i in zone:
        lo, hi = i["box_world"]
        # AABB of a randomly rotated drum base, so allow a little past the lane line
        assert lo[0] > urban.HALF - urban.PARK_W - 0.3 and hi[0] < urban.HALF + 0.05
        assert lo[1] == pytest.approx(0.0, abs=0.02) and lo[2] > 7.0 and hi[2] < 21.0
    for v in {i["variant"] for i in items}:
        assert f'ObjectBegin "{v}"' in scene
