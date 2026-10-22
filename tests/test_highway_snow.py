"""Winter / snow options of the highway builder (tools/highway_snow.py)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import build_highway_scene as hw  # noqa: E402
import highway_snow as snow  # noqa: E402

BASE = ["--proxy-cars", "--allow-missing-assets", "--xres", "64", "--yres", "36", "--pixelsamples", "1"]


def _build(tmp_path: Path, *extra: str) -> tuple[str, dict]:
    hw.main([*BASE, "--out-dir", str(tmp_path), *extra])
    return (tmp_path / "highway.pbrt").read_text(), json.loads((tmp_path / "highway_manifest.json").read_text())


def test_albedo_physics():
    wl = np.array([500.0, 1030.0])
    fine, coarse = snow.snow_albedo(wl, 0.1), snow.snow_albedo(wl, 1.0)
    assert fine[0] > 0.98 and 0.75 < fine[1] < 0.9  # Wiscombe & Warren 1980: ~0.8 at 1.03 um for r = 50 um
    assert np.all(coarse < fine) and coarse[1] < 0.6  # bigger grains absorb more, mostly in the NIR
    assert (
        snow.snow_albedo(wl, 0.25, "heavy")[0] < snow.snow_albedo(wl, 0.25, "light")[0] < snow.snow_albedo(wl, 0.25)[0]
    )


def test_snowfall_visibility():
    v1 = 2.996 / snow.snowfall_extinction(1.0, 0.1)
    assert 500 < v1 < 2000  # ~1 km visibility at 1 mm/h (Rasmussen et al. 1999 order of magnitude)
    assert snow.snowfall_extinction(3.0, 0.1) > snow.snowfall_extinction(1.0, 0.1) > 0
    assert snow.snowfall_extinction(0.0, 0.1) == 0.0


def test_default_unchanged(tmp_path):
    scene, m = _build(tmp_path)
    assert "snow" not in m and "snow:" not in scene and not (tmp_path / "snowfall.pbrt").exists()


def test_verges(tmp_path):
    scene, m = _build(tmp_path, "--snow", "verges")
    assert 'MakeNamedMaterial "grass" "string type" "diffuse" "spectrum reflectance" "spd/snow_verge.spd"' in scene
    assert 'NamedMaterial "snow:verge"' in scene and 'NamedMaterial "snow:road"' not in scene
    assert m["snow"]["surfaces"]["verge"]["albedo_450_650nm"] > 0.95 and "banks" in m["snow"]
    assert scene.index('MakeNamedMaterial "snow:verge"') > scene.index("WorldBegin")


def test_slush_sets_wet_and_leaves_tracks(tmp_path):
    scene, m = _build(tmp_path, "--snow", "slush", "--no-snowbanks")
    assert m["road"]["wet"]["level"] == "wet" and m["snow"]["notes"] and "banks" not in m["snow"]
    assert '"snow:road" "string type" "coateddiffuse"' in scene
    geo = {"paved": [(0.0, 10.0)], "lane_centres": [1.8, 5.4, 9.0]}
    rects = snow.road_rects(geo, "slush", np.random.default_rng(0), 0.0, 3.0)
    covered = sum((b - a) for a, b, za, zb in rects if za == 0.0)
    assert 10.0 - 6 * 1.3 * snow.TRACK_W < covered < 10.0 - 6 * 0.75 * snow.TRACK_W + 1e-9
    for c in (1.8 + 0.85, 5.4 - 0.85):  # track centres stay clear
        assert not any(a < c < b for a, b, za, zb in rects if za == 0.0)


def test_packed_and_flakes(tmp_path):
    scene, m = _build(tmp_path, "--snow", "packed", "--snowfall-mmh", "0.5", "--snowfall-range-m", "6", "--fov", "40")
    assert m["snow"]["road"]["cover"].startswith("whole") and 'Include "snowfall.pbrt"' in scene
    f = m["snow"]["snowfall"]
    assert f["flakes"] > 1000 and f["equivalent_visibility_m"] > 1000
    rows = (tmp_path / "snowfall.pbrt").read_text().splitlines()
    assert len(rows) == f["flakes"]
    z = np.array([float(r.split()[4]) for r in rows])
    assert z.min() >= 0.5 and z.max() <= 6.0


def test_rejects():
    with pytest.raises(SystemExit):
        hw.main([*BASE, "--snow", "packed", "--road-wetness", "wet"])
    with pytest.raises(SystemExit):
        hw.main([*BASE, "--snowfall-mmh", "-1"])


def _write_box_ply(path: Path, x0, x1, y, z0, z1, up=True) -> None:
    from fetch_highway_assets import write_ply

    p = np.array([[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]], float)
    t = np.array([[0, 2, 1], [0, 3, 2]] if up else [[0, 1, 2], [0, 2, 3]])
    write_ply(path, p, None, None, t)


def test_top_faces_height_map():
    # table top (y = 1) over a floor plate (y = 0) and a vertical wall: only the table top and the
    # uncovered floor margin are tops; either winding counts.
    pos = np.array(
        [[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]]  # table
        + [[-1, 0, -1], [2, 0, -1], [2, 0, 2], [-1, 0, 2]]  # floor, larger than the table
        + [[0.5, 0, 0.5], [0.5, 2, 0.5], [0.5, 2, 0.6]],  # wall triangle (vertical)
        float,
    )
    tri = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [8, 9, 10]])
    cand = np.ones(len(tri), bool)
    top = snow.top_faces(pos, tri, cand)
    assert top[:2].all() and not top[4]
    # floor triangles: their centroids lie under the table -> covered
    assert not top[2] and not top[3]
    assert not snow.top_faces(pos, tri, np.zeros(len(tri), bool)).any()


def test_coat_lines(tmp_path):
    _write_box_ply(tmp_path / "roof.ply", 0, 1, 1, 0, 1)
    _write_box_ply(tmp_path / "glass.ply", 0, 1, 1.2, 1, 2, up=False)
    lines = [
        'MakeNamedMaterial "c:paint" "string type" "coateddiffuse"',
        'MakeNamedMaterial "c:WindscreenGlass" "string type" "dielectric"',
        "Scale 0.5 0.5 0.5",
        'NamedMaterial "c:paint"',
        'Shape "plymesh" "string filename" "roof.ply"',
        'NamedMaterial "c:WindscreenGlass"',
        'Shape "plymesh" "string filename" "glass.ply"',
    ]
    coat = snow.Coat(tmp_path)
    out = coat.lines(lines, "car_x")
    assert out[: len(lines)] == lines and out[-2] == 'NamedMaterial "snow:verge"'
    from highway_taillamps import read_ply_mesh

    p, t = read_ply_mesh(tmp_path / "snow_coat" / "car_x.ply")
    assert len(t) == 2  # the roof only, not the glass
    np.testing.assert_allclose(p[:, 1], 1.0 + snow.COAT_OFFSET_M / 0.5)
    assert coat.meta()["assets"] == {"car_x": 2}
    obj = coat.lines(['ObjectBegin "v"', *lines[3:5], "ObjectEnd"], "veg", 2.0)
    assert obj[-1] == "ObjectEnd" and obj[-2].startswith('Shape "plymesh"')


def test_snow_coat_args(tmp_path):
    scene, m = _build(tmp_path, "--snow", "verges")
    coated = m["snow"]["coat"]["assets"]  # empty without assets (CI); vegetation/cars where present
    for k in coated:
        assert f"snow_coat/{k}.ply" in scene and (tmp_path / "snow_coat" / f"{k}.ply").is_file()
    scene2, m2 = _build(tmp_path / "n", "--snow", "verges", "--snow-coat", "none")
    assert "coat" not in m2["snow"] and "snow_coat/" not in scene2


def test_coat_split_plants_per_object(tmp_path):
    import build_highway_scene as bhs
    from fetch_highway_assets import write_ply

    tri = np.array([[0, 1, 2]])
    for k, y in enumerate((1.0, 2.0)):
        write_ply(tmp_path / f"p{k}.ply", np.array([[0, y, 0], [1, y, 0], [0, y, 1]], float), None, None, tri)
    lines = [
        'MakeNamedMaterial "v:0" "string type" "diffuse"',
        'ObjectBegin "v#0"',
        'NamedMaterial "v:0"',
        'Shape "plymesh" "string filename" "p0.ply"',
        "ObjectEnd",
        'ObjectBegin "v#1"',
        'NamedMaterial "v:0"',
        'Shape "plymesh" "string filename" "p1.ply"',
        "ObjectEnd",
    ]
    coat = snow.Coat(tmp_path)
    out = bhs.coat_vegetation(coat, lines, "v", 1.0)
    assert out[0] == lines[0] and out.count("ObjectEnd") == 2 and out.count('ObjectBegin "v#0"') == 1
    assert sorted(coat.meta()["assets"]) == ["v#0", "v#1"]  # one coat per plant, inside its own object
    i0, i1 = out.index('ObjectBegin "v#1"'), out.index("ObjectEnd")
    assert i1 < i0 and any("snow_coat/v#0.ply" in ln for ln in out[:i1])
