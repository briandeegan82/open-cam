"""Winter / snow options of the highway builder (tools/highway_snow.py)."""

import json
import math
import os
import subprocess
import sys
import tempfile
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
    assert 'MakeNamedMaterial "grass" "string type" "snow" "spectrum reflectance" "spd/snow_verge.spd"' in scene
    assert 'MakeNamedMaterial "snow:verge" "string type" "snow"' in scene and m["snow"]["brdf"].startswith("ART")
    assert 'NamedMaterial "snow:verge"' in scene and 'NamedMaterial "snow:road"' not in scene
    assert m["snow"]["surfaces"]["verge"]["albedo_450_650nm"] > 0.95 and "banks" in m["snow"]
    assert scene.index('MakeNamedMaterial "snow:verge"') > scene.index("WorldBegin")


def test_slush_sets_wet_and_leaves_tracks(tmp_path):
    scene, m = _build(tmp_path, "--snow", "slush", "--no-snowbanks")
    assert m["road"]["wet"]["level"] == "wet" and m["snow"]["notes"] and "banks" not in m["snow"]
    assert '"snow:road" "string type" "coateddiffuse"' in scene
    assert m["snow"]["road"]["marking_cover"] == 1.0 and "exposed_lane_lines" not in m["snow"]["road"]
    geo = {"paved": [(0.0, 10.0)], "lane_centres": [1.8, 5.4, 9.0], "lane_lines": [3.6, 7.2]}
    quads, exposed = snow.slush_quads(geo, np.random.default_rng(0), 0.0, 30.0)
    assert exposed == {}
    first = [q for q in quads if q[4] == 0.0]
    covered = sum(x1a - x0a for x0a, x1a, x0b, x1b, za, zb in first)
    assert 10.0 - 6 * 1.3 * snow.TRACK_W < covered < 10.0 - 6 * 0.75 * snow.TRACK_W + 1e-9
    for c in (1.8 + 0.85, 5.4 - 0.85):  # track centres stay clear
        assert not any(a < c < b for a, b, *_ in first)
    for x in geo["lane_lines"]:  # lane lines lie between the tracks: under the slush
        assert all(any(q[0] < x < q[1] and q[2] < x < q[3] for q in quads if q[4] == za) for za in (0.0, 10.0, 29.5))
    # continuous edges: each strip's far edge is the next segment's near edge
    by_z = {}
    for q in quads:
        by_z.setdefault(q[4], []).append(q)
    zs = sorted(by_z)
    for za, zb in zip(zs[:-1], zs[1:], strict=True):
        far = sorted((q[2], q[3]) for q in by_z[za])
        near = sorted((q[0], q[1]) for q in by_z[zb])
        assert np.allclose(far, near)


def test_slush_marking_cover(tmp_path):
    geo = {"paved": [(0.0, 10.0)], "lane_centres": [1.8, 5.4, 9.0], "lane_lines": [3.6, 7.2]}
    quads, exposed = snow.slush_quads(geo, np.random.default_rng(1), 0.0, 2000.0, cover=0.6)
    for x in geo["lane_lines"]:
        assert 0.25 < exposed[f"{x:.2f}"] / 2000.0 < 0.55  # ~40 % exposed
        bare = [q for q in quads if q[1] == x - snow.LINE_CLEAR_W / 2]
        assert bare and not any(q[0] < x < q[1] for q in bare)
    full, _ = snow.slush_quads(geo, np.random.default_rng(1), 0.0, 2000.0, cover=1.0)
    assert len(quads) > len(full)
    assert snow.exposed_runs(np.random.default_rng(0), 0.0, 0.0, 50.0) == [(0.0, 50.0)]
    _, m = _build(tmp_path, "--snow", "slush", "--slush-marking-cover", "0.5")
    assert m["snow"]["road"]["exposed_lane_lines"]["exposed_length_m"]


def test_lambertian_opt_out(tmp_path):
    scene, m = _build(tmp_path, "--snow", "packed", "--snow-brdf", "lambertian")
    assert '"string type" "snow"' not in scene and m["snow"]["brdf"] == "Lambertian"
    assert 'MakeNamedMaterial "snow:road" "string type" "diffuse"' in scene


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
    with pytest.raises(SystemExit):
        hw.main([*BASE, "--snow", "packed", "--slush-marking-cover", "0.5"])
    with pytest.raises(SystemExit):
        hw.main([*BASE, "--snow", "slush", "--slush-marking-cover", "1.5"])


def art_r(rs: float, mu0: float, mu: float, cos_scatter: float) -> float:
    """ART snow reflection function (Kokhanovsky & Zege 2004; R0 of Kokhanovsky et al. 2005)."""
    th = math.degrees(math.acos(cos_scatter))
    p = 11.1 * math.exp(-0.087 * th) + 1.1 * math.exp(-0.014 * th)
    r0 = (1.247 + 1.186 * (mu + mu0) + 5.157 * mu * mu0 + p) / (4 * (mu + mu0))
    return r0 * rs ** ((3 / 7) ** 2 * (1 + 2 * mu0) * (1 + 2 * mu) / r0)


def test_art_reflection_function():
    # non-absorbing snow: plane albedo ~1; absorbing: close to the ART plane albedo rs^u(mu0)
    for rs in (1.0, 0.9):
        for mu0 in (1.0, 0.5, 0.2):
            n, acc = 120, 0.0
            for i in range(n):
                t = (i + 0.5) / n * math.pi / 2
                for j in range(36):
                    ph = (j + 0.5) * math.pi / 18
                    s0 = math.sqrt(1 - mu0 * mu0)
                    c = -(mu0 * math.cos(t) + s0 * math.sin(t) * math.cos(ph))
                    acc += (
                        art_r(rs, mu0, math.cos(t), c) * math.cos(t) * math.sin(t) * (math.pi / 2 / n) * (math.pi / 18)
                    )
            assert abs(acc / math.pi - rs ** (3 / 7 * (1 + 2 * mu0))) < 0.03
    # forward scattering at a low sun: brighter than nadir, which is darker than Lambertian
    mu0 = math.cos(math.radians(67))
    assert art_r(0.9, mu0, 1.0, -mu0) < 0.9 < art_r(0.9, mu0, math.cos(math.radians(75)), math.cos(math.radians(38)))


PBRT = Path(os.environ.get("OPENCAM_PBRT", REPO / "third_party" / "pbrt-v4" / "build" / "pbrt"))


def _pbrt_has_snow() -> bool:
    if not PBRT.is_file():
        return False
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "t.pbrt").write_text(
            'Film "rgb" "integer xresolution" [1] "integer yresolution" [1] "string filename" "t.exr"\n'
            'Camera "perspective"\nWorldBegin\nMaterial "snow"\n'
        )
        return subprocess.run([str(PBRT), "--quiet", "t.pbrt"], capture_output=True, cwd=td).returncode == 0


PBRT_SNOW = _pbrt_has_snow()
if os.environ.get("OPENCAM_REQUIRE_PBRT") == "1" and not PBRT_SNOW:
    raise RuntimeError(f"OPENCAM_REQUIRE_PBRT=1 but {PBRT} lacks the snow material patch (tools/build_pbrt.sh)")


@pytest.mark.skipif(not PBRT_SNOW, reason="patched pbrt not built; see docs/BUILD_PBRT.txt")
def test_pbrt_snow_material(tmp_path):
    from exr_multispectral import read_separate_exr_channels

    def render(mat: str, ts: float, tv: float, az: float) -> float:
        s = (math.sin(math.radians(ts)), math.cos(math.radians(ts)), 0.0)
        v = [math.sin(math.radians(tv)) * math.cos(math.radians(az)), math.cos(math.radians(tv))]
        v.append(math.sin(math.radians(tv)) * math.sin(math.radians(az)))
        up = "0 0 1" if tv == 0 else "0 1 0"
        (tmp_path / "s.pbrt").write_text(
            f"LookAt {10 * v[0]} {10 * v[1]} {10 * v[2]} 0 0 0 {up}\n"
            'Camera "perspective" "float fov" [1]\nSampler "independent" "integer pixelsamples" [16]\n'
            'Integrator "path" "integer maxdepth" [1]\n'
            'Film "rgb" "integer xresolution" [4] "integer yresolution" [4] "string filename" "o.exr"\n'
            f'WorldBegin\nLightSource "distant" "point3 from" [{s[0]} {s[1]} {s[2]}] "point3 to" [0 0 0]\n'
            f'Material "{mat}" "spectrum reflectance" [360 0.9 830 0.9]\n'
            'Shape "trianglemesh" "point3 P" [-100 0 -100 100 0 -100 100 0 100 -100 0 100] "integer indices" [0 1 2 0 2 3]\n'
        )
        subprocess.run([str(PBRT), "--quiet", "--seed", "3", "s.pbrt"], check=True, cwd=tmp_path, capture_output=True)
        return float(np.mean(read_separate_exr_channels(tmp_path / "o.exr")["G"]))

    for ts, tv, az in ((67, 0, 0), (67, 75, 180), (67, 60, 0), (30, 60, 90)):
        r = 0.9 * render("snow", ts, tv, az) / render("diffuse", ts, tv, az)
        mu0, mu = math.cos(math.radians(ts)), math.cos(math.radians(tv))
        wi = np.array([math.sin(math.radians(ts)), 0.0, mu0])
        wo = np.array([math.sin(math.radians(tv)) * math.cos(math.radians(az)), 0.0, mu])
        wo[1] = math.sin(math.radians(tv)) * math.sin(math.radians(az))
        assert r == pytest.approx(art_r(0.9, mu0, mu, float(-wi @ wo)), rel=2e-3)


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
