"""tools/plant_variants.py: split Poly Haven plant rows into separate plants."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import plant_variants as pv  # noqa: E402
from fetch_highway_assets import write_ply  # noqa: E402

PREP = ROOT / "scenes" / "assets" / "highway" / "prepared"


def _box(x0, x1, y1, z0=-0.3, z1=0.3, y0=0.0, n=6) -> np.ndarray:
    g = np.linspace(0, 1, n)
    return np.array([[x0 + a * (x1 - x0), y0 + b * (y1 - y0), z0 + c * (z1 - z0)] for a in g for b in g for c in g])


def _asset(tmp: Path, boxes) -> dict:
    shapes = []
    for k, (b, mat) in enumerate(boxes):
        tri = np.array([[0, 1, 2]] * 4, dtype=np.int32)
        write_ply(tmp / f"mesh_{k:03d}.ply", b.astype(np.float32), None, None, tri)
        shapes.append({"ply": f"mesh_{k:03d}.ply", "material": mat, "triangles": 4})
    return {"shapes": shapes}


def test_row_of_trees_splits_trunk_with_crown(tmp_path: Path) -> None:
    boxes = []
    for cx, h in ((0.0, 10.0), (6.0, 8.0), (12.0, 6.0)):
        boxes += [(_box(cx - 0.3, cx + 0.3, h * 0.6), 0), (_box(cx - 2, cx + 2, h, y0=h * 0.3, z0=-2, z1=2), 1)]
    boxes.append((_box(20.0, 20.1, 0.5), 0))  # stray sprig below 30 % of the tallest
    plants = pv.split("veg_test", _asset(tmp_path, boxes), tmp_path)
    assert [p["shapes"] for p in plants] == [[0, 1], [2, 3], [4, 5]]
    np.testing.assert_allclose([p["centre"][0] for p in plants], [0, 6, 12], atol=0.05)
    np.testing.assert_allclose([p["height_m"] for p in plants], [10, 8, 6], atol=0.05)


def test_single_plant_and_whole_list(tmp_path: Path) -> None:
    info = _asset(tmp_path, [(_box(-1, 1, 3), 0), (_box(-2, 2, 4, y0=1, z0=-2, z1=2), 1)])
    assert len(pv.split("veg_test", info, tmp_path)) == 1
    info = _asset(tmp_path, [(_box(0, 0.05, 0.1), 0), (_box(1, 1.05, 0.1), 0)])
    assert len(pv.split("veg_test", info, tmp_path)) == 2
    assert len(pv.split(next(iter(pv.WHOLE)), info, tmp_path)) == 1


EXPECTED = {
    "veg_fir_tree_01": 3,
    "veg_fir_sapling": 3,
    "veg_fir_sapling_medium": 3,
    "veg_pine_sapling_small": 3,
    "veg_searsia_burchellii": 3,
    "veg_searsia_lucida": 7,
    "veg_wild_rooibos_bush": 5,
    "veg_shrub_02": 4,
    "veg_shrub_03": 4,
    "veg_grass_medium_02": 5,
    "veg_island_tree_02": 1,
    "veg_jacaranda_tree": 1,
    "veg_grass_bermuda_01": 1,
}


@pytest.mark.parametrize("vid", sorted(EXPECTED))
def test_prepared_assets(vid: str) -> None:
    d = PREP / vid
    if not (d / "prepared.json").is_file():
        pytest.skip(f"{vid} not prepared")
    info = json.loads((d / "prepared.json").read_text())
    plants = pv.split(vid, info, d)
    assert len(plants) == EXPECTED[vid]
    assert sorted(k for p in plants for k in p["shapes"]) == sorted(set(k for p in plants for k in p["shapes"]))
    assert all(p["height_m"] > 0.1 for p in plants)
