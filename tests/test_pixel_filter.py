"""Scene builders must use the pixel aperture (0.5 px box) as pbrt's pixel filter."""

from __future__ import annotations

import re

from synthetic_data import REPO

BOX = 'PixelFilter "box" "float xradius" [0.5] "float yradius" [0.5]'


def test_scene_builders_use_box_pixel_filter():
    builders = sorted((REPO / "tools").glob("build_*.py"))
    emitting = [p for p in builders if "PixelFilter" in p.read_text()]
    assert len(emitting) >= 10
    for path in emitting:
        lines = re.findall(r"PixelFilter[^'\"\n]*\"[^\n]*", path.read_text())
        assert lines and all(BOX in ln for ln in lines), (path.name, lines)
