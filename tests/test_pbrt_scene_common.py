from __future__ import annotations

import numpy as np
from pbrt_scene_common import rel_to, rgb_film_lines, spectral_film_lines, write_spd_table


def test_film_blocks():
    assert spectral_film_lines("out/a.exr", 640, 480.0, 32, 360, 830.0) == [
        'Film "spectral"',
        '    "string filename" ["out/a.exr"]',
        '    "integer xresolution" [640]',
        '    "integer yresolution" [480]',
        '    "bool savefp16" false',
        '    "integer nbuckets" [32]',
        '    "float lambdamin" [360]',
        '    "float lambdamax" [830]',
    ]
    assert rgb_film_lines("o.exr", 4, 2)[-1] == '    "float iso" [100]'
    assert rgb_film_lines("o.exr", 4, 2, iso=None)[-1] == '    "bool savefp16" false'


def test_spd_table_and_rel(tmp_path):
    p = tmp_path / "d" / "x.spd"
    write_spd_table(p, np.array([400.0, 405.0]), np.array([0.5, 1 / 3]))
    assert p.read_text() == "# wavelength_nm\tvalue\n400\t0.5\n405\t0.333333333333\n"
    assert rel_to(tmp_path, p) == "d/x.spd"
    assert rel_to(tmp_path / "d", tmp_path / "e") == str(tmp_path / "e")
