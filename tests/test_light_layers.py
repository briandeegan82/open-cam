"""tools/light_layers.py: split a scene's lights into layers, render each, sum them."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import light_layers as ll
import numpy as np
import pytest
from exr_multispectral import read_separate_exr_channels, write_separate_channels_exr
from synthetic_data import REPO

PBRT = Path(os.environ.get("OPENCAM_PBRT", REPO / "third_party" / "pbrt-v4" / "build" / "pbrt"))

RED = '"spd/lamp_red_led.spd"'


def _scene(d: Path) -> Path:
    """Floor lit by a white quad and a sky, plus a red 'rear lamp' quad in an Included file."""
    (d / "spd").mkdir(parents=True, exist_ok=True)
    (d / "spd" / "lamp_red_led.spd").write_text("360 0\n600 0\n630 1\n660 0\n830 0\n")
    (d / "cars").mkdir(exist_ok=True)
    (d / "cars" / "car.pbrt").write_text(
        "AttributeBegin\n"
        f'    AreaLightSource "diffuse" "spectrum L" {RED} "float scale" [2]\n'
        '    Shape "bilinearmesh" "point3 P" [0.6 0.2 -0.5  1.0 0.2 -0.5  0.6 0.5 -0.5  1.0 0.5 -0.5]\n'
        "AttributeEnd\n"
        '# a comment that mentions LightSource "point" must not count\n'
    )
    scene = d / "s.pbrt"
    scene.write_text(
        "\n".join(
            [
                "LookAt 0 3 -2  0 0 0  0 1 0",
                'Camera "perspective" "float fov" [40]',
                'Sampler "independent" "integer pixelsamples" [64]',
                'Integrator "path" "integer maxdepth" [3]',
                'Film "rgb" "integer xresolution" [24] "integer yresolution" [24] "string filename" "out/s.exr"',
                "WorldBegin",
                'LightSource "infinite" "blackbody L" [6500]',
                '    "float scale" [0.05]',
                'AttributeBegin AreaLightSource "diffuse" "blackbody L" [3000] "float scale" [3]',
                'Shape "bilinearmesh" "point3 P" [-1.5 1 -0.5  -1 1 -0.5  -1.5 1.5 -0.5  -1 1.5 -0.5] AttributeEnd',
                'Include "cars/car.pbrt"',
                'Material "diffuse" "rgb reflectance" [0.5 0.5 0.5]',
                'Shape "bilinearmesh" "point3 P" [-5 0 -5  5 0 -5  -5 0 5  5 0 5]',
            ]
        )
        + "\n"
    )
    return scene


def _lights(path: Path, root: Path) -> list[str]:
    text = path.read_text()
    own = [text[s:e] for s, e, d in ll.statements(text) if d in ll._LIGHT_DIRECTIVES]
    return own + [x for _, _, f in ll._includes(text) for x in _lights(root / f, root)]


def test_statements_skip_comments_and_keep_parameter_lines(tmp_path):
    scene = _scene(tmp_path)
    found = [(d, scene.read_text()[s:e]) for s, e, d in ll.statements(scene.read_text())]
    infinite = [t for d, t in found if d == "LightSource"]
    assert len(infinite) == 1 and '"float scale" [0.05]' in infinite[0]
    assert len(_lights(tmp_path / "cars" / "car.pbrt", tmp_path)) == 1


def test_split_puts_each_light_in_one_layer(tmp_path):
    scene = _scene(tmp_path)
    layers = ll.split(scene, [{"name": "rear", "integrator": {"string lightsampler": '"uniform"'}}, {"name": "main"}])
    assert [(e["name"], e["lights"]) for e in layers] == [("rear", 1), ("main", 2)]
    rear, main = (_lights(tmp_path / e["scene"], tmp_path) for e in layers)
    assert len(rear) == 1 and RED in rear[0]
    assert len(main) == 2 and not any(RED in t for t in main)
    rear_text = (tmp_path / layers[0]["scene"]).read_text()
    assert 'Include "cars/car_layer_rear.pbrt"' in rear_text
    assert 'Integrator "path" "integer maxdepth" [3] "string lightsampler" ["uniform"]' in rear_text
    # geometry is the same in both: the rear lamp quad stays in the main layer as a surface
    assert "0.6 0.2 -0.5" in (tmp_path / "cars" / "car_layer_main.pbrt").read_text()
    with pytest.raises(ValueError):
        ll.split(scene, [{"name": "nosuch"}, {"name": "main"}])


def test_set_integrator_replaces_existing_parameter():
    text = 'Integrator "path" "string lightsampler" "bvh" "integer maxdepth" [5]\nWorldBegin\n'
    out = ll._set_integrator(text, {"string lightsampler": '"uniform"', "integer maxdepth": "2"})
    assert out.startswith('Integrator "path" "string lightsampler" ["uniform"] "integer maxdepth" [2]\n')


def test_merge_sums_and_weights_layers(tmp_path):
    (tmp_path / "out").mkdir()
    layers = [{"name": "rear"}, {"name": "main"}]
    meta = {"film": {"filename": "out/x.exr"}, "light_layers": {"pixelsamples": 4, "layers": layers}}
    (tmp_path / "x_manifest.json").write_text(json.dumps(meta))
    for name, v in (("rear", 1.0), ("main", 2.0)):
        write_separate_channels_exr(
            tmp_path / ll.layer_output("out/x.exr", name), {c: np.full((2, 3), v, np.float32) for c in ("R", "G", "B")}
        )
    assert read_separate_exr_channels(ll.merge(tmp_path))["G"].tolist() == [[3.0] * 3] * 2
    assert read_separate_exr_channels(ll.merge(tmp_path, {"rear": 0}))["G"].tolist() == [[2.0] * 3] * 2


@pytest.mark.skipif(not PBRT.is_file(), reason=f"pbrt binary not built ({PBRT})")
def test_layered_render_matches_single_render(tmp_path):
    scene = _scene(tmp_path)
    layers = ll.split(scene, [{"name": "rear", "spp_scale": 0.5}, {"name": "main"}])
    meta = {"film": {"filename": "out/s.exr"}, "light_layers": {"pixelsamples": 256, "layers": layers}}
    (tmp_path / "s_manifest.json").write_text(json.dumps(meta))
    lay = [ll.render(tmp_path, PBRT, seed=s, tag=f"_l{s}") for s in (1, 2)]
    one = []
    for s in (1, 2):
        out = f"out/one_{s}.exr"
        subprocess.run(
            [str(PBRT), "--quiet", "--seed", str(s), "--spp", "256", "--outfile", out, "s.pbrt"],
            check=True,
            cwd=tmp_path,
        )
        one.append(tmp_path / out)

    def mean(paths):
        return np.mean([np.mean([read_separate_exr_channels(p)[c] for c in "RGB"]) for p in paths])

    assert mean(lay) == pytest.approx(mean(one), rel=0.02)
