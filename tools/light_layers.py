#!/usr/bin/env python3
"""Render a pbrt scene as light layers and sum them.

Radiance is linear in the emission of the lights, so a scene whose lights are split into groups
renders as one image per group, and their sum is the full image (same expectation; each layer is an
independent unbiased estimate). Every light statement (``LightSource``, ``AreaLightSource``) goes to
exactly one layer; geometry, materials and camera are the same in every layer, so an emitter that is
off in a layer is still there as a surface. Included files that hold lights are written once per
layer.

Why: a layer can be re-weighted or dropped after rendering (rear lamps brighter, dimmer or off)
without rendering the scene again, and each layer can have its own sample count and integrator
parameters. It does not lower the noise per render time of the night highway: each layer traces all
of its paths again (measured in docs/HIGHWAY_SCENES.md, "Light layers").

  python tools/light_layers.py render SCENE_DIR [--pbrt PATH] [--seed N] [--spp N]
  python tools/light_layers.py merge SCENE_DIR [--weight rear=0]

The layers are described by ``light_layers`` in the scene manifest (see ``split`` and
``build_highway_scene.py --light-layers``; docs/HIGHWAY_SCENES.md, "Light layers").
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exr_multispectral import read_separate_exr_channels, write_separate_channels_exr  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# pbrt scene tokens: quoted strings, comments, brackets, bare words (directives, numbers, true/false)
_TOKEN = re.compile(r'"[^"]*"|#[^\n]*|\[|\]|[^\s\[\]"#]+')
_DIRECTIVE = re.compile(r"[A-Z][A-Za-z]*$")
_LIGHT_DIRECTIVES = ("LightSource", "AreaLightSource")

# Rear signal lamps (tools/highway_taillamps.py, highway_night.py proxy tail lamps, highway_vru.py
# cycle lamps): tail, stop, CHMSL, fog, indicators, reversing.
REAR_LAMP_SPDS = (
    "spd/lamp_red_led.spd",
    "spd/lamp_red_incandescent.spd",
    "spd/lamp_amber_led.spd",
    "spd/lamp_amber_incandescent.spd",
    "spd/lamp_white_led.spd",
    "spd/lamp_white_incandescent.spd",
)


def rear_lamp(statement: str) -> bool:
    """True for a light statement of a rear signal lamp (by its emission spectrum)."""
    return statement.startswith("AreaLightSource") and any(f'"{s}"' in statement for s in REAR_LAMP_SPDS)


# layer name -> predicate on the light statement text; a light goes to the first layer that takes it,
# the last layer of a split takes the rest
PRESETS: dict[str, Callable[[str], bool]] = {"rear": rear_lamp}

# build_highway_scene.py --light-layers. The rear layer keeps the full sample count: the lamps seen
# directly are narrow-band, and their spectral noise grows as 1/spp (docs/HIGHWAY_SCENES.md, "Light layers").
HIGHWAY_LAYERS = [{"name": "rear", "spp_scale": 1.0}, {"name": "main", "spp_scale": 1.0}]


def statements(text: str) -> list[tuple[int, int, str]]:
    """(start, end, directive) character spans of the directives in a pbrt scene text.

    A directive's span runs to the start of the next directive (comments in between stay with it)."""
    toks = [(m.start(), m.group()) for m in _TOKEN.finditer(text) if not m.group().startswith("#")]
    starts = [(pos, tok) for pos, tok in toks if _DIRECTIVE.match(tok)]
    return [(pos, starts[i + 1][0] if i + 1 < len(starts) else len(text), tok) for i, (pos, tok) in enumerate(starts)]


def _includes(text: str) -> list[tuple[int, int, str]]:
    """(start, end, filename) spans of the quoted file names of Include/Import directives."""
    out = []
    for s, e, d in statements(text):
        if d in ("Include", "Import"):
            m = re.search(r'"([^"]*)"', text[s:e])
            if m:
                out.append((s + m.start(1), s + m.end(1), m.group(1)))
    return out


def _has_lights(path: Path, root: Path, seen: set[Path] | None = None) -> bool:
    seen = set() if seen is None else seen
    if path in seen or not path.is_file():
        return False
    seen.add(path)
    text = path.read_text()
    if any(d in _LIGHT_DIRECTIVES for _, _, d in statements(text)):
        return True
    return any(_has_lights(root / f, root, seen) for _, _, f in _includes(text))


def _classify(statement: str, layers: list[str]) -> str:
    for name in layers[:-1]:
        if PRESETS[name](statement):
            return name
    return layers[-1]


def _write_layer(src: Path, root: Path, layer: str, layers: list[str], counts: dict[str, int], memo: dict) -> Path:
    """Write ``src`` for ``layer`` (lights of other layers dropped, light-holding includes rewritten).

    Returns the path to Include: ``src`` itself when it holds no lights. pbrt resolves Include paths
    against the main scene's directory ``root``, also in nested includes."""
    if (src, layer) in memo:
        return memo[(src, layer)]
    if not _has_lights(src, root):
        memo[(src, layer)] = src
        return src
    text = src.read_text()
    edits: list[tuple[int, int, str]] = []
    for s, e, d in statements(text):
        if d in _LIGHT_DIRECTIVES:
            owner = _classify(re.sub(r"\s+", " ", text[s:e]).strip(), layers)
            counts[owner] = counts.get(owner, 0) + (layer == layers[0])
            if owner != layer:
                edits.append((s, e, f"# [light layer {owner}] {d}\n"))
    for s, e, f in _includes(text):
        child = _write_layer(root / f, root, layer, layers, counts, memo)
        if child != root / f:
            edits.append((s, e, str(Path(f).with_name(child.name))))
    for s, e, new in sorted(edits, reverse=True):
        text = text[:s] + new + text[e:]
    out = src.with_name(f"{src.stem}_layer_{layer}{src.suffix}")
    out.write_text(text)
    memo[(src, layer)] = out
    return out


def _set_integrator(text: str, params: dict[str, str]) -> str:
    """Set (replace or add) parameters of the Integrator directive, e.g. {"string lightsampler": '"uniform"'}."""
    for s, e, d in statements(text):
        if d != "Integrator":
            continue
        body = text[s:e].rstrip()
        for decl, value in params.items():
            pat = re.compile(rf'"{re.escape(decl)}"\s*(\[[^\]]*\]|\S+)')
            new = f'"{decl}" [{value}]'
            body = pat.sub(new, body) if pat.search(body) else f"{body} {new}"
        return text[:s] + body + "\n" + text[e:]
    raise ValueError("scene has no Integrator directive")


def split(scene: Path, layers: list[dict]) -> list[dict]:
    """Write one scene per layer next to ``scene``; returns the layer entries for the manifest.

    ``layers``: [{"name": preset name or the last, catch-all layer, "spp_scale": float,
    "integrator": {decl: value}}]. Each entry gets "scene" and "lights" (number of light statements)."""
    names = [lay["name"] for lay in layers]
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError(f"need two or more distinct layers, got {names}")
    if unknown := [n for n in names[:-1] if n not in PRESETS]:
        raise ValueError(f"unknown light layers {unknown} (known: {sorted(PRESETS)}; the last layer takes the rest)")
    scene = Path(scene)
    counts: dict[str, int] = {}
    memo: dict = {}
    out = []
    for lay in layers:
        path = _write_layer(scene, scene.parent, lay["name"], names, counts, memo)
        if lay.get("integrator"):
            path.write_text(_set_integrator(path.read_text(), lay["integrator"]))
        out.append({**lay, "scene": path.name})
    for entry in out:
        entry["lights"] = counts.get(entry["name"], 0)
    if empty := [e["name"] for e in out if e["lights"] == 0]:
        print(f"warning: light layers without lights: {empty}", file=sys.stderr)
    return out


def layer_output(film_output: str, name: str) -> str:
    """Film path of one layer: ``out/highway_spectral.exr`` -> ``out/highway_spectral_layer_rear.exr``."""
    p = Path(film_output)
    return str(p.with_name(f"{p.stem}_layer_{name}{p.suffix}"))


def _manifest(scene_dir: Path) -> tuple[Path, dict]:
    found = sorted(Path(scene_dir).glob("*_manifest.json"))
    for m in found:
        meta = json.loads(m.read_text())
        if "light_layers" in meta:
            return m, meta
    raise ValueError(f"no *_manifest.json with light_layers in {scene_dir} (build with --light-layers)")


def merge(scene_dir: Path, weights: dict[str, float] | None = None, tag: str = "") -> Path:
    """Sum the layer EXRs (times ``weights``, default 1) into the manifest's film output; return it."""
    scene_dir = Path(scene_dir)
    _, meta = _manifest(scene_dir)
    film = meta["film"]["filename"]
    weights = weights or {}
    if unknown := set(weights) - {e["name"] for e in meta["light_layers"]["layers"]}:
        raise ValueError(f"unknown layers in weights: {sorted(unknown)}")
    total = None
    for entry in meta["light_layers"]["layers"]:
        d = read_separate_exr_channels(scene_dir / _tagged(layer_output(film, entry["name"]), tag))
        w = float(weights.get(entry["name"], 1.0))
        if total is None:
            total = {k: w * v.astype(np.float64) for k, v in d.items()}
        elif set(d) != set(total):
            raise ValueError(f"layer {entry['name']}: channel mismatch {sorted(set(d) ^ set(total))}")
        else:
            for k in total:
                total[k] += w * d[k]
    out = scene_dir / _tagged(film, tag)
    write_separate_channels_exr(out, {k: v.astype(np.float32) for k, v in total.items()})
    return out


def _tagged(name: str, tag: str) -> str:
    p = Path(name)
    return str(p.with_name(f"{p.stem}{tag}{p.suffix}"))


def render(
    scene_dir: Path, pbrt: Path, seed: int = 0, spp: int | None = None, extra: list[str] = (), tag: str = ""
) -> Path:
    """Render every layer (its own seed and ``spp`` x its spp_scale) and merge; return the film EXR."""
    scene_dir = Path(scene_dir).resolve()
    _, meta = _manifest(scene_dir)
    base_spp = spp or meta["light_layers"]["pixelsamples"]
    for k, entry in enumerate(meta["light_layers"]["layers"]):
        n = max(1, round(base_spp * float(entry.get("spp_scale", 1.0))))
        out = _tagged(layer_output(meta["film"]["filename"], entry["name"]), tag)
        (scene_dir / out).parent.mkdir(parents=True, exist_ok=True)
        # independent layers: a different seed each (the same seed would correlate their samples)
        cmd = [str(pbrt), "--quiet", "--seed", str(seed * 1009 + k), "--spp", str(n), "--outfile", out]
        subprocess.run([*cmd, *extra, entry["scene"]], check=True, cwd=scene_dir)
    return merge(scene_dir, tag=tag)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render", help="render all layers and merge")
    r.add_argument("scene_dir", type=Path)
    r.add_argument("--pbrt", type=Path, default=REPO / "third_party" / "pbrt-v4" / "build" / "pbrt")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--spp", type=int, default=None, help="base samples per pixel (default: the scene's)")
    r.add_argument("--tag", default="", help="suffix for the output file names, e.g. _seed1")
    m = sub.add_parser("merge", help="sum rendered layers (optionally re-weighted)")
    m.add_argument("scene_dir", type=Path)
    m.add_argument("--weight", action="append", default=[], metavar="LAYER=W", help="e.g. rear=0 (lamps off)")
    m.add_argument("--tag", default="")
    args, extra = ap.parse_known_args(argv)
    if args.cmd == "render":
        out = render(args.scene_dir, args.pbrt, args.seed, args.spp, extra, args.tag)
    else:
        if extra:
            ap.error(f"unrecognized arguments: {extra}")
        weights = {k: float(v) for k, v in (w.split("=", 1) for w in args.weight)}
        out = merge(args.scene_dir, weights, args.tag)
    print(out)


if __name__ == "__main__":
    main()
