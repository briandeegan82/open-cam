#!/usr/bin/env python3
"""Rolling-shutter / LED-flicker rendering of a highway scene by time slicing.

pbrt has a global shutter only. This renders the scene built by tools/build_highway_scene.py
(with ``--exposure-s``) as a set of slices, each a pbrt run over a row band (``--pixelbounds``)
and a sub-interval of that band's exposure window (shutter open/close), and composites them:

* Rolling shutter: row r integrates [r * line_time, r * line_time + integration_time]
  (``exposure`` in the manifest). Rows are grouped into ``--bands`` bands that use the window of
  their centre row; the timing error is at most half a band of line times.
* LED flicker: every PWM emitter in ``manifest["emitters"]`` (tools/highway_incar.py
  ``write_emitter``) is switched to its "on"/"off" include for each sub-interval between PWM
  edges inside the window, so a short exposure can miss an LED's on-pulse entirely.

Each slice renders the time-averaged radiance over its sub-interval; the composite weights it by
``duration / integration_time``, so the result is the exposure-averaged radiance, in the same
units as a normal render (absolute radiometry and the electrons conversion are unchanged).
Samples per pixel are shared out in proportion to slice duration (at least 1).

    venv/bin/python tools/render_time_slices.py scenes/generated/highway --spp 64 --bands 24
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exr_multispectral import read_separate_exr_channels, write_separate_channels_exr  # noqa: E402
from highway_incar import PWM  # noqa: E402


@dataclass
class Slice:
    y0: int
    y1: int
    t0: float
    t1: float
    weight: float
    states: dict[str, str] = field(default_factory=dict)
    spp: int = 1


def plan_slices(
    exposure: dict, emitters: list[dict], yres: int, bands: int, spp: int, flicker: bool = True
) -> list[Slice]:
    t_int = float(exposure["integration_time_s"])
    t_open = float(exposure.get("shutter_open_s", 0.0))
    line = float(exposure.get("rolling_shutter_line_time_s", 0.0))
    nb = 1 if line <= 0 else max(1, min(int(bands), yres))
    edges = np.linspace(0, yres, nb + 1).round().astype(int)
    pwms = {e["id"]: PWM(**e["pwm"]) for e in emitters if flicker and e.get("pwm")}
    out: list[Slice] = []
    for y0, y1 in zip(edges[:-1], edges[1:]):
        if y1 <= y0:
            continue
        a = t_open + 0.5 * (y0 + y1 - 1) * line
        b = a + t_int
        cuts = sorted({a, b, *(x for p in pwms.values() for x in p.edges(a, b))})
        for t0, t1 in zip(cuts[:-1], cuts[1:]):
            mid = 0.5 * (t0 + t1)
            states = {eid: "on" if p.is_on(mid) else "off" for eid, p in pwms.items()}
            w = (t1 - t0) / t_int
            out.append(Slice(int(y0), int(y1), t0, t1, w, states, max(1, round(spp * w))))
    return out


ISO_RE = re.compile(r'"float iso" \[[^\]]*\]')
SHUTTER_RE = re.compile(r'"float shutteropen" \[[^\]]*\] "float shutterclose" \[[^\]]*\]')


def slice_scene_text(scene: str, sl: Slice, emitters: list[dict]) -> str:
    if not SHUTTER_RE.search(scene):
        raise ValueError("scene has no shutter (build it with --exposure-s)")
    text = SHUTTER_RE.sub(f'"float shutteropen" [{sl.t0:.9g}] "float shutterclose" [{sl.t1:.9g}]', scene, count=1)
    # Keep each slice in radiance units (pbrt multiplies by shutter duration x ISO / 100).
    text = ISO_RE.sub(f'"float iso" [{100.0 / (sl.t1 - sl.t0):.9g}]', text, count=1)
    for e in emitters:
        st = sl.states.get(e["id"])
        if st is None:
            continue
        old = f'Include "{e["include"]}"'
        if old not in text:
            raise ValueError(f"emitter {e['id']}: {old} not found in scene")
        text = text.replace(old, f'Include "{e["states"][st]}"')
    return text


def default_pbrt(repo: Path) -> Path:
    return Path(os.environ.get("PBRT", repo / "third_party/pbrt-v4/build/pbrt"))


def render(
    scene_dir: Path,
    *,
    pbrt: Path,
    spp: int,
    bands: int,
    out: Path,
    jobs: int = 1,
    flicker: bool = True,
    keep: bool = False,
    exposure_override: dict | None = None,
) -> dict:
    manifest = json.loads((scene_dir / "highway_manifest.json").read_text())
    if "exposure" not in manifest:
        raise ValueError(f"{scene_dir}: manifest has no exposure (build with --exposure-s)")
    xres, yres = int(manifest["film"]["xresolution"]), int(manifest["film"]["yresolution"])
    emitters = manifest.get("emitters", [])
    exposure = {**manifest["exposure"], **(exposure_override or {})}
    slices = plan_slices(exposure, emitters, yres, bands, spp, flicker)
    scene = (scene_dir / "highway.pbrt").read_text()
    work = scene_dir / "slices"
    work.mkdir(exist_ok=True)
    nthreads = max(1, (os.cpu_count() or 1) // max(1, jobs))

    def run(i: int) -> tuple[int, dict[str, np.ndarray], float]:
        sl = slices[i]
        sp = scene_dir / f".slice_{i:04d}.pbrt"
        exr = work / f"slice_{i:04d}.exr"
        sp.write_text(slice_scene_text(scene, sl, emitters))
        cmd = [str(pbrt), "--quiet", "--nthreads", str(nthreads), "--spp", str(sl.spp)]
        cmd += ["--pixelbounds", f"0,{xres},{sl.y0},{sl.y1}", "--outfile", str(exr.resolve()), sp.name]
        t = time.perf_counter()
        try:
            subprocess.run(cmd, cwd=scene_dir, check=True, stdout=subprocess.DEVNULL)
        finally:
            sp.unlink(missing_ok=True)
        ch = read_separate_exr_channels(exr)
        if not keep:
            exr.unlink()
        return i, ch, time.perf_counter() - t

    acc: dict[str, np.ndarray] = {}
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        for i, ch, _ in ex.map(run, range(len(slices))):
            sl = slices[i]
            for name, arr in ch.items():
                if name not in acc:
                    acc[name] = np.zeros((yres, xres), np.float64)
                band = sl.y1 - sl.y0
                arr = arr[-band:] if arr.shape[0] > band else arr  # pbrt may write full or cropped rows
                acc[name][sl.y0 : sl.y1] += sl.weight * arr.astype(np.float64)
    wall = time.perf_counter() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    write_separate_channels_exr(out, {k: v.astype(np.float32) for k, v in acc.items()})
    if not keep and not any(work.iterdir()):
        work.rmdir()
    info = {
        "output": str(out),
        "slices": len(slices),
        "bands": len({(s.y0, s.y1) for s in slices}),
        "samples_per_pixel_total": int(max(sum(s.spp for s in slices if s.y0 == y0) for y0 in {s.y0 for s in slices})),
        "wall_time_s": wall,
        "plan": [s.__dict__ for s in slices],
    }
    out.with_suffix(".slices.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main(argv: list[str] | None = None) -> None:
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_dir", type=Path, nargs="?", default=repo / "scenes/generated/highway")
    ap.add_argument("--pbrt", type=Path, default=default_pbrt(repo))
    ap.add_argument("--spp", type=int, default=64, help="Samples per pixel per row (shared across time slices).")
    ap.add_argument("--bands", type=int, default=24, help="Row bands for the rolling shutter.")
    ap.add_argument("--jobs", type=int, default=1, help="Concurrent pbrt processes (threads are split).")
    ap.add_argument("--no-flicker", action="store_true", help="Use the frame-averaged emitter levels instead.")
    ap.add_argument("--keep", action="store_true", help="Keep the per-slice EXRs in <scene_dir>/slices.")
    ap.add_argument("--out", type=Path, default=None, help="default: the scene's film filename")
    ap.add_argument("--shutter-open-s", type=float, default=None, help="Override the manifest exposure start (s).")
    ap.add_argument("--integration-time-s", type=float, default=None, help="Override the manifest integration time.")
    args = ap.parse_args(argv)
    sd = args.scene_dir.resolve()
    out = args.out
    if out is None:
        m = re.search(r'"string filename" \["([^"]+)"\]', (sd / "highway.pbrt").read_text())
        out = sd / (m.group(1) if m else "out/highway_spectral.exr")
    info = render(
        sd,
        pbrt=args.pbrt,
        spp=args.spp,
        bands=args.bands,
        out=out.resolve(),
        jobs=args.jobs,
        flicker=not args.no_flicker,
        keep=args.keep,
        exposure_override={
            k: v
            for k, v in (("shutter_open_s", args.shutter_open_s), ("integration_time_s", args.integration_time_s))
            if v is not None
        },
    )
    print(f"wrote {info['output']}: {info['slices']} slices in {info['bands']} bands, {info['wall_time_s']:.1f} s")


if __name__ == "__main__":
    main()
