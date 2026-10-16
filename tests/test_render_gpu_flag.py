"""render.gpu_enabled adds pbrt --gpu only on GPU builds, never --wavefront."""

from __future__ import annotations

import os

import run_pipeline as rp
from synthetic_data import REPO


def test_gpu_flag_only_with_gpu_build():
    render = {"gpu_enabled": True, "pbrt_args": ["--stats"]}
    assert rp.parse_render_pbrt_args(render, gpu_available=False) == ["--stats"]
    assert rp.parse_render_pbrt_args(render, gpu_available=True) == ["--gpu", "--stats"]
    assert rp.parse_render_pbrt_args({"gpu_enabled": False}, gpu_available=True) == []
    assert rp.parse_render_pbrt_args({"gpu_enabled": True, "pbrt_args": ["--gpu"]}, True) == ["--gpu"]


def test_pbrt_has_gpu_reads_help(tmp_path):
    fake = tmp_path / "pbrt"
    fake.write_text("#!/bin/sh\necho '  --gpu                         Use the GPU for rendering.'\n")
    fake.chmod(0o755)
    assert rp.pbrt_has_gpu(fake)
    assert not rp.pbrt_has_gpu(tmp_path / "missing")
    cpu_build = REPO / "third_party/pbrt-v4/build/pbrt"
    if os.access(cpu_build, os.X_OK):
        assert rp.pbrt_has_gpu(cpu_build) is False
