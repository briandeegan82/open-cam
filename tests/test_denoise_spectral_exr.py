"""Opt-in OIDN denoiser stage (tools/denoise_spectral_exr.py): PFM I/O, bucket grouping, pipeline wiring.

Most tests use a fake ``oidnDenoise`` (3x3 box blur); the real-binary test runs only when OIDN is installed
(``tools/fetch_oidn.py``).
"""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import denoise_spectral_exr as dn  # noqa: E402

FAKE_OIDN = """#!{python}
import sys
import numpy as np
sys.path.insert(0, {tools!r})
import denoise_spectral_exr as dn
a = sys.argv[1:]
img = dn.read_pfm(a[a.index("--hdr") + 1])
pad = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")
out = sum(pad[i : i + img.shape[0], j : j + img.shape[1]] for i in range(3) for j in range(3)) / 9.0
dn.write_pfm(a[a.index("-o") + 1], out)
with open({log!r}, "a") as f:
    f.write(" ".join(a) + "\\n")
"""


def _box3(img):
    pad = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")
    return sum(pad[i : i + img.shape[0], j : j + img.shape[1]] for i in range(3) for j in range(3)) / 9.0


@pytest.fixture
def fake_oidn(tmp_path):
    exe = tmp_path / "oidnDenoise"
    log = tmp_path / "oidn_calls.log"
    exe.write_text(FAKE_OIDN.format(python=sys.executable, tools=str(REPO / "tools"), log=str(log)))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return exe, log


def test_pfm_round_trip_keeps_orientation(tmp_path):
    img = np.random.default_rng(0).uniform(0, 5, (5, 7, 3)).astype(np.float32)
    img[0, 0] = [100.0, 0.0, 0.0]
    dn.write_pfm(tmp_path / "a.pfm", img)
    assert (tmp_path / "a.pfm").read_bytes().startswith(b"PF\n7 5\n-1.0\n")
    np.testing.assert_array_equal(dn.read_pfm(tmp_path / "a.pfm"), img)


def test_guide_scene_swaps_film_and_spp_and_drops_dispersion_split():
    scene = (
        'Sampler "zsobol" "integer pixelsamples" [1024]\nCamera "realistic" "bool dispersionsplit" true\n'
        'Film "spectral"\n    "string filename" ["x.exr"]\n    "integer nbuckets" [32]\nWorldBegin\n'
    )
    s = dn.guide_scene_text(scene, 40, 30, 16, "g.exr")
    assert 'Film "gbuffer"' in s and '"string filename" ["g.exr"]' in s and '"integer xresolution" [40]' in s
    assert "nbuckets" not in s and 'Film "spectral"' not in s
    assert '"integer pixelsamples" [16]' in s and '"bool dispersionsplit" false' in s
    assert s.rstrip().endswith("WorldBegin")
    with pytest.raises(ValueError):
        dn.guide_scene_text("WorldBegin\n", 4, 4, 1, "g.exr")


def test_stack_denoised_in_triplets_with_padding_and_negative_clip():
    rng = np.random.default_rng(1)
    stack = rng.uniform(0, 1, (6, 8, 5)).astype(np.float32)
    stack[0, 0, 4] = -1.0
    calls = []

    def denoise(trip):
        calls.append(trip.copy())
        return _box3(trip)

    out, neg = dn.denoise_stack(stack, denoise)
    assert len(calls) == 2 and np.all(calls[1][..., 2] == 0)  # 5 planes -> 3 + 2 (zero-padded)
    assert neg == pytest.approx(1 / stack.size)
    np.testing.assert_allclose(out[..., :3], _box3(np.maximum(stack[..., :3], 0)), rtol=1e-6)
    np.testing.assert_allclose(out[..., 3:], _box3(np.maximum(stack, 0)[..., 3:6].copy())[..., :2], rtol=1e-6)


def test_channels_sorted_by_wavelength_and_other_channels_copied():
    from exr_multispectral import parse_s0_wavelength_nm

    rng = np.random.default_rng(2)
    lams = [450.0, 400.0, 550.0, 500.0]
    names = [f"S0.{lam:g}nm" for lam in lams]
    assert all(parse_s0_wavelength_nm(n) == lam for n, lam in zip(names, lams))
    ch = {n: rng.uniform(0, 1, (6, 6)).astype(np.float32) for n in names}
    ch |= {c: rng.uniform(0, 1, (6, 6)).astype(np.float32) for c in "RGBA"}
    seen = []

    def ident(trip):
        seen.append(trip)
        return trip

    out, rep = dn.denoise_channels(ch, ident)
    np.testing.assert_array_equal(seen[0][..., 0], ch["S0.400nm"])  # first triplet = 400, 450, 500 nm
    np.testing.assert_array_equal(seen[0][..., 2], ch["S0.500nm"])
    assert out["A"] is ch["A"]
    assert rep["n_spectral_buckets"] == 4 and rep["spectral_energy_ratio"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        dn.denoise_channels({"Y": ch["A"]}, ident)


def test_denoise_exr_in_place_keeps_noisy_copy_and_marks_sidecar(tmp_path, fake_oidn):
    from exr_multispectral import read_separate_exr_channels
    from synthetic_data import write_spectral_exr

    exe, log = fake_oidn
    lams = np.arange(400.0, 701.0, 50.0)
    planes = np.random.default_rng(3).uniform(0, 1, (lams.size, 12, 16)).astype(np.float32)
    exr = write_spectral_exr(tmp_path / "r.exr", planes, lams)
    noisy = read_separate_exr_channels(exr)
    rep = dn.denoise_exr(exr, exr, guides="none", oidn=exe)
    assert dn.noisy_copy_path(exr).is_file()
    kept = read_separate_exr_channels(dn.noisy_copy_path(exr))
    assert all(np.array_equal(kept[k], noisy[k]) for k in noisy)
    out = read_separate_exr_channels(exr)
    spectral = [k for k in out if k.startswith("S0.")]
    assert spectral and all(out[k].std() < noisy[k].std() for k in spectral)
    side = json.loads(Path(f"{exr}.denoise.json").read_text())
    assert side["denoised"] is True and "do not use for IQ metrics" in side["warning"]
    assert side == json.loads(json.dumps(rep))
    calls = log.read_text().splitlines()
    assert calls and all("-d cpu --affinity 0" in c and "--alb" not in c for c in calls)


def test_default_output_is_separate_file(tmp_path, fake_oidn):
    from synthetic_data import write_spectral_exr

    exr = write_spectral_exr(tmp_path / "r.exr", np.ones((3, 4, 4), np.float32), np.array([450.0, 550.0, 650.0]))
    before = exr.read_bytes()
    rep = dn.denoise_exr(exr, guides="none", oidn=fake_oidn[0])
    assert exr.read_bytes() == before and rep["noisy_copy"] is None
    assert Path(rep["exr_out"]).name == "r_denoised.exr"


def test_guides_need_scene_and_missing_binary_is_explicit(tmp_path, fake_oidn, monkeypatch):
    from synthetic_data import write_spectral_exr

    exr = write_spectral_exr(tmp_path / "r.exr", np.ones((3, 4, 4), np.float32), np.array([450.0, 550.0, 650.0]))
    with pytest.raises(ValueError, match="needs the pbrt scene"):
        dn.denoise_exr(exr, oidn=fake_oidn[0])
    monkeypatch.setenv("OPENCAM_OIDN", str(tmp_path / "missing"))
    monkeypatch.setattr(dn, "DEFAULT_OIDN", tmp_path / "missing")
    monkeypatch.setattr(dn.shutil, "which", lambda _: None)
    with pytest.raises(FileNotFoundError, match="fetch_oidn.py"):
        dn.find_oidn()


def test_pipeline_runs_denoise_stage_only_when_enabled(tmp_path):
    import run_pipeline
    import yaml
    from synthetic_data import run_tool_main

    base = yaml.safe_load((REPO / "config" / "pipeline.yaml").read_text())
    assert base["render"]["denoise"]["enabled"] is False
    stages = {}
    for enabled in (False, True):
        cfg = json.loads(json.dumps(base))
        cfg["render"]["denoise"]["enabled"] = enabled
        cfg_path = tmp_path / f"p{enabled}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))
        stdout = run_tool_main(run_pipeline.main, ["--config", str(cfg_path), "--dry-run", "--name", f"dn{enabled}"])
        man_path = stdout.split("Wrote run manifest:")[-1].strip().splitlines()[0]
        man = json.loads(Path(man_path).read_text())
        Path(man_path).unlink()
        stages[enabled] = [" ".join(map(str, e.get("cmd", []))) for e in man["commands"]]
    assert not any("denoise_spectral_exr.py" in c for c in stages[False])
    on = stages[True]
    i = next(k for k, c in enumerate(on) if "denoise_spectral_exr.py" in c)
    assert "pbrt" in on[i - 1] and "--exr-out" in on[i]
    later = " ".join(on[i + 1 :])
    assert "lens_ghosts.py" not in " ".join(on[:i]) and "post_psf" not in " ".join(on[:i])
    assert later  # the sensor model still follows


def test_real_oidn_removes_noise_and_keeps_mean(tmp_path):
    try:
        exe = dn.find_oidn()
    except FileNotFoundError:
        pytest.skip("oidnDenoise not installed (tools/fetch_oidn.py)")
    rng = np.random.default_rng(4)
    clean = np.ones((64, 64, 3), np.float32) * np.array([0.2, 0.5, 0.8], np.float32)
    noisy = np.maximum(clean + rng.normal(0, 0.1, clean.shape).astype(np.float32), 0)
    run = dn.OidnRunner(exe, quality="high")
    try:
        out = run(noisy)
    finally:
        run.close()
    assert (out - clean)[8:-8, 8:-8].std() < 0.2 * (noisy - clean).std()
    np.testing.assert_allclose(out.mean((0, 1)), noisy.mean((0, 1)), rtol=0.03)
