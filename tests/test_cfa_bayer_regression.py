"""Regression guard: the legacy 2x2 Bayer / RGB paths of ``apply_emva_noise`` stay bit-identical.

The SHA-256 digests in ``tests/data/bayer_path_golden.json`` were recorded with the code on
``main`` before generic (non-Bayer) CFA support was added. Re-record with
``OPENCAM_RECORD_GOLDEN=1 venv/bin/python -m pytest tests/test_cfa_bayer_regression.py``
only when a deliberate change to the Bayer path is made.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import apply_emva_noise  # noqa: E402  (synthetic_data puts tools/ on sys.path)
import imageio.v3 as iio
import numpy as np
import pytest
from synthetic_data import SPECTRAL_LAMBDAS_NM, run_tool_main, write_gaussian_qe, write_spectral_exr, write_yaml

GOLDEN = Path(__file__).parent / "data" / "bayer_path_golden.json"
H, W = 20, 28

CASES = {
    "rggb_malvar_xtalk_integrate_qe": {
        "enabled": True,
        "pattern": "RGGB",
        "demosaic": "malvar",
        "demosaic_srgb": True,
        "spatial_crosstalk": {"enabled": True, "sigma_pixels": 0.3, "sigma_pixels_r": 0.45},
    },
    "gbrg_bilinear_uniform_xtalk": {
        "enabled": True,
        "pattern": "GBRG",
        "demosaic": "bilinear",
        "spatial_crosstalk": {"enabled": True, "sigma_pixels": 0.3},
    },
    "rgb_no_cfa": {"enabled": False},
}


def _spectral_scene() -> np.ndarray:
    lam = SPECTRAL_LAMBDAS_NM
    yy, xx = np.mgrid[0:H, 0:W]
    a = (xx / (W - 1))[:, :, None]
    b = (yy / (H - 1))[:, :, None]
    s1 = np.exp(-0.5 * ((lam - 470.0) / 35.0) ** 2)[None, None, :]
    s2 = np.exp(-0.5 * ((lam - 610.0) / 45.0) ** 2)[None, None, :]
    return (0.02 + 0.6 * a * s1 + 0.5 * b * s2).astype(np.float32)


def _run_case(tmp: Path, bayer: dict) -> dict[str, str]:
    qe = write_gaussian_qe(tmp)
    exr = write_spectral_exr(tmp / "scene.exr", _spectral_scene(), SPECTRAL_LAMBDAS_NM)
    raw = tmp / "noisy.raw16"
    cfg = {
        "sensor": {
            "pixel_pitch_um": 3.0,
            "f_number": 2.8,
            "integration_time_s": 0.01,
            "fill_factor": 1.0,
            "quantum_efficiency": qe,
        },
        "emva": {
            "overall_system_gain_K_e_per_DN": 0.5,
            "sigma_d_e": 2.0,
            "dsnu_std_e": 0.3,
            "dark_current_e_per_s": 5.0,
            "prnu_std_fraction": 0.01,
            "prnu_std_fraction_r": 0.02,
            "spatial_noise_seed": 11,
            "row_fpn_std_e": 0.4,
        },
        "adc": {"full_well_e": 10000.0, "bit_depth": 12},
        "processing": {
            "linear_exr_mode": "integrate_qe",
            "exposure_scale_e_per_unit": 1.0,
            "preview_white_balance": {"enabled": True},
            "preview_color_correction": {"enabled": True, "method": "lstsq_exr_reference"},
        },
        "bayer": bayer,
        "output": {"linear_rgb_in": str(exr), "raw_out": str(raw)},
    }
    cfg_path = write_yaml(tmp / "noise.yaml", cfg)
    run_tool_main(apply_emva_noise.main, ["--repo-root", str(tmp), "--config", str(cfg_path), "--seed", "3"])
    files = [raw, *sorted((raw.parent / f"{raw.stem}_png").glob("*.png"))]
    out = {}
    for f in files:
        # Hash decoded pixels (not PNG bytes) so zlib/encoder versions cannot cause false alarms.
        arr = np.fromfile(f, dtype=np.uint16) if f.suffix == ".raw16" else np.asarray(iio.imread(f))
        out[f.name] = hashlib.sha256(str(arr.shape).encode() + str(arr.dtype).encode() + arr.tobytes()).hexdigest()
    return out


@pytest.mark.parametrize("case", sorted(CASES))
def test_bayer_and_rgb_paths_bit_identical(case: str) -> None:
    with tempfile.TemporaryDirectory() as d:
        digests = _run_case(Path(d), CASES[case])
    if os.environ.get("OPENCAM_RECORD_GOLDEN") == "1":
        data = json.loads(GOLDEN.read_text()) if GOLDEN.is_file() else {}
        data[case] = digests
        GOLDEN.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        pytest.skip("recorded golden digests")
    assert digests == json.loads(GOLDEN.read_text())[case]
