import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import build_cabin_scene as cabin  # noqa: E402
import nir_skin  # noqa: E402


def test_absorbers_and_skin_model_bounded():
    wl = np.arange(400.0, 1101.0, 10.0)
    assert nir_skin.water_mua(np.array([970.0]))[0] > 5 * nir_skin.water_mua(np.array([850.0]))[0]
    assert nir_skin.blood_mua(np.array([420.0]))[0] > 100 * nir_skin.blood_mua(np.array([800.0]))[0]
    r = [nir_skin.skin_reflectance(wl, f) for f in nir_skin.MELANIN.values()]
    assert all(np.all((x > 0) & (x < 1)) for x in r)
    assert all(np.all(a[:10] >= b[:10]) for a, b in zip(r, r[1:]))  # more melanin -> darker (blue)


def test_measured_extension_matches_join_and_has_water_dip():
    tone, fitz = cabin.SKIN["light"]
    d = np.loadtxt(cabin.REPO / "spectra/xrite" / f"{tone}.csv", delimiter=",")
    wl = np.array([600.0, 730.0, 850.0, 940.0, 970.0])
    r = nir_skin.extend_measured(wl, d[:, 0], d[:, 1], f_mel=nir_skin.MELANIN[fitz])
    assert r[1] == pytest.approx(np.interp(730.0, d[:, 0], d[:, 1]))
    assert r[4] < r[3] < r[2]
    assert (
        cabin.skin_spectrum("dark")[np.searchsorted(cabin.WL, 940)]
        < cabin.skin_spectrum("light")[np.searchsorted(cabin.WL, 940)]
    )


def test_builder_writes_radiometric_led_and_gaze_truth(tmp_path):
    cabin.main(["--out-dir", str(tmp_path), "--gaze-yaw-deg", "10", "--glasses"])
    txt = (tmp_path / "cabin.pbrt").read_text()
    m = json.loads((tmp_path / "cabin_manifest.json").read_text())
    assert 'AreaLightSource "diffuse"' in txt and '"bool photometric" false' in txt
    assert '"float lambdamax" [1100]' in txt and '"float eta" [1.376]' in txt
    assert m["led"]["peak_nm"] == 940 and len(m["driver"]["eyes"]) == 2
    for e in m["driver"]["eyes"]:
        d = np.subtract(e["pupil_centre"], e["eye_centre"])
        assert np.linalg.norm(d) == pytest.approx(cabin.EYE_R, rel=1e-3)
        assert d[0] > 0  # yaw +10 deg turns the gaze towards +x
