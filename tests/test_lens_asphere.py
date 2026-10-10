"""Even-asphere surfaces and the US 8,390,940 B2 phone lens against analytic / published values."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import lens_asphere as la
import lens_diffraction_psf as L
import numpy as np
import pytest
from lens_glass import pbrt_camera_lines
from lens_prescription import load_lens_file, paraxial_focal_lengths_mm

REPO = Path(__file__).resolve().parent.parent
PHONE = REPO / "config" / "lenses" / "phone_6p_2.97mm.dat"


def test_zero_asphere_is_the_sphere():
    a, R = la.Asphere(), -7.5
    r = np.linspace(0, 5, 11)
    np.testing.assert_allclose(a.sag(R, r), R - np.sign(R) * np.sqrt(R * R - r * r), atol=1e-12)
    wide = L.LensSystem.from_file(REPO / "config" / "lenses" / "wide_22mm.dat")
    asph = replace(wide, aspheres=tuple(None if r == 0 else la.Asphere() for r in wide.radius))
    rng = np.random.default_rng(0)
    o = np.column_stack([rng.uniform(-3, 3, (200, 2)), np.full(200, -5.0)])
    d = np.column_stack([rng.uniform(-0.15, 0.15, (200, 2)), np.ones(200)])
    t0, t1 = L.trace(wide, o, d), L.trace(asph, o, d)
    assert t0["ok"].sum() > 50
    np.testing.assert_array_equal(t0["ok"], t1["ok"])
    np.testing.assert_allclose(t1["o"][t0["ok"]], t0["o"][t0["ok"]], atol=1e-9)
    np.testing.assert_allclose(t1["d"][t0["ok"]], t0["d"][t0["ok"]], atol=1e-9)


def test_newton_hits_the_surface_and_slope_matches_sag():
    a, R = la.Asphere(-2.0, (-9.64344e-2, -3.95194e-1, 1.29192e-1, -2.86778e-2, 0, 0, 0)), 22.9253
    r = np.linspace(0.05, 0.7, 8)
    h = 1e-6
    np.testing.assert_allclose(a.slope(R, r), (a.sag(R, r + h) - a.sag(R, r - h)) / (2 * h), rtol=1e-6)
    o = np.array([[0.1, 0.3, -1.0], [-0.4, 0.2, -1.0]])
    d = np.array([[0.05, -0.1, 1.0], [0.2, 0.0, 1.0]])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    t, n, ok = la.intersect(o, d, 0.0, R, a)
    p = o + t[:, None] * d
    assert ok.all()
    np.testing.assert_allclose(p[:, 2], a.sag(R, np.hypot(p[:, 0], p[:, 1])), atol=1e-12)


def _phone(aspheres=True):
    lens = L.LensSystem.from_file(PHONE)
    return lens if aspheres else replace(lens, aspheres=None)


def _collimated(lens, theta_deg, n=801):
    t = math.radians(theta_deg)
    y = np.linspace(-1.5, 1.5, n)
    o = np.column_stack([np.zeros(n), y, np.full(n, -1.0)])
    d = np.tile([0.0, math.sin(t), math.cos(t)], (n, 1))
    res = L.trace(lens, o, d)
    return res["o"][res["ok"]], res["d"][res["ok"]], res["stop_xy"][res["ok"]]


def _at(o, d, z):
    return o + ((z - o[:, 2]) / d[:, 2])[:, None] * d


def _best_focus(lens):
    o, d, _ = _collimated(lens, 0.0)
    zs = o[:, 2].max() + np.linspace(0.3, 1.5, 1201)
    rms = [np.sqrt(np.mean(_at(o, d, z)[:, 1] ** 2)) for z in zs]
    k = int(np.argmin(rms))
    return zs[k], rms[k]


def test_phone_lens_matches_the_patent():
    rows = load_lens_file(PHONE)
    efl, bfd = paraxial_focal_lengths_mm(rows)
    assert efl == pytest.approx(2.97, abs=0.01)  # Table 1-1: f = 2.97 mm
    assert sum(r[1] for r in rows[1:]) == pytest.approx(3.73, abs=0.005)  # TTL 3.73 mm, IR-cut plate air-equivalent
    lines = pbrt_camera_lines(PHONE)
    assert any('"float asphere"' in s for s in lines) and any('"float dispersion"' in s for s in lines)


def test_aspheres_correct_the_on_axis_spot():
    z, rms = _best_focus(_phone())
    _, rms_sph = _best_focus(_phone(aspheres=False))
    assert rms < 2.0e-3  # mm: below the 1.75 um Airy radius at F/2.45, 588 nm
    assert rms_sph > 5 * rms


def test_distortion_within_the_patent_curve():
    lens = _phone()
    z, _ = _best_focus(lens)
    efl = paraxial_focal_lengths_mm(load_lens_file(PHONE))[0]
    for th in (10.0, 20.0, 30.0, 37.0):
        o, d, sxy = _collimated(lens, th, 4001)
        k = int(np.argmin(np.abs(sxy[:, 1])))  # chief ray: through the stop centre
        h = _at(o[k : k + 1], d[k : k + 1], z)[0, 1]
        dist = 100 * (abs(h) - efl * math.tan(math.radians(th))) / (efl * math.tan(math.radians(th)))
        assert -2.5 < dist < 2.0, (th, dist)  # Fig. 1D: -2.0 % .. +1.5 %
