"""CRA / microlens angular response (tools/pixel_angular_response.py): analytic and regression checks."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import apply_emva_noise
import numpy as np
import pbrt_spectral_exr_to_electrons as pbrt_tool
import pixel_angular_response as par
from lens_prescription import traced_f_number
from sensor_radiometry import integrate_spectral_planes
from synthetic_data import REPO, SPECTRAL_LAMBDAS_NM, write_gaussian_qe, write_spectral_exr

WIDE = REPO / "config/lenses/wide_22mm.dat"
FISHEYE = REPO / "config/lenses/fisheye.10mm.dat"
H, W = 12, 16


def _box_gauss_numeric(delta, spot, sigma, window):
    """(1/s) ∫ over the box of [Φ((w/2 − δ − u)/σ) − Φ((−w/2 − δ − u)/σ)] du by adaptive quadrature."""
    from scipy.integrate import quad
    from scipy.stats import norm

    def inside(u):
        return norm.cdf((0.5 * window - delta - u) / sigma) - norm.cdf((-0.5 * window - delta - u) / sigma)

    return quad(inside, -0.5 * spot, 0.5 * spot, epsabs=1e-12)[0] / spot


def _monte_carlo_response(theta_deg, lam_nm, stack, shift_um=0.0, n=400_000, seed=0):
    """Independent ray model: sample lenslet points, exact Snell into stack and silicon,
    ideal thin lenslet (slope change -u/f), Gaussian diffraction blur, Beer-Lambert depth."""
    rng = np.random.default_rng(seed)
    n_si, alpha = par.silicon_optical_constants(np.array([lam_nm]), REPO)
    n_si, alpha = float(n_si[0]), float(alpha[0])
    d, f, ns, D = stack.stack_height_um, stack.focal_um, stack.stack_index, stack.aperture_um
    a = math.sin(math.radians(theta_deg))
    ts = (a / ns) / math.sqrt(1.0 - (a / ns) ** 2)
    u = rng.uniform(-D / 2, D / 2, n)
    v = rng.uniform(-D / 2, D / 2, n)
    sx, sy = ts - u / f, -v / f
    x = u + d * sx
    y = v + d * sy
    zc = stack.collection_depth_um
    q = rng.uniform(0.0, 1.0, n)
    z = -np.log1p(-q * (-math.expm1(-alpha * zc))) / alpha
    norm = np.sqrt(1.0 + sx * sx + sy * sy)
    ax, ay = ns * sx / norm, ns * sy / norm  # transverse direction cosines × index (conserved)
    bx, by = ax / n_si, ay / n_si
    cz = np.sqrt(1.0 - bx * bx - by * by)
    x = x + z * bx / cz + shift_um
    y = y + z * by / cz
    sigma = par.AIRY_GAUSSIAN_SIGMA_COEFF * lam_nm * 1e-3 / stack.numerical_aperture
    x = x + rng.normal(0.0, sigma, n)
    y = y + rng.normal(0.0, sigma, n)
    hw, p = stack.window_um / 2, stack.pitch_um
    own = np.mean((np.abs(x) <= hw) & (np.abs(y) <= hw))
    right = np.mean((np.abs(x - p) <= hw) & (np.abs(y) <= hw))
    return float(own), float(right)


class TestClosedForm(unittest.TestCase):
    def test_window_fraction_matches_numeric_convolution(self):
        for delta, spot, sigma, window in [(0.0, 1.0, 0.2, 1.2), (0.7, 0.5, 0.3, 1.0), (1.3, 2.0, 0.1, 1.1)]:
            got = float(par.window_fraction_1d(delta, spot, sigma, window))
            self.assertAlmostEqual(got, _box_gauss_numeric(delta, spot, sigma, window), places=9)

    def test_box_overlap_limit(self):
        s, w = 0.4, 1.0
        for delta in np.linspace(0.0, 1.0, 21):
            expect = np.clip(((w + s) / 2 - abs(delta)) / s, 0.0, 1.0)
            self.assertAlmostEqual(float(par.window_fraction_1d(delta, s, 1e-9, w)), float(expect), places=6)

    def test_focused_lenslet_cutoff_angle(self):
        """Ideal focus on the silicon surface: collimated light is collected until the focused point
        leaves the window, θc = asin(n_s · sin(atan(w / 2d)))."""
        st = par.PixelStack(pitch_um=3.0, stack_height_um=3.0, diffraction=False, silicon_penetration=False)
        theta_c = math.degrees(math.asin(st.stack_index * math.sin(math.atan(st.window_um / (2 * st.stack_height_um)))))
        for theta, expect in [(theta_c - 0.3, 1.0), (theta_c + 0.3, 0.0)]:
            a = np.array([math.sin(math.radians(theta))])
            r = par.pixel_response(a, np.zeros(1), np.ones(1), (0.0, 0.0), np.array([550.0]), st, repo=REPO)
            self.assertAlmostEqual(float(r[1, 1, 0]), expect, places=6)

    def test_matched_shift_restores_normal_incidence(self):
        st = par.PixelStack(pitch_um=1.4, silicon_penetration=False)
        lam = np.array([450.0, 550.0, 650.0])
        ref = par.normal_incidence_response(lam, st, repo=REPO)
        a = math.sin(math.radians(25.0))
        tx, _ = par.stack_tangents(np.array([a]), np.zeros(1), st.stack_index)
        shifted = par.pixel_response(
            np.array([a]), np.zeros(1), np.ones(1), (-st.stack_height_um * float(tx[0]), 0.0), lam, st, repo=REPO
        )[1, 1]
        unshifted = par.pixel_response(np.array([a]), np.zeros(1), np.ones(1), (0.0, 0.0), lam, st, repo=REPO)[1, 1]
        np.testing.assert_allclose(shifted, ref, rtol=1e-12)
        self.assertTrue(np.all(unshifted < 0.6 * ref))

    def test_energy_conservation_without_gaps(self):
        st = par.PixelStack(pitch_um=1.4, window_fraction=1.0, diffraction=False)
        a = np.array([math.sin(math.radians(20.0))])
        r = par.pixel_response(a, np.zeros(1), np.ones(1), (0.0, 0.0), np.array([450.0, 650.0]), st, repo=REPO)
        np.testing.assert_allclose(r.sum(axis=(0, 1)), 1.0, atol=1e-9)

    def test_closed_form_matches_monte_carlo_ray_model(self):
        st = par.PixelStack(pitch_um=1.4)
        for theta in (0.0, 10.0, 20.0, 30.0):
            for lam in (450.0, 550.0, 650.0):
                own_mc, right_mc = _monte_carlo_response(theta, lam, st)
                r = par.pixel_response(
                    np.array([math.sin(math.radians(theta))]),
                    np.zeros(1),
                    np.ones(1),
                    (0.0, 0.0),
                    np.array([lam]),
                    st,
                    repo=REPO,
                )
                self.assertAlmostEqual(float(r[1, 1, 0]), own_mc, delta=0.02, msg=f"own θ={theta} λ={lam}")
                self.assertAlmostEqual(float(r[1, 2, 0]), right_mc, delta=0.02, msg=f"xtalk θ={theta} λ={lam}")

    def test_red_walks_off_further_than_blue(self):
        st = par.PixelStack(pitch_um=1.4)
        a = np.array([math.sin(math.radians(20.0))])
        lam = np.array([450.0, 650.0])
        r = par.pixel_response(a, np.zeros(1), np.ones(1), (0.0, 0.0), lam, st, repo=REPO)
        rel = r[1, 1] / par.normal_incidence_response(lam, st, repo=REPO)
        self.assertLess(rel[1], rel[0])
        self.assertGreater(r[1, 2, 1], r[1, 2, 0])


class TestSilicon(unittest.TestCase):
    def test_green_2008_absorption_coefficients(self):
        # Green (2008) Table 1: α(500 nm) = 1.11e4 /cm, α(800 nm) = 8.50e2 /cm.
        _n, alpha = par.silicon_optical_constants(np.array([500.0, 800.0]), REPO)
        np.testing.assert_allclose(alpha * 1e4, [1.11e4, 8.50e2], rtol=0.01)


class TestIncidence(unittest.TestCase):
    def test_traced_on_axis_cone_matches_traced_f_number(self):
        tl = par.TracedLens.from_file(WIDE)
        _a, _b, w = tl.incidence_directions(0.0, 0.0, n_grid=64)
        n = traced_f_number(WIDE)
        self.assertAlmostEqual(float(w.sum()) / (math.pi / (4 * n * n)), 1.0, delta=0.01)

    def test_traced_chief_ray_matches_paraxial_exit_pupil(self):
        for lens in (WIDE, FISHEYE):
            tl = par.TracedLens.from_file(lens)
            xp = par.paraxial_exit_pupil_distance_mm(tl.rows)
            for h in (0.25, 0.5, 1.0):
                self.assertAlmostEqual(tl.chief_ray_angle_deg(h), math.degrees(math.atan(h / xp)), delta=0.01)

    def test_energy_centroid_tracks_chief_ray_near_axis(self):
        tl = par.TracedLens.from_file(WIDE)
        for h in (1.0, 3.0):
            a, b, w = tl.incidence_directions(h, 0.0, n_grid=32)
            cen = math.degrees(math.asin(abs(float(np.average(a, weights=w)))))
            self.assertAlmostEqual(cen, tl.chief_ray_angle_deg(h), delta=0.1)
            self.assertLess(abs(float(np.average(b, weights=w))), 1e-9)

    def test_pinhole_and_table_sources(self):
        lam = np.array([550.0])
        cfg = {"incidence": "pinhole", "fov_deg": 60.0, "field_grid": [3, 3], "microlens_shift": {"mode": "none"}}
        m = par.compute_maps(100, 100, lam, cfg, pitch_um=1.4, f_number=8.0, repo=REPO)
        # Corner of a square frame: tan θ = √2 · tan(fov/2) (pbrt fov spans the shorter axis).
        expect = math.degrees(math.atan(math.hypot(49.5, 49.5) / 50.0 * math.tan(math.radians(30.0))))
        self.assertAlmostEqual(float(m.chief_ray_deg[0, 0]), expect, delta=0.05)
        lens = {"chief_ray_angle_table": [[0.0, 0.0], [1.0, 30.0]]}
        m2 = par.compute_maps(
            100, 100, lam, {**cfg, "incidence": "auto"}, pitch_um=1.4, f_number=8.0, lens_cfg=lens, repo=REPO
        )
        self.assertEqual(m2.meta["incidence"], "table")
        self.assertAlmostEqual(float(m2.chief_ray_deg[0, 0]), 30.0, delta=0.05)
        self.assertAlmostEqual(float(m2.chief_ray_deg[1, 1]), 0.0, delta=1e-9)

    def test_microlens_shift_mismatch_shading(self):
        lam = np.array([450.0, 550.0, 650.0])
        base = {"incidence": "table", "field_grid": [3, 3]}
        lens = {"chief_ray_angle_table": [[0.0, 0.0], [1.0, 25.0]]}
        out = {}
        for mode in ({"mode": "matched"}, {"mode": "none"}, {"mode": "linear", "max_cra_deg": 25.0}):
            m = par.compute_maps(
                64, 64, lam, {**base, "microlens_shift": mode}, pitch_um=1.4, f_number=2.8, lens_cfg=lens, repo=REPO
            )
            out[mode["mode"]] = m.own[0, 0] / m.own[1, 1]
        np.testing.assert_allclose(out["linear"], out["matched"], atol=0.01)
        # A matched shift re-centres the chief ray at the surface; residual loss is the walk-off of
        # deeply absorbed (red) light in silicon, which no surface shift can remove.
        self.assertTrue(np.all(out["matched"] > 0.94))
        self.assertGreater(out["matched"][0], out["matched"][2])
        self.assertTrue(np.all(out["none"] < 0.7))


class TestPipelineHook(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        lam = SPECTRAL_LAMBDAS_NM
        yy, xx = np.mgrid[0:H, 0:W]
        self.L = (
            (1.0 + 0.05 * xx[..., None] + 0.03 * yy[..., None]) * np.linspace(1, 2, lam.size)[None, None, :] * 10
        ).astype(np.float32)
        self.sensor = {
            "pixel_pitch_um": 3.0,
            "f_number": 2.8,
            "integration_time_s": 0.01,
            "fill_factor": 0.9,
            "quantum_efficiency": write_gaussian_qe(self.tmp),
        }
        self.model = {
            "calibration": {"mode": "photon_counting", "irradiance_scale_W_m2nm_per_unit": 1e-3},
            "optics_transmittance_spatial": {"enabled": True, "edge_factor": 0.6, "exponent": 2.0},
        }

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, model, scene=None):
        return pbrt_tool.spectral_radiance_to_electrons(
            self.L, SPECTRAL_LAMBDAS_NM, repo=self.tmp, sensor=self.sensor, model=model, scene=scene
        )

    def test_disabled_is_bit_identical_to_main(self):
        # Golden values produced by origin/main's spectral_radiance_to_electrons on this input.
        for model in (self.model, {**self.model, "pixel_angular_response": {"enabled": False, "fov_deg": 40}}):
            e, meta = self._run(model)
            self.assertEqual(e.dtype, np.float32)
            self.assertFalse(meta["pixel_angular_response"]["enabled"])
            self.assertAlmostEqual(float(e.astype(np.float64).sum()), 14308507.780273438, delta=0.05)
            np.testing.assert_array_equal(e[0, 0], np.float32([14685.423828125, 11754.283203125, 7855.3046875]))
            np.testing.assert_array_equal(e[5, 7], np.float32([36628.671875, 29317.763671875, 19592.85546875]))
            np.testing.assert_array_equal(e[11, 15], np.float32([30545.681640625, 24448.908203125, 16339.0341796875]))

    def test_unit_response_reproduces_plain_integration(self):
        lam = SPECTRAL_LAMBDAS_NM
        weights = np.abs(np.random.default_rng(1).normal(size=(lam.size, 3)))
        m = par.compute_maps(
            W,
            H,
            lam,
            {"incidence": "pinhole", "fov_deg": 40, "field_grid": [3, 3]},
            pitch_um=3.0,
            f_number=2.8,
            repo=REPO,
        )
        m.own[:] = 1.0
        out, _ = par.apply_pixel_angular_response(self.L, lam, weights, {}, pitch_um=3.0, f_number=2.8, maps=m)
        np.testing.assert_allclose(out, integrate_spectral_planes(self.L, weights), rtol=1e-5)

    def test_enabled_produces_radial_colour_shading(self):
        cfg = {"enabled": True, "incidence": "pinhole", "fov_deg": 70.0, "microlens_shift": {"mode": "none"}}
        e_off, _ = self._run(self.model)
        e_on, meta = self._run({**self.model, "pixel_angular_response": cfg})
        self.assertTrue(meta["pixel_angular_response"]["enabled"])
        rel = e_on / e_off
        self.assertLess(rel[0, 0, 1], rel[H // 2, W // 2, 1])
        self.assertLess(rel[0, 0, 0] / rel[0, 0, 1], rel[H // 2, W // 2, 0] / rel[H // 2, W // 2, 1])

    def test_integrate_qe_path_shares_the_model(self):
        lam = SPECTRAL_LAMBDAS_NM
        exr = write_spectral_exr(self.tmp / "s.exr", self.L, lam)
        cfg = {
            "enabled": True,
            "incidence": "pinhole",
            "fov_deg": 70.0,
            "microlens_shift": {"mode": "linear", "max_cra_deg": 10},
        }
        model = {**self.model, "pixel_angular_response": cfg}
        e_direct, _ = self._run(model)
        e_qe = apply_emva_noise.integrate_exr_spectral_qe(
            exr, self.tmp, self.sensor["quantum_efficiency"], self.sensor, model["calibration"], model_cfg=model
        )
        np.testing.assert_allclose(e_qe, e_direct, rtol=1e-5)

    def test_crosstalk_routing_and_sign(self):
        lam = SPECTRAL_LAMBDAS_NM
        L = np.zeros((H, W, lam.size), dtype=np.float32)
        r0, c0 = 2, 12  # upper-right quadrant, R site in RGGB
        L[r0, c0] = 1.0
        weights = np.ones((lam.size, 3))
        cfg = {"incidence": "pinhole", "fov_deg": 90.0, "microlens_shift": {"mode": "none"}, "field_grid": [H, W]}
        cfg["crosstalk"] = {"enabled": True, "cfa_pattern": "RGGB"}
        out, meta = par.apply_pixel_angular_response(L, lam, weights, cfg, pitch_um=1.4, f_number=2.8, repo=REPO)
        m = par.compute_maps(W, H, lam, cfg, pitch_um=1.4, f_number=2.8, repo=REPO)
        nb = m.neighbours[r0, c0]
        # Light at an upper-right pixel tilts outward (+x, +y on film) → spills right (col+1) and up (row-1).
        np.testing.assert_allclose(out[r0, c0 + 1, 1], float(np.sum(nb[1, 2])), rtol=1e-5)
        np.testing.assert_allclose(out[r0 - 1, c0, 1], float(np.sum(nb[2, 1])), rtol=1e-5)
        self.assertGreater(nb[1, 2].sum(), nb[1, 0].sum())
        self.assertEqual(meta["crosstalk"]["cfa_pattern"], "RGGB")


if __name__ == "__main__":
    unittest.main()
