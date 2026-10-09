"""Validation of the traced lens-ghost model (tools/lens_ghosts.py, tools/lens_coatings.py)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import lens_coatings as lc  # noqa: E402
import lens_ghosts as lg  # noqa: E402

WIDE = REPO / "config" / "lenses" / "wide_22mm.dat"
LAMS = np.array([400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0])


# ---------------------------------------------------------------- coatings


@pytest.mark.parametrize("coating", ["uncoated", "mgf2", "qhq"])
@pytest.mark.parametrize("n_sub", [1.5, 1.7, 1.9])
def test_coating_energy_conservation(coating, n_sub):
    c = lc.coating_for_interface(coating, 1.0, n_sub)
    beta = np.linspace(0.0, 0.999, 41)[:, None]
    for n1, n2 in ((1.0, n_sub), (n_sub, 1.0)):
        r, t = lc.interface_rt(LAMS[None, :], beta * n1, n1, n2, c)
        assert np.all(r >= -1e-12) and np.all(r <= 1 + 1e-12)
        np.testing.assert_allclose(r + t, 1.0, atol=1e-10)


def test_uncoated_matches_fresnel_normal_incidence_and_brewster():
    r, _ = lc.stack_rt(550.0, 0.0, 1.0, 1.5)
    assert float(r) == pytest.approx(0.04, abs=1e-12)
    # At Brewster's angle R_p = 0 so unpolarised R = R_s / 2 (Born & Wolf sec. 1.5.3).
    tb = math.atan(1.5)
    tt = math.asin(math.sin(tb) / 1.5)
    rs = (math.sin(tb - tt) / math.sin(tb + tt)) ** 2
    r, _ = lc.stack_rt(550.0, math.sin(tb), 1.0, 1.5)
    assert float(r) == pytest.approx(0.5 * rs, rel=1e-9)
    # Total internal reflection beyond the critical angle.
    r, t = lc.stack_rt(550.0, 1.5 * math.sin(math.radians(45.0)), 1.5, 1.0)
    assert float(r) == pytest.approx(1.0) and float(t) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("n_sub", [1.5, 1.62, 1.8])
def test_quarter_wave_mgf2_analytic(n_sub):
    c = lc.coating_for_interface("mgf2", 1.0, n_sub, design_nm=550.0)
    r, _ = lc.stack_rt(550.0, 0.0, 1.0, n_sub, c.layers)
    n1 = lc.MGF2_N
    expect = ((n_sub - n1**2) / (n_sub + n1**2)) ** 2  # Macleod 2010 ch. 3, single quarter-wave
    assert float(r) == pytest.approx(expect, rel=1e-10)
    # Half-wave thickness is absentee: back to bare Fresnel at lambda = 2 n d.
    r_hw, _ = lc.stack_rt(275.0, 0.0, 1.0, n_sub, c.layers)
    assert float(r_hw) == pytest.approx(((n_sub - 1) / (n_sub + 1)) ** 2, rel=1e-10)


def test_qhq_zero_at_design_and_broadband():
    c = lc.coating_for_interface("qhq", 1.0, 1.52, design_nm=550.0)
    r, _ = lc.stack_rt(LAMS, 0.0, 1.0, 1.52, c.layers)
    assert r[LAMS == 550.0][0] < 1e-12
    mg = lc.coating_for_interface("mgf2", 1.0, 1.52, design_nm=550.0)
    r_mg, _ = lc.stack_rt(LAMS, 0.0, 1.0, 1.52, mg.layers)
    band = (LAMS >= 450) & (LAMS <= 650)
    assert r[band].mean() < r_mg[band].mean() < 0.04


def test_reflectance_table_matches_exact():
    c = lc.coating_for_interface("qhq", 1.0, 1.6)
    tab = lc.reflectance_table(1.6, 1.0, c, tuple(LAMS))
    beta = np.random.default_rng(1).uniform(0, 0.6, 200)
    r_exact, _ = lc.interface_rt(LAMS[None, :], beta[:, None], 1.6, 1.0, c)
    np.testing.assert_allclose(lc.lookup_reflectance(tab, beta), r_exact, atol=2e-5)


# ---------------------------------------------------------------- lens geometry helpers


def _plate_rows(t_mm=5.0, n=1.5, ap=20.0):
    # Stop, then a near-flat plate (|R| = 1e9 mm: optical power 1e-9 / mm, negligible here).
    return ((0.0, 2.0, 0.0, ap), (1e9, t_mm, n, ap), (-1e9, 10.0, 0.0, ap))


def _plate(coating="uncoated", **kw):
    lens = lg.load_lens(None, rows=_plate_rows(**kw), coating=coating)
    lens.z_film = 30.0  # afocal: put the film anywhere behind the plate
    return lens


def _singlet_rows():
    return ((60.0, 6.0, 1.6, 24.0), (0.0, 4.0, 0.0, 16.0), (-80.0, 60.0, 0.0, 24.0))


@pytest.mark.parametrize("coating", ["uncoated", "mgf2"])
def test_single_surface_pair_plate_energy_analytic(coating):
    """Plate ghost at normal incidence: T1 R2 R1 T2 relative to the unit-transmission primary."""
    lens = _plate(coating)
    res = lg.ghost_energy_fractions(lens, [0, 0, 1], LAMS, n_grid=64)
    assert list(res["ghosts"]) == [(1, 2)]
    c = lens.surfaces[1].coating
    r_in, t_in = lc.interface_rt(LAMS, 0.0, 1.0, 1.5, c)
    r_out, t_out = lc.interface_rt(LAMS, 0.0, 1.5, 1.0, c)
    r_back, _ = lc.interface_rt(LAMS, 0.0, 1.5, 1.0, c)
    expect = t_in * r_out * r_back * t_out
    np.testing.assert_allclose(res["ghosts"][(1, 2)]["energy"], expect, rtol=1e-6)
    np.testing.assert_allclose(res["primary_transmittance"], t_in * t_out, rtol=1e-6)
    if coating == "uncoated":
        np.testing.assert_allclose(expect, 0.96**2 * 0.04**2, rtol=1e-12)


@pytest.mark.parametrize("theta_deg", [5.0, 15.0, 30.0])
def test_plate_ghost_offset_analytic(theta_deg):
    """Plate ghost is displaced from the primary by 2 t tan(theta') along x on any z-plane."""
    t, n = 5.0, 1.5
    lens = _plate(t_mm=t, n=n)
    th = math.radians(theta_deg)
    o, d, _c, _s = lg.collimated_beam(lens, lg.direction_from_angles(th), 3)
    o, d = o[4:5], d[4:5]  # beam centre ray
    prim = lg.trace(lens, lg.primary_path(lens), o, d)
    gh = lg.trace(lens, lg.ghost_path(lens, 1, 2), o, d)
    tt = math.asin(math.sin(th) / n)
    assert float(gh.film_xy[0, 0] - prim.film_xy[0, 0]) == pytest.approx(2 * t * math.tan(tt), abs=1e-6)
    assert float(gh.dir3[0, 0]) == pytest.approx(math.sin(th), abs=1e-7)


def _meridional_trace(rows, z_film, steps, y0, u_angle):
    """Independent scalar 2-D trace (y, z) used to cross-check the vectorised 3-D tracer."""
    zs, z = [], 0.0
    for r in rows:
        zs.append(z)
        z += r[1]
    eta = [1.0 if r[2] == 0 else r[2] for r in rows]
    p = np.array([y0 - 5.0 * math.tan(u_angle), -5.0])  # 5 mm in front of the front vertex
    v = np.array([math.sin(u_angle), math.cos(u_angle)])
    forward = True
    for k, kind in steps:
        radius, _t, _e, ap = rows[k]
        if radius == 0:
            p = p + (zs[k] - p[1]) / v[1] * v
            if abs(p[0]) > ap / 2:
                return None
            continue
        c = np.array([0.0, zs[k] + radius])
        oc = p - c
        b = oc @ v
        disc = b * b - (oc @ oc - radius * radius)
        if disc < 0:
            return None
        ts = [
            tc
            for tc in (-b - math.sqrt(disc), -b + math.sqrt(disc))
            if tc > 1e-9 and (p[1] + tc * v[1] - c[1]) * radius < 0
        ]
        if not ts:
            return None
        p = p + min(ts) * v
        if abs(p[0]) > ap / 2:
            return None
        nrm = (p - c) / radius
        if nrm @ v > 0:
            nrm = -nrm
        n_b = 1.0 if k == 0 else eta[k - 1]
        n1, n2 = (n_b, eta[k]) if forward else (eta[k], n_b)
        ci = -(nrm @ v)
        if kind == "R":
            v = v + 2 * ci * nrm
            forward = not forward
        else:
            mu = n1 / n2
            k2 = 1 - mu * mu * (1 - ci * ci)
            if k2 < 0:
                return None
            v = mu * v + (mu * ci - math.sqrt(k2)) * nrm
    if v[1] <= 0:
        return None
    return float((p + (z_film - p[1]) / v[1] * v)[0])


@pytest.mark.parametrize("theta_deg", [2.0, 8.0, 16.0, 24.0])
def test_ghost_positions_match_independent_meridional_trace(theta_deg):
    lens = lg.load_lens(WIDE, aperture_diameter_mm=8.0, focus_distance_m=4.0)
    rows = lg.load_lens_file(WIDE)
    rows = tuple((r, t, e, 8.0 if r == 0 else a) for r, t, e, a in rows)
    th = math.radians(theta_deg)
    n_checked = 0
    for y0 in (-3.0, 0.0, 2.5):
        o = np.array([[y0 - 5.0 * math.tan(th), 0.0, -5.0]])
        d = np.array([[math.sin(th), 0.0, math.cos(th)]])
        for pair in lg.ghost_pairs(lens):
            steps = lg.ghost_path(lens, *pair)
            res = lg.trace(lens, steps, o, d)
            ref = _meridional_trace(rows, lens.z_film, steps, y0, th)
            if ref is None:
                assert not res.alive[0]
                continue
            assert res.alive[0]
            assert float(res.film_xy[0, 0]) == pytest.approx(ref, abs=1e-7)
            n_checked += 1
    assert n_checked > 20


@pytest.mark.parametrize("pair", [(0, 1), (0, 2), (2, 3), (1, 3)])
def test_paraxial_ghost_matrix_matches_trace(pair):
    """Small-angle ghost image height vs. the paraxial ghost matrix (Lee & Eisemann 2013)."""
    rows = ((60.0, 6.0, 1.6, 24.0), (-200.0, 2.0, 0.0, 24.0), (0.0, 4.0, 0.0, 16.0), (-80.0, 50.0, 0.0, 24.0))
    rows = (
        (60.0, 6.0, 1.6, 24.0),
        (-200.0, 4.0, 0.0, 24.0),
        (0.0, 4.0, 0.0, 16.0),
        (-80.0, 3.0, 1.5, 24.0),
        (-60.0, 50.0, 0.0, 24.0),
    )
    lens = lg.load_lens(None, rows=rows, coating="uncoated")
    i, j = (lens.refracting[pair[0]], lens.refracting[pair[1]])
    steps = lg.ghost_path(lens, i, j)
    m = lg.paraxial_path_matrix(lens, steps)
    for y0 in (0.05, -0.1):
        for u in (1e-4, -2e-4):
            o = np.array([[y0 - 1e-3 * u, 0.0, -1e-3]])
            d = np.array([[u, 0.0, 1.0]])
            res = lg.trace(lens, steps, o, d)
            if not res.alive[0]:
                continue
            assert float(res.film_xy[0, 0]) == pytest.approx(
                m[0, 0] * y0 + m[0, 1] * u, abs=2e-5 * (1 + abs(m[0, 0]) + abs(m[0, 1]))
            )
    # Primary path: the film sits at pbrt's thick-lens focus, so an axial beam converges (A ~ 0).
    pm = lg.paraxial_path_matrix(lens, lg.primary_path(lens))
    assert abs(pm[0, 0]) < 1e-5


def test_paraxial_primary_efl():
    lens = lg.load_lens(WIDE)
    m = lg.paraxial_path_matrix(lens, lg.primary_path(lens))
    assert -1.0 / m[1, 0] == pytest.approx(lens.efl_mm, rel=2e-3)
    assert lens.efl_mm == pytest.approx(22.0, rel=0.01)


def test_ghost_enumeration_counts():
    lens = lg.load_lens(WIDE)
    k = len(lens.refracting)
    assert len(lg.ghost_pairs(lens)) == k * (k - 1) // 2 == 66
    lens.sensor_reflectance = 0.05
    assert len(lg.ghost_pairs(lens)) == 66 + k


def test_iterated_ghosts_match_explicit_paths():
    lens = lg.load_lens(WIDE, sensor_reflectance=0.05)
    o, d, _c, _s = lg.collimated_beam(lens, lg.direction_from_angles(math.radians(12), 0.4), 24)
    it = dict(lg.iter_ghost_traces(lens, o, d))
    for pair in lg.ghost_pairs(lens):
        ref = lg.trace(lens, lg.ghost_path(lens, *pair), o, d)
        if pair not in it:
            assert not ref.alive.any()
            continue
        np.testing.assert_array_equal(it[pair].alive, ref.alive)
        np.testing.assert_allclose(lg.path_weights(lens, it[pair], LAMS), lg.path_weights(lens, ref, LAMS))


# ---------------------------------------------------------------- energy


@pytest.mark.parametrize("coating", ["uncoated", "mgf2", "qhq"])
def test_lens_energy_budget(coating):
    """Primary transmittance + all two-reflection ghosts cannot exceed the entering flux."""
    lens = lg.load_lens(WIDE, aperture_diameter_mm=8.0, coating=coating)
    for theta in (0.0, 10.0, 25.0):
        res = lg.ghost_energy_fractions(lens, lg.direction_from_angles(math.radians(theta)), LAMS, n_grid=48)
        ghosts = sum(v["energy"] for v in res["ghosts"].values())
        assert np.all(res["primary_transmittance"] + ghosts < 1.0)
        assert np.all(ghosts > 0)
    if coating == "uncoated":
        # 12 air-glass interfaces at ~4-7 % each: primary transmittance well below 0.7.
        assert res["primary_transmittance"].mean() < 0.7


def test_coatings_reduce_ghosts():
    tot = {}
    for coating in ("uncoated", "mgf2", "qhq"):
        lens = lg.load_lens(WIDE, aperture_diameter_mm=8.0, coating=coating)
        res = lg.ghost_energy_fractions(lens, lg.direction_from_angles(math.radians(10.0)), [550.0], n_grid=48)
        tot[coating] = float(sum(v["energy"][0] for v in res["ghosts"].values()))
    assert tot["qhq"] < tot["mgf2"] < tot["uncoated"]


def test_monte_carlo_nonsequential_agrees_with_enumeration():
    """Per-path energies from the enumerated product of R/T agree with a non-sequential MC trace."""
    rows = ((60.0, 6.0, 1.6, 24.0), (0.0, 4.0, 0.0, 16.0), (-80.0, 3.0, 1.5, 24.0), (-60.0, 50.0, 0.0, 24.0))
    lens = lg.load_lens(None, rows=rows, coating="uncoated")
    direction = lg.direction_from_angles(math.radians(6.0))
    lam = 550.0
    mc = lg.monte_carlo_nonsequential(lens, direction, lam, n_rays=60000, seed=3)
    assert sum(mc["budget"].values()) == pytest.approx(1.0)
    # Sequential estimate with rays sampled over the same front-element disc.
    o, d, _c, _s = lg.collimated_beam(lens, direction, 400)
    s0 = lens.surfaces[0]
    on_disc = np.hypot(o[:, 0] + 0, o[:, 1]) >= 0  # placeholder; restrict below at the vertex plane
    p_vertex = o + ((s0.z - o[:, 2]) / d[:, 2])[:, None] * d
    on_disc = p_vertex[:, 0] ** 2 + p_vertex[:, 1] ** 2 <= s0.ap_r**2
    n_disc = int(on_disc.sum())
    hits = mc["hits"]
    prim = lg.trace(lens, lg.primary_path(lens), o[on_disc], d[on_disc])
    e_prim = float(lg.path_weights_exact(lens, prim, [lam]).sum() / n_disc)
    f_prim = sum(1 for h in hits if h[2] == ()) / mc["n_rays"]
    assert f_prim == pytest.approx(e_prim, abs=4 * math.sqrt(e_prim * (1 - e_prim) / mc["n_rays"]) + 2e-3)
    for pair in lg.ghost_pairs(lens):
        res = lg.trace(lens, lg.ghost_path(lens, *pair), o[on_disc], d[on_disc])
        e = float(lg.path_weights_exact(lens, res, [lam]).sum() / n_disc)
        f = sum(1 for h in hits if h[2] == (pair[1], pair[0])) / mc["n_rays"]
        sigma = math.sqrt(max(e, 1e-6) / mc["n_rays"])
        assert f == pytest.approx(e, abs=5 * sigma + 3e-4), pair


def test_rasterize_conserves_energy():
    lens = lg.load_lens(WIDE, aperture_diameter_mm=8.0)
    o, d, _c, _s = lg.collimated_beam(lens, lg.direction_from_angles(math.radians(8.0)), 32)
    mapping = lg.FilmMapping(lens, 320, 240, "realistic")
    checked = 0
    for _pair, res in lg.iter_ghost_traces(lens, o, d):
        if res.alive.sum() < 10:
            continue
        w = lg.path_weights(lens, res, LAMS)
        px, py = mapping.film_to_pixel(res.film_xy[:, 0], res.film_xy[:, 1])
        alive = res.alive
        px, py = np.where(alive, px, 0), np.where(alive, py, 0)
        tot = np.zeros(LAMS.size)
        for _x, _y, ws, _sig in lg.rasterize_ray_grid(px, py, alive, w, 32):
            tot += ws.sum(0)
        np.testing.assert_allclose(tot, w.sum(0), rtol=1e-12)
        checked += 1
    assert checked > 10


def _point_source_cube(h=96, w=128, x=40, y=30, n=7):
    cube = np.full((h, w, n), 1e-4, np.float32)
    cube[y, x, :] = 1000.0
    return cube


def test_render_ghosts_energy_matches_traced_fractions():
    lens = lg.load_lens(WIDE, aperture_diameter_mm=8.0, focus_distance_m=4.0, coating="uncoated")
    cfg = lg.resolve_config({"enabled": True, "pupil_samples": 48, "film_diagonal_mm": 35.0}, {"camera": "realistic"})
    cube = _point_source_cube()
    mask = np.ones(LAMS.size, bool)
    ghost, _t, rep = lg.render_ghosts(cube, list(LAMS), mask, lens, cfg)
    assert rep["sources"]["n_sources"] == 1
    assert np.all(ghost >= 0)
    mapping = lg.FilmMapping(lens, 128, 96, "realistic", 35.0)
    fx, fy = mapping.pixel_to_film(40.0, 30.0)
    ref = lg.ghost_energy_fractions(lens, mapping.direction_for_film(float(fx), float(fy)), LAMS, n_grid=48)
    expect = sum(v["energy"] for v in ref["ghosts"].values()) * 1000.0
    in_frame = ghost.sum((0, 1))
    assert rep["ghost_energy_in_frame"] <= 1.0 + 1e-6
    np.testing.assert_allclose(in_frame / rep["ghost_energy_in_frame"], expect, rtol=2e-3)


def test_film_mapping_roundtrip_and_direction():
    lens = lg.load_lens(WIDE, aperture_diameter_mm=8.0, focus_distance_m=4.0)
    for cam, kw in (("realistic", {"film_diagonal_mm": 35.0}), ("pinhole", {"fov_deg": 60.0})):
        m = lg.FilmMapping(lens, 400, 300, cam, **kw)
        px = np.array([10.0, 200.0, 333.3])
        py = np.array([20.0, 150.0, 280.0])
        fx, fy = m.pixel_to_film(px, py)
        bx, by = m.film_to_pixel(fx, fy)
        np.testing.assert_allclose(bx, px, atol=1e-3)
        np.testing.assert_allclose(by, py, atol=1e-3)
        # A beam from the recovered direction images back onto the same film point.
        d = m.direction_for_film(float(fx[0]), float(fy[0]))
        o, dd, _c, _s = lg.collimated_beam(lens, d, 32)
        res = lg.trace(lens, lg.primary_path(lens), o, dd)
        cen = np.nanmean(res.film_xy[res.alive], 0)
        assert np.hypot(cen[0] - fx[0], cen[1] - fy[0]) < 0.02


def test_iris_blades_shape_ghost():
    """A hexagonal iris gives a hexagonal (not circular) ghost footprint."""
    lens = lg.load_lens(WIDE, aperture_diameter_mm=4.0, iris_blades=6)
    lens_c = lg.load_lens(WIDE, aperture_diameter_mm=4.0)
    o, d, _c, _s = lg.collimated_beam(lens, [0, 0, 1], 300)
    n_hex = int(lg.trace(lens, lg.primary_path(lens), o, d).alive.sum())
    n_circ = int(lg.trace(lens_c, lg.primary_path(lens_c), o, d).alive.sum())
    # Inscribed-in-circle regular hexagon area ratio = 3 sqrt(3) / (2 pi).
    assert n_hex / n_circ == pytest.approx(3 * math.sqrt(3) / (2 * math.pi), rel=0.01)


# ---------------------------------------------------------------- opt-in / regression


def test_disabled_by_default():
    assert lg.DEFAULTS["enabled"] is False
    import yaml

    for p in sorted((REPO / "config" / "lens_models").glob("*.yaml")):
        doc = yaml.safe_load(p.read_text()) or {}
        tg = (doc.get("lens") or {}).get("traced_ghosts") or {}
        if p.name != "research_realistic_wide22_traced_ghosts.yaml":
            assert not tg.get("enabled", False), p


def test_pipeline_dry_run_adds_ghost_stage_only_when_enabled(tmp_path):
    import json

    import run_pipeline
    import yaml
    from synthetic_data import run_tool_main

    base = yaml.safe_load((REPO / "config" / "pipeline.yaml").read_text())
    stages = {}
    for recipe in ("research_realistic_wide22", "research_realistic_wide22_traced_ghosts"):
        cfg = json.loads(json.dumps(base))
        cfg.setdefault("paths", {}).pop("camera_model_name", None)
        cfg["paths"]["camera_model_config"] = f"config/camera_recipes/{recipe}.yaml"
        cfg_path = tmp_path / f"{recipe}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))
        stdout = run_tool_main(run_pipeline.main, ["--config", str(cfg_path), "--dry-run", "--name", recipe])
        man_path = stdout.split("Wrote run manifest:")[-1].strip().splitlines()[0]
        man = json.loads(Path(man_path).read_text())
        Path(man_path).unlink()
        stages[recipe] = [" ".join(map(str, e.get("cmd", []))) for e in man["commands"]]
    assert not any("lens_ghosts.py" in c for c in stages["research_realistic_wide22"])
    assert any("lens_ghosts.py" in c for c in stages["research_realistic_wide22_traced_ghosts"])


def test_cli_disabled_leaves_exr_unchanged(tmp_path):
    from synthetic_data import run_tool_main, spectral_channel_name, write_spectral_exr

    lams = np.arange(400.0, 701.0, 50.0)
    planes = np.random.default_rng(0).uniform(0, 1, (lams.size, 24, 32)).astype(np.float32)
    planes[:, 5, 7] = 500.0
    exr = write_spectral_exr(tmp_path / "in.exr", planes, lams)
    before = exr.read_bytes()
    cfg = tmp_path / "g.yaml"
    cfg.write_text("traced_ghosts:\n  enabled: false\n  lens_file: config/lenses/wide_22mm.dat\n")
    with pytest.raises(SystemExit) as exc:
        run_tool_main(lg.main, ["--exr-in", str(exr), "--config", str(cfg)])
    assert exc.value.code == 0
    assert exr.read_bytes() == before
    # Enabled: ghosts only add light, and the source pixel's channel names are preserved.
    out = tmp_path / "out.exr"
    run_tool_main(
        lg.main,
        ["--exr-in", str(exr), "--exr-out", str(out), "--config", str(cfg), "--enable", "--pupil-samples", "24"],
    )
    from exr_multispectral import read_separate_exr_channels

    a, b = read_separate_exr_channels(exr), read_separate_exr_channels(out)
    assert set(a) == set(b)
    for lam in lams:
        name = spectral_channel_name(lam)
        assert np.all(b[name] >= a[name] - 1e-6)
        assert b[name].sum() > a[name].sum()


def test_parametric_stray_light_unchanged():
    """The parametric model in apply_spectral_psf is kept bit-for-bit (rot90 ghost + veiling glare)."""
    import apply_spectral_psf as asp

    img = np.random.default_rng(2).uniform(0, 1, (16, 20, 3)).astype(np.float32)
    cfg = {
        "enabled": True,
        "veiling_glare_fraction": 0.01,
        "halo_strength": 0.0,
        "ghost_reflections": {"enabled": True, "ghost_strength": 0.02},
    }
    out = asp.apply_stray_light(img.copy(), cfg)
    blended = 0.99 * img + 0.01 * float(img.mean())
    np.testing.assert_allclose(out, blended + 0.02 * np.rot90(blended, k=2), rtol=1e-5, atol=1e-6)
