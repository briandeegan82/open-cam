"""Recipe scorecard: recipe discovery, optics grouping, world-space ROIs and a pbrt smoke run."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import run_recipe_scorecard as sc  # noqa: E402


def test_discovers_every_recipe():
    names = sc.discover_recipes()
    assert names == sorted(p.stem for p in (REPO / "config" / "camera_recipes").glob("*.yaml"))
    assert "default" in names
    with pytest.raises(SystemExit):
        sc.discover_recipes(["no_such_recipe"])


def test_groups_share_optics_and_cover_all_recipes():
    names = sc.discover_recipes()
    groups = sc.group_recipes(names)
    assert sorted(r for g in groups.values() for r in g["recipes"]) == names
    assert len(groups) < len(names)
    assert {g["optics"]["camera"] for g in groups.values()} <= {"perspective", "thinlens", "realistic"}
    hdr_family = {"default", "default_hdr_dcg", "default_hdr_split_pixel", "default_hdr_lofic", "default_hdr_3exp"}
    keys = {k for k, g in groups.items() if hdr_family & set(g["recipes"])}
    assert len(keys) == 1  # same optics -> one render reused by all five sensor variants


def test_realistic_fov_from_focal_length():
    o = {"camera": "realistic", "efl_mm": 50.0}
    fov = sc.vertical_fov(o, 960, 640)
    h = 35.0 * 640 / np.hypot(960, 640)
    assert fov == pytest.approx(2 * np.degrees(np.arctan(h / 100.0)))
    assert sc.vertical_fov({"camera": "perspective", "fov": 45.0}, 960, 640) == 45.0


def test_framing_keeps_target_plane_in_focus():
    args = sc.builder_args("diorama", {"camera": "thinlens", "fov": 20.0, "lens_radius": 0.04}, 360, 240, 4)
    a = dict(zip(args[::2], args[1::2], strict=False))
    d, z = float(a["--cam-dist"]), sc.SCENES["diorama"][3]
    assert float(a["--focus-distance"]) == pytest.approx(d - z, rel=1e-5)
    assert float(a["--thinlens-focal-distance"]) == pytest.approx(d - z, rel=1e-5)
    # narrower FOV -> camera moves back so the cards fill the same fraction of the frame
    assert (d - z) * np.tan(np.radians(10)) == pytest.approx((1.6 - z) * np.tan(np.radians(20)), rel=1e-4)


def test_world_roi_selects_pixels_inside_rectangle():
    yy, xx = np.mgrid[0:50, 0:80].astype(float)
    P = np.stack([xx / 10.0, (50 - yy) / 10.0, np.zeros_like(xx)], axis=-1)
    roi = sc.world_roi(P, [2.0, 1.0, 4.0, 3.0], 0.0, margin=0.0)
    x0, y0, x1, y1 = roi
    assert P[y0, x0, 0] >= 2.0 and P[y1 - 1, x1 - 1, 0] <= 4.0
    assert P[y1 - 1, x0, 1] >= 1.0 and P[y0, x0, 1] <= 3.0
    assert sc.world_roi(P, [2.0, 1.0, 4.0, 3.0], 5.0) is None


def test_guide_sites_pick_green_of_bayer():
    lay = sc.cm.resolve_layout({"layout": "GBRG"})
    e = np.arange(16.0).reshape(4, 4)
    sites, th, tw = sc.guide_sites(e, lay)
    assert (th, tw) == (2, 2)
    assert np.array_equal(sites, e[0::2, 0::2])


@pytest.mark.skipif(
    not sc.PBRT.is_file() and not os.environ.get("OPENCAM_REQUIRE_PBRT"), reason="pbrt binary not built"
)
def test_smoke_run_writes_report(tmp_path):
    rc = sc.main(
        ["--recipes", "default", "--scenes", "skin", "flare", "--xres", "120", "--yres", "80",
         "--pixelsamples", "4", "--out-dir", str(tmp_path)]
    )  # fmt: skip
    assert rc == 0
    data = json.loads((tmp_path / "scorecard.json").read_text())
    row = data["rows"][0]
    assert row["recipe"] == "default" and np.isfinite(row["skin_de00_mean"])
    assert "| default |" in (tmp_path / "scorecard.md").read_text()
    assert (tmp_path / "scorecard.csv").read_text().startswith("recipe,optics,cfa,hdr")


def test_binned_layout_collapses_quad_bayer_to_bayer():
    quad = sc.cm.resolve_layout({"layout": "QUAD_BAYER"})
    lay, b = sc.binned_layout(quad, (320, 480), (640, 960))
    assert b == 2 and lay.tile == (("R", "G"), ("G", "B"))
    assert sc.binned_layout(quad, (640, 960), (640, 960)) == (quad, 1)


def _const_mosaic(layout, values, shape=(24, 24)):
    th, tw = len(layout.tile), len(layout.tile[0])
    e = np.zeros(shape)
    for r in range(th):
        for c in range(tw):
            e[r::th, c::tw] = values[layout.tile[r][c]]
    return e


def test_white_signal_meters_on_brightest_channel():
    rgbw = sc.cm.resolve_layout({"layout": "RGBW"})
    e = _const_mosaic(rgbw, {"R": 1000.0, "G": 1200.0, "W": 3000.0, "B": 1100.0})
    assert sc.white_signal_e(e, rgbw, [4, 4, 20, 20]) == pytest.approx(3000.0, rel=1e-3)
    with pytest.raises(ValueError, match="empty"):
        sc.white_signal_e(e, rgbw, [10, 10, 10, 20])


def test_meter_signal_includes_card_highlights():
    bayer = sc.cm.resolve_layout({"layout": "RGGB"})
    e = _const_mosaic(bayer, {"R": 1000.0, "G": 1000.0, "B": 1000.0}, (48, 48))
    e[24:, 24:] *= 2.0
    assert sc.meter_signal_e(e, bayer, [2, 2, 20, 20]) == pytest.approx(1000.0, rel=1e-3)
    assert sc.meter_signal_e(e, bayer, [2, 2, 20, 20], [[26, 26, 46, 46]]) == pytest.approx(2000.0, rel=1e-3)


def test_despeckle_removes_fireflies_but_keeps_edge():
    crop = np.where(np.arange(40)[None, :] < 20, 0.1, 1.0) * np.ones((40, 1))
    hot = crop.copy()
    hot[5, 30] = hot[17, 25] = 8.0
    np.testing.assert_allclose(sc.despeckle(hot), crop)
    np.testing.assert_array_equal(sc.despeckle(crop), crop)


def test_saturation_e_uses_adc_ceiling_and_first_hdr_transition():
    quad = {"full_well_effective_e": 19200.0, "bit_depth": 10, "black_level_DN": 16.0, "K_effective_e_per_DN": 4.77}
    assert sc.saturation_e(quad) == pytest.approx((1023 - 16) * 4.77)
    dcg = {"full_well_effective_e": 25000.0, "hdr": {"transitions_reference_e": {"hcg": 2520.0, "lcg": 22500.0}}}
    assert sc.saturation_e(dcg) == 2520.0


def test_flare_reports_nan_when_holes_are_missing():
    img = np.ones((90, 90))
    for cy in (15, 45, 75):
        for cx in (15, 45, 75):
            img[cy - 3 : cy + 3, cx - 3 : cx + 3] = 0.001
    full = sc.score_flare(np.repeat(img[..., None], 3, -1))
    assert full["n_holes"] == 9 and 0 < full["veiling_glare_pct"] < 1
    img[:60] = 1.0
    part = sc.score_flare(np.repeat(img[..., None], 3, -1))
    assert part["n_holes"] == 3 and np.isnan(part["veiling_glare_pct"])


def test_texture_buried_in_noise_is_nan_without_warnings():
    rng = np.random.default_rng(0)
    rgb = 0.5 + 0.01 * rng.standard_normal((64, 64, 3))
    noise = 0.2 * rng.standard_normal((64, 64, 3))
    with np.errstate(all="raise"):
        out = sc.score_diorama(rgb, {"dead_leaves": [0, 0, 64, 64]}, noise=noise)
    assert np.isnan(out["texture_acutance"])


def test_mtf_rebound_flags_non_monotonic_mtf():
    f = np.linspace(0, 0.5, 51)
    assert sc.mtf_rebound(f, np.exp(-((f / 0.3) ** 2))) < 1e-9
    dip = 1 - 0.6 * np.sin(np.pi * f / 0.4) ** 2 * (f < 0.4)
    assert sc.mtf_rebound(f, dip) > 0.5


def test_guide_image_picks_brightest_channel():
    lay = sc.cm.resolve_layout({"layout": "RCCB"})
    t = np.array(lay.tile)
    e = np.tile(np.where(t == "C", 20.0, 5.0), (8, 8))
    img = sc.guide_image(e, lay, [0, 0, e.shape[1], e.shape[0]])
    assert img.shape == e.shape and np.allclose(img, 20.0)


def test_add_lens_stray_light_fills_black_hole(tmp_path):
    from exr_multispectral import read_separate_exr_channels, write_separate_channels_exr

    h, w, lams = 40, 60, (450.0, 550.0, 650.0)
    field = np.ones((h, w), np.float32)
    field[15:25, 25:35] = 0.0
    exr = tmp_path / "flare_spectral.exr"
    write_separate_channels_exr(exr, {f"S0.{lam:g}nm": field for lam in lams})
    optics = {"camera": "realistic", "lensfile": "config/lenses/dgauss.50mm.dat", "aperture_diameter_mm": 12.34}
    cfg = {"ghosts": {"cluster_px": 20, "pupil_samples": 8, "max_sources": 1000}, "scatter": {}}
    out, rep = sc.add_lens_stray_light(exr, optics, 2.0, cfg)
    ch = read_separate_exr_channels(str(out))
    assert sorted(ch) == sorted(f"S0.{lam:g}nm" for lam in lams)
    hole = [float(ch[f"S0.{lam:g}nm"][18:22, 28:32].mean()) for lam in lams]
    assert all(v > 0 for v in hole)
    assert hole[0] > hole[2]  # Rayleigh-Rice scatter is bluer
    assert 0 < rep["ghost_to_source_energy"] < 0.05
    assert 0 < rep["scatter_tis_reference"] < 0.01


def test_to_srgb_linear_keeps_clipped_highlights_white():
    lay = sc.cm.resolve_layout({"layout": "GBRG"})
    e = np.tile(np.array([[500.0, 300.0], [250.0, 500.0]]), (8, 8))  # G brightest on the white patch
    e[4:12, 4:12] = 9500.0  # raw-clipped highlight
    ccm = np.eye(3)[[1, 2, 0]]  # channels (G, B, R) -> RGB
    tinted = sc.to_srgb_linear(e, lay, ccm, [0, 0, 4, 4])[7, 7]
    assert tinted[1] < tinted[0] and tinted[1] < tinted[2]  # magenta without a highlight stage
    white = sc.to_srgb_linear(e, lay, ccm, [0, 0, 4, 4], clip_e=9500.0)[7, 7]
    np.testing.assert_allclose(white, white[1])


@pytest.mark.parametrize(
    ("lens_extra", "flag"),
    [
        ({}, None),
        ({"dispersion": False}, "--no-lens-dispersion"),
        ({"dispersion_split": True}, "--lens-dispersion-split"),
    ],
)
def test_recipe_lens_dispersion_options_reach_builder(lens_extra, flag):
    lens = {"camera": "realistic", "realistic_lensfile": "config/lenses/dgauss.50mm.dat"}
    optics = sc.optics_of({"lens": {**lens, "realistic_aperture_diameter_mm": 6.0, **lens_extra}})
    args = sc.builder_args("diorama", optics, 360, 240, 4)
    dispersion_flags = [a for a in args if "lens-dispersion" in a]
    assert dispersion_flags == ([flag] if flag else [])


def test_edge_mtf_noise_floor_grows_with_noise():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import sfr_analysis as sfr_mod

    edge = sfr_mod.synthetic_slanted_edge(96, 96)
    rgb = np.repeat(edge[..., None], 3, -1)
    rng = np.random.default_rng(1)
    floors = []
    for amp in (0.002, 0.02):
        noise = amp * rng.standard_normal(rgb.shape)
        out = sc.score_diorama(rgb + noise, {"slanted_edge": [0, 0, 96, 96]}, noise=noise)
        assert 0.0 <= out["edge_mtf_0p4"] <= 1.2
        floors.append(out["edge_mtf_0p4_noise"])
    assert 0 < floors[0] < floors[1]


def test_roi_band_covers_rois_on_cfa_grid():
    rois = {"slanted_edge": [432, 200, 497, 265], "dead_leaves": [268, 200, 332, 266], "white": [71, 256, 81, 266]}
    band = sc.roi_band(rois, 720, 480)
    assert all(v % sc.ROI_ALIGN == 0 for v in band)
    assert band[0] <= 71 - sc.ROI_MARGIN and band[1] <= 200 - sc.ROI_MARGIN
    assert band[2] >= 497 + sc.ROI_MARGIN and band[3] >= 266 + sc.ROI_MARGIN
    assert sc.roi_band({"slanted_edge": [2, 3, 718, 477]}, 720, 480) == [0, 0, 720, 480]
    assert sc.roi_band({"slanted_edge": None}, 720, 480) is None


def test_roi_band_pastes_crop_plus_caustics(tmp_path, monkeypatch):
    h, w, band = 16, 24, [4, 8, 12, 16]
    split = {"path_scene": "p.pbrt", "caustic_output": "c.exr"}
    (tmp_path / "diorama.json").write_text(json.dumps({"caustic_split": split}))
    sc.write_separate_channels_exr(tmp_path / "f.exr", {"S0.500,0nm": np.ones((h, w), np.float32)})
    sc.write_separate_channels_exr(tmp_path / "c.exr", {"S0.500,0nm": np.full((h, w), 0.25, np.float32)})
    calls = []

    def fake_pbrt(scene_file, *extra):
        calls.append(extra)
        cw = [float(v) for v in extra[extra.index("--cropwindow") + 1].split(",")]
        x0, x1, y0, y1 = (int(np.ceil(c * r)) for c, r in zip(cw, (w, w, h, h)))
        crop = np.full((y1 - y0, x1 - x0), 3.0, np.float32)
        sc.write_separate_channels_exr(tmp_path / extra[extra.index("--outfile") + 1], {"S0.500,0nm": crop})

    monkeypatch.setattr(sc, "_pbrt", fake_pbrt)
    out = sc.render_roi_band(tmp_path, tmp_path / "f.exr", band, 256, 7)
    img = sc.read_separate_exr_channels(out)["S0.500,0nm"]
    x0, y0, x1, y1 = band
    np.testing.assert_allclose(img[y0:y1, x0:x1], 3.25)
    img[y0:y1, x0:x1] = 1.0
    np.testing.assert_allclose(img, 1.0)
    assert calls[0][:4] == ("--seed", "7", "--spp", "256")


def test_failed_recipe_is_an_error_row_in_the_csv_and_exit_code_1(tmp_path, monkeypatch):
    import csv

    def score(recipe, renders, work, seed):
        if recipe == "default":
            raise RuntimeError("boom")
        return {"recipe": recipe, "cfa": "bayer", "hdr": False, "edge_mtf50_cy_px": 0.3}

    monkeypatch.setattr(sc, "render_scene", lambda *a, **k: {})
    monkeypatch.setattr(sc, "score_recipe", score)
    rc = sc.main(["--recipes", "default", "iphone_8", "--scenes", "skin", "--out-dir", str(tmp_path)])
    assert rc == 1
    rows = {r["recipe"]: r for r in csv.DictReader((tmp_path / "scorecard.csv").open())}
    assert rows["default"]["error"] == "RuntimeError: boom"
    assert rows["iphone_8"]["error"] == "" and rows["iphone_8"]["edge_mtf50_cy_px"] == "0.3"
    assert "## Failed recipes" in (tmp_path / "scorecard.md").read_text()


def test_noise_spread_is_half_the_plus_minus_difference_and_nan_on_failure():
    assert sc.noise_spread(lambda a: a.sum(), np.ones(4), np.full(4, 0.25)) == pytest.approx(1.0)

    def fails(a):
        raise ValueError("no edge")

    assert np.isnan(sc.noise_spread(fails, np.ones(4), np.ones(4)))


def test_score_diorama_texture_acutance_matches_ground_truth():
    """The ROI sees the card inset by CARD_ROI_MARGIN; the reference must be that region at full resolution.
    (Zooming the whole card into the ROI read ~0.25 low with ~0.2 seed scatter.)"""
    from iqlab import cpiq
    from iqlab.dead_leaves import dead_leaves
    from scipy import ndimage

    n, sigma = 96, 1.0
    f = np.linspace(0, 0.5, 2001)
    t = np.exp(-2 * np.pi**2 * sigma**2 * f**2) * np.abs(np.sinc(f))
    view = cpiq.viewing_condition(sc.VIEW, 480)
    truth = cpiq.acutance(f, t / t[(f > 0) & (f <= 0.05)].mean(), view)
    for seed in (0, 1):
        sub = dead_leaves(1024, seed=seed)[51:973, 51:973]
        hi = ndimage.zoom(ndimage.gaussian_filter(sub, sigma * sub.shape[0] / n), 10 * n / sub.shape[0], order=1)
        cap = hi.reshape(n, 10, n, 10).mean((1, 3)) * (1 + 0.16 * (0.5 - (np.arange(n) + 0.5) / n))[:, None]
        rgb = np.full((480, 720, 3), cap.mean())
        rgb[100 : 100 + n, 200 : 200 + n] = cap[..., None]
        out = sc.score_diorama(rgb, {"dead_leaves": [200, 100, 200 + n, 100 + n]}, seed=seed)
        assert out["texture_acutance"] == pytest.approx(truth, abs=0.05)


def test_flare_ignores_noise_blobs_in_vignetted_corners():
    # Measured on google_pixel_3: noise fragments the dark corners into small blobs at ~78 % "glare"; the
    # median over 9 holes + 10 blobs read 75.9 % instead of 0.1 %.
    img = np.ones((120, 180))
    for cy in (25, 60, 95):
        for cx in (35, 90, 145):
            img[cy - 5 : cy + 5, cx - 5 : cx + 5] = 0.001
    for cy, cx in (
        (4, 4),
        (4, 12),
        (12, 4),
        (4, 172),
        (12, 172),
        (112, 4),
        (112, 172),
        (104, 172),
        (112, 12),
        (4, 164),
    ):
        img[cy - 2 : cy + 3, cx - 2 : cx + 3] = 0.45
    assert len(sc.flare.black_hole_glare(img)) == 19
    out = sc.score_flare(np.repeat(img[..., None], 3, -1))
    assert out["n_holes"] == 9 and 0 < out["veiling_glare_pct"] < 1
