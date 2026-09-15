"""Tests for the colour / ISP adapter.

The colour science itself is covered by ``tests/test_colour_science.py`` at the
repo root. These check that the staged pipeline is wired correctly and that the
claims the lecture scenarios make are true: that each stage helps, that
gray-world fails on a chart in the specific way it is supposed to, and that
Malvar's advantage comes from the image statistics rather than from arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from opencam_gui.core import isp_engine as ie
from opencam_gui.topics.isp.scenarios import SCENARIOS

ALL = set(ie.STAGES)


@pytest.fixture(scope="module")
def chart():
    return ie.load_chart(illuminant_id="D65")


def _accuracy(chart, *, enabled=ALL, **kw):
    return ie.colour_accuracy(chart, ie.run_isp(chart, ie.IspConfig(enabled=set(enabled), **kw)))


class TestChart:
    def test_loads_all_twenty_four_patches(self, chart):
        assert chart.camera_rgb.shape == (24, 3)
        assert chart.reference_srgb_linear.shape == (24, 3)
        assert len(chart.names) == 24

    def test_the_raw_scene_keeps_the_illuminant_cast(self, chart):
        """If the scene were normalised per channel it would arrive already white
        balanced, and the white-balance stage would have nothing to teach."""
        neutral = chart.camera_rgb[ie._cs().COLORCHECKER_NEUTRAL_SLICE].mean(axis=0)
        assert not np.allclose(neutral / neutral.mean(), 1.0, atol=0.02)

    def test_changing_the_illuminant_changes_the_raw_response(self):
        a = ie.load_chart(illuminant_id="D65").camera_rgb
        b = ie.load_chart(illuminant_id="A").camera_rgb
        assert not np.allclose(a, b, atol=0.01)

    def test_tungsten_starves_the_blue_channel(self):
        """Illuminant A is overwhelmingly red, and the raw data should show it."""
        d65 = ie.load_chart(illuminant_id="D65").camera_rgb
        tungsten = ie.load_chart(illuminant_id="A").camera_rgb
        assert (tungsten[:, 2] / tungsten[:, 1]).mean() < (d65[:, 2] / d65[:, 1]).mean()

    def test_all_eighteen_illuminants_load(self):
        assert len(ie.list_illuminants()) == 18

    def test_the_chart_image_has_the_standard_layout(self):
        img = ie.chart_image(np.ones((24, 3)))
        assert img.ndim == 3 and img.shape[2] == 3
        assert ie.patch_mask().shape == img.shape[:2]
        # Patches cover most of the frame, but the gaps are real.
        covered = ie.patch_mask().mean()
        assert 0.6 < covered < 0.95


class TestStagedPipeline:
    def test_every_stage_reports_a_preview_and_a_note(self, chart):
        result = ie.run_isp(chart, ie.IspConfig())
        assert [s.id for s in result.stages] == ["scene", *ie.STAGES]
        for stage in result.stages:
            assert stage.image.ndim == 3
            assert stage.note
            assert np.all(np.isfinite(stage.image))

    def test_disabled_stages_are_marked_and_pass_the_image_through(self, chart):
        result = ie.run_isp(chart, ie.IspConfig(enabled={"mosaic"}))
        by_id = {s.id: s for s in result.stages}
        assert by_id["mosaic"].applied
        assert not by_id["ccm"].applied
        assert np.allclose(by_id["ccm"].image, by_id["white_balance"].image)

    def test_the_mosaic_collapses_colour_to_one_channel_per_pixel(self, chart):
        result = ie.run_isp(chart, ie.IspConfig(enabled={"mosaic"}))
        mosaic = {s.id: s for s in result.stages}["mosaic"].image
        assert np.allclose(mosaic[:, :, 0], mosaic[:, :, 1])

    def test_demosaic_restores_three_distinct_channels(self, chart):
        result = ie.run_isp(chart, ie.IspConfig(enabled={"mosaic", "demosaic"}))
        out = {s.id: s for s in result.stages}["demosaic"].image
        assert not np.allclose(out[:, :, 0], out[:, :, 2])

    def test_srgb_encoding_lifts_the_shadows(self, chart):
        with_srgb = ie.run_isp(chart, ie.IspConfig(enabled=ALL))
        without = ie.run_isp(chart, ie.IspConfig(enabled=ALL - {"srgb"}))
        dark = without.final_display < 0.5
        assert with_srgb.final_display[dark].mean() > without.final_display[dark].mean()

    def test_unknown_methods_are_rejected(self, chart):
        with pytest.raises(ValueError, match="demosaic"):
            ie.run_isp(chart, ie.IspConfig(demosaic_method="vng"))
        with pytest.raises(ValueError, match="white balance"):
            ie.run_isp(chart, ie.IspConfig(wb_method="magic"))

    @pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
    def test_every_bayer_phase_gives_the_same_colour_accuracy(self, chart, pattern):
        """The CFA phase is a layout detail, not a colour property."""
        got = _accuracy(chart, bayer_pattern=pattern, wb_method="white_patch")
        base = _accuracy(chart, bayer_pattern="RGGB", wb_method="white_patch")
        assert got.mean_delta_e == pytest.approx(base.mean_delta_e, rel=0.25)


class TestColourAccuracy:
    def test_each_stage_improves_on_the_one_before(self, chart):
        raw = _accuracy(chart, enabled={"mosaic", "demosaic"})
        balanced = _accuracy(chart, enabled=ALL - {"ccm", "srgb"}, wb_method="white_patch")
        corrected = _accuracy(chart, enabled=ALL - {"srgb"}, wb_method="white_patch")
        assert raw.mean_delta_e > balanced.mean_delta_e > corrected.mean_delta_e

    def test_raw_camera_rgb_is_badly_wrong(self, chart):
        assert _accuracy(chart, enabled={"mosaic", "demosaic"}).mean_delta_e > 8.0

    def test_the_full_pipeline_reaches_good_camera_colour(self, chart):
        """Under delta-E 2 is what a well-profiled camera achieves."""
        assert _accuracy(chart, enabled=ALL - {"srgb"},
                         wb_method="white_patch").mean_delta_e < 2.0

    def test_exposure_is_normalised_before_colour_is_judged(self, chart):
        """Halving the exposure is not a colour error, and must not read as one."""
        normal = ie.colour_accuracy(chart, ie.run_isp(chart, ie.IspConfig(enabled=ALL - {"srgb"})))
        dim = ie.colour_accuracy(chart, ie.run_isp(
            chart, ie.IspConfig(enabled=ALL - {"srgb"}, exposure_scale=0.5)))
        assert dim.mean_delta_e == pytest.approx(normal.mean_delta_e, rel=0.05)

    def test_the_worst_patches_are_the_saturated_ones(self, chart):
        """Neutrals are easy; saturated colours are where cameras disagree."""
        accuracy = _accuracy(chart, enabled=ALL - {"srgb"}, wb_method="white_patch")
        neutral_error = accuracy.delta_e_2000[ie._cs().COLORCHECKER_NEUTRAL_SLICE].mean()
        assert accuracy.max_delta_e > 2.0 * neutral_error


class TestWhiteBalance:
    def test_white_patch_neutralises_the_greys(self, chart):
        cast = _accuracy(chart, enabled=ALL - {"ccm", "srgb"},
                         wb_method="white_patch").neutral_cast_rgb
        assert np.allclose(cast, 1.0, atol=0.05), cast

    def test_gray_world_leaves_a_cast_on_a_chart(self, chart):
        """Gray world assumes the scene averages to grey. A ColorChecker does not,
        so it over-corrects towards blue -- a real and well-known failure."""
        cast = _accuracy(chart, enabled=ALL - {"ccm", "srgb"},
                         wb_method="gray_world").neutral_cast_rgb
        assert cast[2] > cast[0] * 1.1

    def test_white_patch_beats_gray_world_on_this_scene(self, chart):
        white = _accuracy(chart, enabled=ALL - {"ccm", "srgb"}, wb_method="white_patch")
        gray = _accuracy(chart, enabled=ALL - {"ccm", "srgb"}, wb_method="gray_world")
        assert white.mean_delta_e < gray.mean_delta_e

    def test_gains_compensate_the_illuminant(self):
        """Under tungsten the blue channel is starved, so it needs the most gain."""
        chart = ie.load_chart(illuminant_id="A")
        gains = ie.run_isp(chart, ie.IspConfig(enabled=ALL, wb_method="white_patch")).wb_gains
        assert gains[2] > gains[0]

    def test_skipping_white_balance_leaves_the_gains_at_unity(self, chart):
        result = ie.run_isp(chart, ie.IspConfig(enabled=ALL - {"white_balance"}))
        assert np.allclose(result.wb_gains, 1.0)


class TestCcm:
    def test_an_identity_matrix_is_reported_when_the_stage_is_off(self, chart):
        result = ie.run_isp(chart, ie.IspConfig(enabled=ALL - {"ccm"}))
        assert np.allclose(result.ccm, np.eye(3))

    def test_the_fitted_matrix_has_a_dominant_diagonal(self, chart):
        """A CCM sharpens the channels' effective responses. It should not be
        reordering them."""
        ccm = ie.run_isp(chart, ie.IspConfig(enabled=ALL, wb_method="white_patch")).ccm
        for i in range(3):
            assert ccm[i, i] > 0.5
            assert ccm[i, i] > abs(ccm[i, (i + 1) % 3])

    def test_the_off_diagonal_terms_are_mostly_negative(self, chart):
        """Camera QE curves overlap more than the CMFs do, so the correction is
        to subtract the neighbouring channels."""
        ccm = ie.run_isp(chart, ie.IspConfig(enabled=ALL, wb_method="white_patch")).ccm
        off = ccm[~np.eye(3, dtype=bool)]
        assert (off < 0).sum() >= 4

    def test_the_ccm_cannot_fix_what_the_luther_condition_forbids(self, chart):
        """Even a perfectly fitted 3x3 leaves residual error, because the camera
        is not a linear transform of the observer."""
        assert chart.luther_error > 0.05
        assert _accuracy(chart, enabled=ALL - {"srgb"}, wb_method="white_patch").max_delta_e > 0.5


class TestDemosaicComparison:
    def test_malvar_beats_bilinear_on_natural_image_statistics(self):
        cmp_ = ie.compare_demosaic()
        assert cmp_.malvar_rmse < cmp_.bilinear_rmse
        assert cmp_.edge_malvar_rmse < cmp_.edge_bilinear_rmse

    def test_the_advantage_is_largest_on_edges(self):
        """Flat areas are easy for both; edges are the whole point."""
        cmp_ = ie.compare_demosaic()
        overall = cmp_.bilinear_rmse / cmp_.malvar_rmse
        on_edges = cmp_.edge_bilinear_rmse / cmp_.edge_malvar_rmse
        assert on_edges > overall

    def test_the_methods_agree_on_a_flat_field(self):
        """With no gradient there is nothing for a gradient correction to do."""
        noise = ie._noise()
        flat = np.full((32, 32, 3), 0.4, dtype=np.float32)
        cfa = noise.bayer_sample_rgb(flat, "RGGB")
        a = np.asarray(noise.bilinear_demosaic(np.asarray(cfa, np.float32), "RGGB"))
        b = np.asarray(noise.malvar_demosaic(np.asarray(cfa, np.float32), "RGGB"))
        assert np.allclose(a[4:-4, 4:-4], b[4:-4, 4:-4], atol=1e-5)

    def test_the_two_methods_differ_most_where_the_image_has_structure(self):
        cmp_ = ie.compare_demosaic()
        edges = ie._edge_mask(ie._zipper_target(96))
        assert cmp_.difference[edges].mean() > 2.0 * cmp_.difference[~edges].mean()

    def test_the_advantage_reverses_without_chroma_smoothness(self):
        """Malvar's gain comes from assuming chroma varies slowly. Break that
        assumption and bilinear wins -- the lesson is about images, not maths."""
        noise = ie._noise()
        rng = np.random.default_rng(0)
        truth = np.clip(np.kron(rng.random((12, 12, 3)), np.ones((8, 8, 1))), 0, 1)
        cfa = noise.bayer_sample_rgb(truth.astype(np.float32), "RGGB")
        interior = (slice(4, -4), slice(4, -4))

        def rmse(fn):
            out = np.asarray(fn(np.asarray(cfa, np.float32), "RGGB"), dtype=np.float64)
            return float(np.sqrt(((out[interior] - truth[interior]) ** 2).mean()))

        assert rmse(noise.malvar_demosaic) > rmse(noise.bilinear_demosaic)


class TestSpectralOverlay:
    def test_returns_the_three_curves_and_their_product(self, chart):
        overlay = ie.spectral_overlay(chart, 21)
        k = chart.wavelength_nm.size
        assert overlay.illuminant.shape == (k,)
        assert overlay.reflectance.shape == (k,)
        assert overlay.qe_rgb.shape == (3, k)
        assert overlay.product_rgb.shape == (3, k)

    def test_the_product_is_the_product(self, chart):
        overlay = ie.spectral_overlay(chart, 5)
        expected = chart.qe_rgb * (overlay.illuminant * overlay.reflectance)[None, :]
        assert np.allclose(overlay.product_rgb, expected)

    def test_the_patch_index_is_clamped_rather_than_wrapping(self, chart):
        assert ie.spectral_overlay(chart, 999).patch_name == chart.names[-1]
        assert ie.spectral_overlay(chart, -5).patch_name == chart.names[0]

    def test_colour_temperature_matches_the_illuminant(self):
        assert ie.spectral_overlay(
            ie.load_chart(illuminant_id="D65")).cct_k == pytest.approx(6504, rel=0.02)
        assert ie.spectral_overlay(
            ie.load_chart(illuminant_id="A")).cct_k == pytest.approx(2856, rel=0.02)


def _chart_for_scenario(sc):
    return ie.load_chart(
        illuminant_id=sc.illuminant_id,
        qe_paths=ie.qe_paths_for_recipe(sc.camera_recipe_id),
    )


class TestScenarios:
    @pytest.mark.parametrize("sid", list(SCENARIOS))
    def test_every_scenario_runs(self, sid):
        sc = SCENARIOS[sid]
        chart = _chart_for_scenario(sc)
        result = ie.run_isp(chart, ie.IspConfig(
            enabled=set(sc.stages), demosaic_method=sc.demosaic_method,
            wb_method=sc.wb_method, bayer_pattern=sc.bayer_pattern))
        assert np.all(np.isfinite(result.final_display))
        assert np.all(np.isfinite(ie.colour_accuracy(chart, result).delta_e_2000))

    def test_scenarios_only_name_real_stages_and_methods(self):
        from opencam_gui.core.catalog import list_camera_recipes

        known_recipes = {r.id for r in list_camera_recipes()}
        for sc in SCENARIOS.values():
            assert set(sc.stages) <= set(ie.STAGES), sc.id
            assert sc.demosaic_method in ie.DEMOSAIC_METHODS, sc.id
            assert sc.wb_method in ie.WB_METHODS, sc.id
            if sc.camera_recipe_id:
                assert sc.camera_recipe_id in known_recipes, sc.id

    def test_the_raw_scenario_really_is_badly_wrong(self):
        sc = SCENARIOS["raw_is_not_a_colour_space"]
        chart = ie.load_chart(illuminant_id=sc.illuminant_id)
        assert _accuracy(chart, enabled=set(sc.stages)).mean_delta_e > 8.0

    def test_the_narrowband_led_is_the_hardest_illuminant(self):
        """Its spectrum has holes, and no downstream stage can fill them."""
        led = _accuracy(ie.load_chart(illuminant_id="LED_RGB1"),
                        enabled=ALL - {"srgb"}, wb_method="white_patch")
        daylight = _accuracy(ie.load_chart(illuminant_id="D65"),
                             enabled=ALL - {"srgb"}, wb_method="white_patch")
        assert led.mean_delta_e > daylight.mean_delta_e


class TestCameraRecipes:
    def test_qe_paths_fall_back_to_defaults_for_an_empty_model(self):
        assert ie.qe_paths_from_model({}) == {
            k: v for k, v in ie.DEFAULT_QE_PATHS.items() if k != "ircf"}

    def test_qe_paths_are_taken_from_the_model_when_present(self):
        model = {"sensor": {"quantum_efficiency": {
            "red_csv": "a.csv", "green_csv": "b.csv", "blue_csv": "c.csv"}}}
        paths = ie.qe_paths_from_model(model)
        assert (paths["red"], paths["green"], paths["blue"]) == ("a.csv", "b.csv", "c.csv")

    def test_rccb_and_cmy_recipes_are_not_the_same_camera_as_bayer(self):
        """The alternative-CFA scenarios only teach if they actually change the QE."""
        bayer = ie.load_chart(qe_paths=ie.qe_paths_for_recipe("nikon_z6"))
        rccb = ie.load_chart(qe_paths=ie.qe_paths_for_recipe("default_rccb"))
        cmy = ie.load_chart(qe_paths=ie.qe_paths_for_recipe("default_cmy"))
        assert not np.allclose(rccb.qe_rgb, bayer.qe_rgb)
        assert not np.allclose(cmy.qe_rgb, bayer.qe_rgb)
        assert not np.allclose(cmy.camera_rgb, bayer.camera_rgb)
        assert rccb.luther_error != pytest.approx(bayer.luther_error, rel=1e-4, abs=1e-6)

    def test_none_recipe_falls_back_to_the_default_curves(self):
        assert ie.qe_paths_for_recipe(None)["red"] == ie.DEFAULT_QE_PATHS["red"]
