"""Tests for the exposure / sensor-defect adapter.

The photometry and the defect models live in ``tools``; these tests check that
the adapter drives them correctly and that the teaching claims the lecture
scenarios make actually hold -- reciprocity, ISO as gain rather than
sensitivity, and each defect having a signature that identifies it.
"""

from __future__ import annotations

import numpy as np
import pytest

from opencam_gui.core import exposure_engine as ee
from opencam_gui.topics.exposure.scenarios import SCENARIOS

# A 12-bit sensor whose full well actually fits in its ADC range.
SENSOR = dict(
    K_e_per_DN=45000.0 / 3800.0,
    full_well_e=45000.0,
    sigma_d_e=25.0,
    bit_depth=12,
    black_level_DN=256.0,
)

EXPOSURE = dict(
    pixel_pitch_um=4.3,
    quantum_efficiency=0.6,
    K_e_per_DN=SENSOR["K_e_per_DN"],
    full_well_e=SENSOR["full_well_e"],
    sigma_d_e=SENSOR["sigma_d_e"],
)


def _point(**overrides):
    kwargs = dict(
        scene_luminance_cd_m2=200.0,
        f_number=5.6,
        integration_time_s=1.0 / 250.0,
        **EXPOSURE,
    )
    kwargs.update(overrides)
    return ee.exposure_point(**kwargs)


class TestExposure:
    def test_reciprocity_holds(self):
        """One stop of aperture cancels one stop of shutter exactly."""
        a = _point(f_number=4.0, integration_time_s=1.0 / 500.0)
        b = _point(f_number=2.0, integration_time_s=1.0 / 2000.0)
        assert a.signal_e == pytest.approx(b.signal_e, rel=1e-9)
        assert a.ev == pytest.approx(b.ev, abs=1e-9)

    def test_electrons_scale_with_time_and_inverse_square_of_aperture(self):
        base = _point(f_number=4.0, integration_time_s=0.01)
        assert _point(f_number=4.0, integration_time_s=0.02).signal_e == pytest.approx(2.0 * base.signal_e, rel=1e-9)
        assert _point(f_number=8.0, integration_time_s=0.01).signal_e == pytest.approx(base.signal_e / 4.0, rel=1e-9)

    def test_sunny_16_lands_within_a_third_of_a_stop_of_the_meter(self):
        """The classic rule: ISO 100, f/16, 1/125 in direct sun."""
        p = _point(scene_luminance_cd_m2=4000.0, f_number=16.0, integration_time_s=1.0 / 125.0)
        assert abs(p.ev - p.ev100_scene) < 0.34

    def test_regimes_split_at_the_photon_transfer_crossover(self):
        """Shot noise overtakes read noise at mu_e = sigma_read^2 -- the knee of
        the PTC, and the reason underexposure is so costly."""
        crossover = SENSOR["sigma_d_e"] ** 2
        dim = _point(scene_luminance_cd_m2=0.05, f_number=16.0, integration_time_s=1.0 / 2000.0)
        assert dim.signal_e < crossover
        assert dim.regime == "read-noise limited"

        bright = _point(scene_luminance_cd_m2=300.0, f_number=4.0, integration_time_s=1.0 / 250.0)
        assert crossover < bright.signal_e < bright.full_well_e
        assert bright.regime == "shot-noise limited"

    def test_clipping_is_reported_rather_than_silently_folded_away(self):
        p = _point(scene_luminance_cd_m2=8000.0, f_number=1.4, integration_time_s=1.0 / 60.0)
        assert p.regime == "clipped"
        assert p.saturation_fraction > 1.0
        assert p.mean_dn <= p.max_dn

    def test_snr_of_a_shot_limited_exposure_is_root_n(self):
        p = _point(scene_luminance_cd_m2=300.0, f_number=4.0, integration_time_s=1.0 / 250.0)
        assert p.shot_noise_e == pytest.approx(np.sqrt(p.signal_e), rel=1e-9)
        expected_db = 20.0 * np.log10(p.signal_e / np.sqrt(p.signal_e + p.read_noise_e**2))
        assert p.snr_db == pytest.approx(expected_db, abs=1e-6)


class TestIsoIsGain:
    def test_iso_does_not_change_the_electrons_collected(self):
        base = _point(iso_gain=1.0)
        pushed = _point(iso_gain=8.0)
        assert pushed.signal_e == pytest.approx(base.signal_e, rel=1e-12)

    def test_iso_spends_highlight_headroom_stop_for_stop(self):
        base = _point(iso_gain=1.0)
        pushed = _point(iso_gain=8.0)
        assert pushed.full_well_e == pytest.approx(base.full_well_e / 8.0, rel=1e-9)
        assert pushed.K_e_per_DN == pytest.approx(base.K_e_per_DN / 8.0, rel=1e-9)

    def test_dynamic_range_falls_as_iso_rises(self):
        sweep = ee.iso_sweep(
            signal_e=2000.0, sigma_amp_e=0.4, **{k: SENSOR[k] for k in ("K_e_per_DN", "full_well_e", "sigma_d_e")}
        )
        assert np.all(np.diff(sweep.dynamic_range_db) < 0)

    def test_amplifier_noise_only_bites_above_unity_gain(self):
        sweep = ee.iso_sweep(
            signal_e=2000.0, sigma_amp_e=0.4, **{k: SENSOR[k] for k in ("K_e_per_DN", "full_well_e", "sigma_d_e")}
        )
        assert sweep.read_noise_e[0] == pytest.approx(SENSOR["sigma_d_e"])
        assert np.all(np.diff(sweep.read_noise_e) > 0)

    def test_without_amplifier_noise_the_floor_is_flat(self):
        """Read noise is input-referred: with a noiseless amplifier, gain moves
        signal and noise together and the electron-domain floor does not move."""
        sweep = ee.iso_sweep(
            signal_e=2000.0, sigma_amp_e=0.0, **{k: SENSOR[k] for k in ("K_e_per_DN", "full_well_e", "sigma_d_e")}
        )
        assert np.allclose(sweep.read_noise_e, SENSOR["sigma_d_e"])


class TestExposureTriangle:
    def test_every_point_on_a_line_is_the_same_exposure(self):
        tri = ee.exposure_triangle(f_number=5.6, integration_time_s=1.0 / 250.0)
        signals = [
            ee.exposure_point(
                scene_luminance_cd_m2=200.0, f_number=float(n), integration_time_s=float(t), **EXPOSURE
            ).signal_e
            for n, t in zip(tri.f_number[::12], tri.shutter_s[::12])
        ]
        assert np.allclose(signals, signals[0], rtol=1e-9)

    def test_adjacent_lines_are_one_stop_apart(self):
        tri = ee.exposure_triangle(f_number=5.6, integration_time_s=1.0 / 250.0)
        evs = [ev for ev, _ in tri.ev_lines]
        assert np.allclose(np.diff(evs), 1.0)
        # One EV step is a factor of two in shutter time at fixed aperture.
        assert tri.ev_lines[0][1][0] == pytest.approx(2.0 * tri.ev_lines[1][1][0], rel=1e-9)

    def test_the_current_setting_sits_on_the_centre_line(self):
        tri = ee.exposure_triangle(f_number=5.6, integration_time_s=1.0 / 250.0)
        centre = next(shutter for ev, shutter in tri.ev_lines if ev == pytest.approx(tri.current_ev))
        assert np.interp(5.6, tri.f_number, centre) == pytest.approx(1.0 / 250.0, rel=1e-3)


class TestDefects:
    def test_no_defects_gives_a_flat_field_plus_read_noise(self):
        frame = ee.render_defects(enabled=(), **SENSOR)
        rows, cols = ee.row_column_profiles(frame)
        # Averaging a row knocks read noise down by roughly sqrt(width).
        assert rows.std() < 0.5
        assert cols.std() < 0.5

    def test_an_unknown_defect_is_rejected_rather_than_ignored(self):
        with pytest.raises(ValueError, match="unknown defects"):
            ee.render_defects(enabled=("purple_fringing",), **SENSOR)

    def test_row_fpn_shows_up_in_the_row_profile_and_not_the_column_profile(self):
        rows, cols = ee.row_column_profiles(ee.render_defects(enabled=("row_fpn",), row_fpn_std_e=25.0, **SENSOR))
        base_rows, base_cols = ee.row_column_profiles(ee.render_defects(enabled=(), **SENSOR))
        assert rows.std() > 3.0 * base_rows.std()
        assert cols.std() == pytest.approx(base_cols.std(), rel=0.5)

    def test_column_fpn_shows_up_in_the_column_profile_and_not_the_row_profile(self):
        rows, cols = ee.row_column_profiles(ee.render_defects(enabled=("column_fpn",), col_fpn_std_e=25.0, **SENSOR))
        base_rows, base_cols = ee.row_column_profiles(ee.render_defects(enabled=(), **SENSOR))
        assert cols.std() > 3.0 * base_cols.std()
        assert rows.std() == pytest.approx(base_rows.std(), rel=0.5)

    def test_flicker_bands_rows_because_a_rolling_shutter_reads_rows_in_time(self):
        rows, cols = ee.row_column_profiles(ee.render_defects(enabled=("flicker",), flicker_std_e=25.0, **SENSOR))
        assert rows.std() > cols.std()

    def test_flicker_is_smoother_than_white_row_fpn(self):
        """Both band rows; the 1/f spectrum is what tells them apart."""
        kw = dict(row_fpn_std_e=25.0, flicker_std_e=25.0, **SENSOR)
        white, _ = ee.row_column_profiles(ee.render_defects(enabled=("row_fpn",), **kw))
        pink, _ = ee.row_column_profiles(ee.render_defects(enabled=("flicker",), **kw))
        assert np.abs(np.diff(pink)).mean() < np.abs(np.diff(white)).mean()

    def test_blooming_spreads_a_saturated_highlight_without_exceeding_full_well(self):
        clean = ee.render_defects(enabled=(), **SENSOR)
        bloomed = ee.render_defects(enabled=("blooming",), **SENSOR)
        threshold = 0.9 * clean.max_dn
        assert (bloomed.dn > threshold).sum() > (clean.dn > threshold).sum()
        assert bloomed.dn.max() <= bloomed.max_dn + 1e-6

    def test_hot_pixels_are_sparse_and_fixed(self):
        a = ee.render_defects(enabled=("hot_pixels",), **SENSOR)
        b = ee.render_defects(enabled=("hot_pixels",), **SENSOR)
        assert a.hot_pixel_count > 0
        # Same camera unit, same defect map: the specks do not move, which is
        # exactly what makes dark-frame subtraction work.
        hot_a = a.dn > 0.9 * a.max_dn
        hot_b = b.dn > 0.9 * b.max_dn
        assert np.array_equal(hot_a, hot_b)
        assert hot_a.sum() < 0.02 * hot_a.size

    def test_ktc_raises_the_noise_floor_and_scales_with_temperature(self):
        cold = ee.render_defects(enabled=("ktc",), temperature_c=-20.0, **SENSOR)
        hot = ee.render_defects(enabled=("ktc",), temperature_c=80.0, **SENSOR)
        assert 0 < cold.sigma_ktc_e < hot.sigma_ktc_e
        # sqrt(T) in kelvin, so 100 C of swing is a small effect -- worth seeing.
        assert hot.sigma_ktc_e / cold.sigma_ktc_e == pytest.approx(np.sqrt(353.15 / 253.15), rel=1e-6)

    def test_disabled_defects_leave_the_frame_untouched(self):
        """Toggling one defect must not perturb the others, or the signatures
        would not be comparable."""
        a = ee.render_defects(enabled=(), **SENSOR)
        b = ee.render_defects(enabled=("adc_inl",), **SENSOR)
        assert np.array_equal(a.electrons, b.electrons)


class TestAdcTransfer:
    def test_inl_is_a_bow_that_vanishes_at_both_endpoints(self):
        t = ee.adc_transfer(inl_fraction=0.03, dnl_std_lsb=0.0)
        assert t.deviation_lsb[0] == pytest.approx(0.0, abs=1e-6)
        assert t.deviation_lsb[-1] == pytest.approx(0.0, abs=1e-6)
        assert t.deviation_lsb[t.deviation_lsb.size // 2] == pytest.approx(t.inl_peak_lsb, rel=0.05)

    def test_inl_peaks_at_a_quarter_of_the_nominal_fraction(self):
        """x(1-x) maxes at 1/4, so a 'fraction f' INL bows by f/4 of full scale."""
        t = ee.adc_transfer(bit_depth=12, inl_fraction=0.04, dnl_std_lsb=0.0)
        assert t.inl_peak_lsb == pytest.approx(0.04 * 4095.0 / 4.0, rel=0.02)

    def test_dnl_is_fixed_per_code_so_it_survives_frame_averaging(self):
        a = ee.adc_transfer(inl_fraction=0.0, dnl_std_lsb=1.0)
        b = ee.adc_transfer(inl_fraction=0.0, dnl_std_lsb=1.0)
        assert np.array_equal(a.code_out, b.code_out)
        assert a.dnl_peak_lsb > 0

    def test_an_ideal_converter_has_no_deviation(self):
        t = ee.adc_transfer(inl_fraction=0.0, dnl_std_lsb=0.0)
        assert np.allclose(t.deviation_lsb, 0.0)
        assert t.inl_peak_lsb == 0.0
        assert t.dnl_peak_lsb == 0.0


class TestScenarios:
    @pytest.mark.parametrize("sid", list(SCENARIOS))
    def test_every_scenario_renders(self, sid):
        sc = SCENARIOS[sid]
        point = ee.exposure_point(
            scene_luminance_cd_m2=sc.scene_luminance_cd_m2,
            f_number=sc.f_number,
            integration_time_s=sc.integration_time_s,
            iso_gain=sc.iso_gain,
            **EXPOSURE,
        )
        assert point.signal_e >= 0
        assert point.regime in ("read-noise limited", "shot-noise limited", "clipped")

        frame = ee.render_defects(
            enabled=sc.defects,
            temperature_c=sc.temperature_c,
            row_fpn_std_e=sc.row_fpn_std_e,
            col_fpn_std_e=sc.col_fpn_std_e,
            flicker_std_e=sc.flicker_std_e,
            adc_inl_fraction=sc.adc_inl_fraction,
            adc_dnl_std_lsb=sc.adc_dnl_std_lsb,
            hot_pixel_fraction=sc.hot_pixel_fraction,
            bloom_spread=sc.bloom_spread,
            **SENSOR,
        )
        assert np.all(np.isfinite(frame.dn))

    def test_the_scenarios_named_defects_all_exist(self):
        for sc in SCENARIOS.values():
            assert set(sc.defects) <= set(ee.DEFECTS), sc.id

    def test_the_underexposed_scenario_really_is_read_noise_limited(self):
        sc = SCENARIOS["read_noise_limited"]
        point = ee.exposure_point(
            scene_luminance_cd_m2=sc.scene_luminance_cd_m2,
            f_number=sc.f_number,
            integration_time_s=sc.integration_time_s,
            iso_gain=sc.iso_gain,
            **EXPOSURE,
        )
        assert point.regime == "read-noise limited"

    def test_the_blown_highlight_scenario_really_clips(self):
        sc = SCENARIOS["blown_highlight"]
        point = ee.exposure_point(
            scene_luminance_cd_m2=sc.scene_luminance_cd_m2,
            f_number=sc.f_number,
            integration_time_s=sc.integration_time_s,
            iso_gain=sc.iso_gain,
            **EXPOSURE,
        )
        assert point.regime == "clipped"


class TestAgainstTheNamedRecipe:
    """The tests above use a deliberately noisy 25 e- sensor. Every scenario
    names ``nikon_z6``, though, whose read noise is 2.3 e- -- and a claim that
    only holds on a noisier sensor than the one loaded is not a claim the demo
    can make on screen.
    """

    Z6 = dict(
        pixel_pitch_um=5.94,
        quantum_efficiency=0.6,
        K_e_per_DN=4.0955,
        full_well_e=65000.0,
        sigma_d_e=2.3,
        black_level_DN=512.0,
        bit_depth=14,
    )

    def _point(self, sc):
        return ee.exposure_point(
            scene_luminance_cd_m2=sc.scene_luminance_cd_m2,
            f_number=sc.f_number,
            integration_time_s=sc.integration_time_s,
            iso_gain=sc.iso_gain,
            **self.Z6,
        )

    def test_every_scenario_names_a_recipe(self):
        assert all(sc.camera_recipe_id for sc in SCENARIOS.values())

    def test_the_underexposed_scenario_is_read_noise_limited_here_too(self):
        """At 2.3 e- of read noise the crossover is about five electrons, so this
        scenario has to be far darker than it would need to be on a noisy sensor."""
        point = self._point(SCENARIOS["read_noise_limited"])
        assert point.regime == "read-noise limited"
        assert point.signal_e < self.Z6["sigma_d_e"] ** 2

    def test_one_stop_is_enough_to_leave_the_read_limited_branch(self):
        """Which is the scenario's actual lesson: the branch is narrow."""
        sc = SCENARIOS["read_noise_limited"]
        opened = ee.exposure_point(
            scene_luminance_cd_m2=sc.scene_luminance_cd_m2,
            f_number=sc.f_number / np.sqrt(2),
            integration_time_s=sc.integration_time_s,
            iso_gain=sc.iso_gain,
            **self.Z6,
        )
        assert opened.regime == "shot-noise limited"

    def test_sunny_16_still_meters_correctly(self):
        assert abs(self._point(SCENARIOS["sunny_16"]).ev - self._point(SCENARIOS["sunny_16"]).ev100_scene) < 0.34

    def test_the_blown_highlight_scenario_clips_here_too(self):
        assert self._point(SCENARIOS["blown_highlight"]).regime == "clipped"

    def test_the_well_and_the_converter_are_matched(self):
        """A well that overruns the top code moves the clipping point somewhere
        the exposure readout cannot account for, and a well that falls short of
        it wastes bits. They should coincide to within a code or so."""
        point = self._point(SCENARIOS["sunny_16"])
        assert point.full_well_e / point.K_e_per_DN + self.Z6["black_level_DN"] == pytest.approx(point.max_dn, abs=2.0)
