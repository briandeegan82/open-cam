"""Dear PyGui desktop app: exposure and sensor defects.

The exposure tab runs the photometric chain from scene luminance to electrons
and shows where the result lands: which corner of the exposure triangle, which
branch of the photon transfer curve, and how much headroom ISO gain just spent.

The defects tab drives the pipeline's own defect models against one shared base
frame, one at a time, so each artefact's signature can be recognised on its own
rather than as part of a general mush of noise.
"""

from __future__ import annotations

import numpy as np
import dearpygui.dearpygui as dpg

from opencam_gui.core import exposure_engine as ee
from opencam_gui.core.camera import emva_summary, optics_summary
from opencam_gui.topics.exposure.scenarios import SCENARIOS, get_scenario
from opencam_gui.ui.desktop.base import DemoApp

PREVIEW_SIZE = ee.PREVIEW_SIZE


def _gray_rgba_flat(gray: np.ndarray) -> list[float]:
    g = np.clip(np.asarray(gray, dtype=np.float64), 0.0, 1.0)
    rgba = np.ones(g.shape + (4,), dtype=np.float64)
    rgba[..., 0] = g
    rgba[..., 1] = g
    rgba[..., 2] = g
    return rgba.ravel().tolist()


class ExposureApp(DemoApp):
    viewport_title = "Open Cam - Exposure and sensor defects"
    viewport_width = 1500
    viewport_height = 980
    window_label = "Exposure / defects"
    control_panel_width = 430
    scenarios = SCENARIOS
    default_banner_title = "Exposure, ISO and the defects underneath"
    default_banner_body = (
        "Light, aperture and time decide how many electrons you get; ISO only decides how "
        "loudly they are amplified. Underneath, the sensor adds defects that noise statistics "
        "alone will not describe."
    )

    def init_state(self) -> None:
        self.scene_luminance_cd_m2 = 4000.0
        self.f_number = 16.0
        self.integration_time_s = 1.0 / 125.0
        self.iso_gain = 1.0
        # A full-frame sensor with the well and the converter matched: 65 ke- at
        # 4.1 e-/DN fills a 14-bit range almost exactly. Mismatching them would
        # put the clipping point somewhere the exposure readout cannot explain.
        self.quantum_efficiency = 0.6
        self.pixel_pitch_um = 5.94
        self.K_e_per_DN = 4.10
        self.full_well_e = 65000.0
        self.sigma_d_e = 2.3
        self.sigma_amp_e = 0.3
        self.black_level_DN = 512.0
        self.bit_depth = 14
        self.temperature_c = 20.0

        self.row_fpn_std_e = 12.0
        self.col_fpn_std_e = 12.0
        self.flicker_std_e = 10.0
        self.adc_inl_fraction = 0.02
        self.adc_dnl_std_lsb = 0.6
        self.hot_pixel_fraction = 2e-3
        self.bloom_spread = 0.5
        self.defects: set[str] = set()

        self._current_theme: int | None = None
        self._warn_theme: int | None = None

    def get_scenario(self, scenario_id: str):
        return get_scenario(scenario_id)

    def apply_scenario_state(self, sc) -> None:
        self.scene_luminance_cd_m2 = sc.scene_luminance_cd_m2
        self.f_number = sc.f_number
        self.integration_time_s = sc.integration_time_s
        self.iso_gain = sc.iso_gain
        self.defects = set(sc.defects)
        self.temperature_c = sc.temperature_c
        self.row_fpn_std_e = sc.row_fpn_std_e
        self.col_fpn_std_e = sc.col_fpn_std_e
        self.flicker_std_e = sc.flicker_std_e
        self.adc_inl_fraction = sc.adc_inl_fraction
        self.adc_dnl_std_lsb = sc.adc_dnl_std_lsb
        self.hot_pixel_fraction = sc.hot_pixel_fraction
        self.bloom_spread = sc.bloom_spread

    def apply_recipe_state(self, model: dict) -> None:
        emva = emva_summary(model)
        self.K_e_per_DN = emva.K_e_per_DN
        self.full_well_e = emva.full_well_e
        self.sigma_d_e = emva.sigma_d_e
        self.black_level_DN = emva.black_level_DN
        self.bit_depth = emva.bit_depth
        self.temperature_c = emva.temperature_c
        self.pixel_pitch_um = optics_summary(model).pixel_pitch_um

    # --- controls --------------------------------------------------
    def read_controls(self) -> None:
        self.scene_luminance_cd_m2 = float(dpg.get_value("scene_luminance"))
        self.f_number = float(dpg.get_value("f_number"))
        self.integration_time_s = 1.0 / max(1e-6, float(dpg.get_value("shutter_denom")))
        self.iso_gain = float(dpg.get_value("iso_gain"))
        self.quantum_efficiency = float(dpg.get_value("quantum_efficiency"))
        self.temperature_c = float(dpg.get_value("temperature_c"))
        self.defects = {d for d in ee.DEFECTS if dpg.get_value(f"defect_{d}")}
        self.row_fpn_std_e = float(dpg.get_value("row_fpn_std"))
        self.col_fpn_std_e = float(dpg.get_value("col_fpn_std"))
        self.flicker_std_e = float(dpg.get_value("flicker_std"))
        self.adc_inl_fraction = float(dpg.get_value("adc_inl"))
        self.adc_dnl_std_lsb = float(dpg.get_value("adc_dnl"))
        self.pixel_pitch_um = float(dpg.get_value("pixel_pitch_um"))
        self.sigma_amp_e = float(dpg.get_value("sigma_amp"))

    def push_controls(self) -> None:
        dpg.set_value("scene_luminance", self.scene_luminance_cd_m2)
        dpg.set_value("f_number", self.f_number)
        dpg.set_value("shutter_denom", 1.0 / max(1e-9, self.integration_time_s))
        dpg.set_value("iso_gain", self.iso_gain)
        dpg.set_value("quantum_efficiency", self.quantum_efficiency)
        dpg.set_value("temperature_c", self.temperature_c)
        for d in ee.DEFECTS:
            dpg.set_value(f"defect_{d}", d in self.defects)
        dpg.set_value("row_fpn_std", self.row_fpn_std_e)
        dpg.set_value("col_fpn_std", self.col_fpn_std_e)
        dpg.set_value("flicker_std", self.flicker_std_e)
        dpg.set_value("adc_inl", self.adc_inl_fraction)
        dpg.set_value("adc_dnl", self.adc_dnl_std_lsb)
        dpg.set_value("pixel_pitch_um", self.pixel_pitch_um)
        dpg.set_value("sigma_amp", self.sigma_amp_e)

    # --- compute + draw ---------------------------------------------
    def refresh(self) -> None:
        self.read_controls()
        point = ee.exposure_point(
            scene_luminance_cd_m2=self.scene_luminance_cd_m2,
            f_number=self.f_number,
            integration_time_s=self.integration_time_s,
            iso_gain=self.iso_gain,
            pixel_pitch_um=self.pixel_pitch_um,
            quantum_efficiency=self.quantum_efficiency,
            K_e_per_DN=self.K_e_per_DN,
            full_well_e=self.full_well_e,
            sigma_d_e=self.sigma_d_e,
            sigma_amp_e=self.sigma_amp_e,
            black_level_DN=self.black_level_DN,
            bit_depth=self.bit_depth,
        )
        self._draw_triangle(point)
        self._draw_iso(point)
        self._draw_defects()
        self._draw_status(point)

    def _draw_triangle(self, point: ee.ExposurePoint) -> None:
        tri = ee.exposure_triangle(
            f_number=self.f_number, integration_time_s=self.integration_time_s
        )
        for i, (ev, shutter) in enumerate(tri.ev_lines):
            dpg.set_value(f"ev_line_{i}", [tri.f_number.tolist(), shutter.tolist()])
            dpg.configure_item(f"ev_line_{i}", label=f"EV {ev:.1f}")
        dpg.set_value("ev_current", [[tri.current_f_number], [tri.current_shutter_s]])

        # Electrons along the current EV line: constant by construction, which is
        # the point -- the trade is free as far as the sensor is concerned.
        signal = ee.exposure_point(
            scene_luminance_cd_m2=self.scene_luminance_cd_m2,
            f_number=self.f_number, integration_time_s=self.integration_time_s,
            iso_gain=self.iso_gain, pixel_pitch_um=self.pixel_pitch_um,
            quantum_efficiency=self.quantum_efficiency, K_e_per_DN=self.K_e_per_DN,
            full_well_e=self.full_well_e, sigma_d_e=self.sigma_d_e,
        ).signal_e
        dpg.set_value(
            "triangle_caption",
            (
                f"EV {point.ev:.2f} at ISO gain {point.iso_gain:.0f}x. The scene meters at "
                f"EV100 {point.ev100_scene:.2f}, so this setting is "
                f"{point.ev - point.ev100_scene:+.2f} stops from the metered exposure.\n"
                f"Image-plane illuminance {point.illuminance_lux:.1f} lux for "
                f"{signal:,.0f} signal electrons per pixel. Every point on the highlighted "
                f"line gives the same electron count."
            ),
        )

    def _draw_iso(self, point: ee.ExposurePoint) -> None:
        sweep = ee.iso_sweep(
            signal_e=point.signal_e,
            K_e_per_DN=self.K_e_per_DN,
            full_well_e=self.full_well_e,
            sigma_d_e=self.sigma_d_e,
            sigma_amp_e=self.sigma_amp_e,
        )
        gains = sweep.iso_gain.tolist()
        dpg.set_value("iso_well", [gains, sweep.full_well_e.tolist()])
        dpg.set_value("iso_read", [gains, sweep.read_noise_e.tolist()])
        dpg.set_value("iso_dr", [gains, sweep.dynamic_range_db.tolist()])
        dpg.set_value("iso_snr", [gains, sweep.snr_db.tolist()])
        dpg.set_value("iso_current", [[point.iso_gain], [point.snr_db]])

    def _draw_defects(self) -> None:
        frame = ee.render_defects(
            enabled=tuple(self.defects),
            full_well_e=self.full_well_e,
            K_e_per_DN=self.K_e_per_DN,
            sigma_d_e=self.sigma_d_e,
            bit_depth=self.bit_depth,
            black_level_DN=self.black_level_DN,
            temperature_c=self.temperature_c,
            bloom_spread=self.bloom_spread,
            hot_pixel_fraction=self.hot_pixel_fraction,
            row_fpn_std_e=self.row_fpn_std_e,
            col_fpn_std_e=self.col_fpn_std_e,
            flicker_std_e=self.flicker_std_e,
            adc_inl_fraction=self.adc_inl_fraction,
            adc_dnl_std_lsb=self.adc_dnl_std_lsb,
        )
        dpg.set_value("defect_texture", _gray_rgba_flat(ee.preview_rgb(frame)[..., 0]))

        rows, cols = ee.row_column_profiles(frame)
        idx = np.arange(rows.size).tolist()
        dpg.set_value("row_profile", [idx, rows.tolist()])
        dpg.set_value("col_profile", [np.arange(cols.size).tolist(), cols.tolist()])

        adc = ee.adc_transfer(
            bit_depth=self.bit_depth,
            black_level_DN=self.black_level_DN,
            inl_fraction=self.adc_inl_fraction if "adc_inl" in self.defects else 0.0,
            dnl_std_lsb=self.adc_dnl_std_lsb if "adc_dnl" in self.defects else 0.0,
        )
        dpg.set_value("adc_deviation", [adc.code_in.tolist(), adc.deviation_lsb.tolist()])

        enabled = ", ".join(ee.DEFECT_LABELS[d] for d in frame.enabled) or "none"
        extra = ""
        if frame.sigma_ktc_e > 0:
            extra += f"\nkTC adds {frame.sigma_ktc_e:.1f} e on top of {self.sigma_d_e:.1f} e read noise."
        if frame.hot_pixel_count:
            extra += f"\n{frame.hot_pixel_count} defective pixels in this {PREVIEW_SIZE}x{PREVIEW_SIZE} crop."
        dpg.set_value(
            "defect_caption",
            (
                f"Enabled: {enabled}.\n"
                f"Row profile spread {rows.std():.2f} DN, column profile spread {cols.std():.2f} DN. "
                f"Collapsing a row averages its read noise down by sqrt(width), so banding well "
                f"below the per-pixel noise floor shows up here first.\n"
                f"ADC peak deviation: INL {adc.inl_peak_lsb:.1f} LSB, DNL {adc.dnl_peak_lsb:.2f} LSB."
                + extra
            ),
        )

    def _draw_status(self, point: ee.ExposurePoint) -> None:
        headroom = (
            "clipped" if point.saturation_fraction >= 1.0
            else f"{-np.log2(max(point.saturation_fraction, 1e-9)):.1f} stops of headroom left"
        )
        dpg.set_value(
            "status_text",
            (
                f"EV {point.ev:.2f}   scene EV100 {point.ev100_scene:.2f}\n"
                f"Signal {point.signal_e:,.0f} e   well {point.full_well_e:,.0f} e "
                f"({point.saturation_fraction * 100:.0f}% full, {headroom})\n"
                f"Shot {point.shot_noise_e:.1f} e   read {point.read_noise_e:.1f} e   "
                f"total {point.total_noise_e:.1f} e\n"
                f"SNR {point.snr_db:.1f} dB   mean level {point.mean_dn:.0f} / {point.max_dn:.0f} DN\n"
                f"Regime: {point.regime}."
            ),
        )

    # --- layout -----------------------------------------------------
    def register_themes(self) -> None:
        with dpg.theme() as current:
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, (255, 210, 90), category=dpg.mvThemeCat_Plots)
        self._current_theme = current

        with dpg.theme() as warn:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (240, 110, 110), category=dpg.mvThemeCat_Plots)
        self._warn_theme = warn

    def register_textures(self) -> None:
        blank = [0.2, 0.2, 0.2, 1.0] * (PREVIEW_SIZE * PREVIEW_SIZE)
        dpg.add_dynamic_texture(PREVIEW_SIZE, PREVIEW_SIZE, blank, tag="defect_texture")

    def build_controls(self) -> None:
        dpg.add_text("Exposure")
        dpg.add_slider_float(
            tag="scene_luminance", label="Scene luminance (cd/m2)",
            default_value=self.scene_luminance_cd_m2, min_value=0.1, max_value=8000.0,
            callback=self.on_control_change,
        )
        dpg.add_slider_float(
            tag="f_number", label="f-number (N)", default_value=self.f_number,
            min_value=1.0, max_value=32.0, callback=self.on_control_change,
        )
        dpg.add_slider_float(
            tag="shutter_denom", label="Shutter 1/x (s)",
            default_value=1.0 / self.integration_time_s, min_value=0.5, max_value=4000.0,
            callback=self.on_control_change,
        )
        dpg.add_slider_float(
            tag="iso_gain", label="ISO gain (x)", default_value=self.iso_gain,
            min_value=1.0, max_value=64.0, callback=self.on_control_change,
        )

        dpg.add_separator()
        dpg.add_text("Defects")
        for d in ee.DEFECTS:
            dpg.add_checkbox(
                tag=f"defect_{d}", label=ee.DEFECT_LABELS[d],
                default_value=d in self.defects, callback=self.on_control_change,
            )

        with dpg.group(tag="advanced_controls"):
            dpg.add_separator()
            dpg.add_text("Advanced")
            dpg.add_slider_float(
                tag="quantum_efficiency", label="Quantum efficiency",
                default_value=self.quantum_efficiency, min_value=0.05, max_value=1.0,
                callback=self.on_control_change,
            )
            dpg.add_slider_float(
                tag="pixel_pitch_um", label="Pixel pitch (um)",
                default_value=self.pixel_pitch_um, min_value=0.7, max_value=9.0,
                callback=self.on_control_change,
            )
            dpg.add_slider_float(
                tag="sigma_amp", label="Amplifier noise at 1x (e)",
                default_value=self.sigma_amp_e, min_value=0.0, max_value=3.0,
                callback=self.on_control_change,
            )
            dpg.add_slider_float(
                tag="temperature_c", label="Sensor temperature (C)",
                default_value=self.temperature_c, min_value=-20.0, max_value=80.0,
                callback=self.on_control_change,
            )
            dpg.add_slider_float(
                tag="row_fpn_std", label="Row FPN sigma (e)", default_value=self.row_fpn_std_e,
                min_value=0.0, max_value=60.0, callback=self.on_control_change,
            )
            dpg.add_slider_float(
                tag="col_fpn_std", label="Column FPN sigma (e)", default_value=self.col_fpn_std_e,
                min_value=0.0, max_value=60.0, callback=self.on_control_change,
            )
            dpg.add_slider_float(
                tag="flicker_std", label="1/f flicker sigma (e)", default_value=self.flicker_std_e,
                min_value=0.0, max_value=60.0, callback=self.on_control_change,
            )
            dpg.add_slider_float(
                tag="adc_inl", label="ADC INL (fraction of range)",
                default_value=self.adc_inl_fraction, min_value=0.0, max_value=0.10,
                callback=self.on_control_change,
            )
            dpg.add_slider_float(
                tag="adc_dnl", label="ADC DNL sigma (LSB)", default_value=self.adc_dnl_std_lsb,
                min_value=0.0, max_value=3.0, callback=self.on_control_change,
            )

    def build_content(self) -> None:
        with dpg.tab_bar():
            with dpg.tab(label="Exposure triangle"):
                self._build_exposure_tab()
            with dpg.tab(label="Sensor defects"):
                self._build_defects_tab()

    def _build_exposure_tab(self) -> None:
        with dpg.plot(label="Iso-exposure lines: aperture against shutter", height=400, width=-1):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="f-number", tag="tri_x", scale=dpg.mvPlotScale_Log10)
            with dpg.plot_axis(dpg.mvYAxis, label="shutter time (s)", tag="tri_y", scale=dpg.mvPlotScale_Log10):
                for i in range(5):
                    dpg.add_line_series([1.0], [1.0], label=f"EV {i}", tag=f"ev_line_{i}")
                marker = dpg.add_scatter_series([1.0], [1.0], label="current setting", tag="ev_current")
                dpg.bind_item_theme(marker, self._current_theme)
        dpg.add_text("", tag="triangle_caption", wrap=940)

        dpg.add_separator()
        with dpg.group(horizontal=True):
            with dpg.plot(label="ISO gain: headroom and noise floor", height=320, width=480):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="ISO gain (x)", tag="iso_x", scale=dpg.mvPlotScale_Log10)
                with dpg.plot_axis(dpg.mvYAxis, label="electrons", tag="iso_y", scale=dpg.mvPlotScale_Log10):
                    dpg.add_line_series([1.0], [1.0], label="effective full well", tag="iso_well")
                    read = dpg.add_line_series([1.0], [1.0], label="read noise", tag="iso_read")
                    dpg.bind_item_theme(read, self._warn_theme)
            with dpg.plot(label="Dynamic range and SNR against ISO", height=320, width=480):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="ISO gain (x)", tag="isodb_x", scale=dpg.mvPlotScale_Log10)
                with dpg.plot_axis(dpg.mvYAxis, label="dB", tag="isodb_y"):
                    dpg.add_line_series([1.0], [1.0], label="dynamic range (dB)", tag="iso_dr")
                    dpg.add_line_series([1.0], [1.0], label="SNR at this signal (dB)", tag="iso_snr")
                    cur = dpg.add_scatter_series([1.0], [1.0], label="current", tag="iso_current")
                    dpg.bind_item_theme(cur, self._current_theme)

    def _build_defects_tab(self) -> None:
        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text("Flat field with one blown highlight")
                dpg.add_image("defect_texture", width=330, height=330)
            with dpg.plot(label="Row profile (median DN per row)", height=330, width=330):
                dpg.add_plot_axis(dpg.mvXAxis, label="row", tag="rowp_x")
                with dpg.plot_axis(dpg.mvYAxis, label="DN", tag="rowp_y"):
                    dpg.add_line_series([0.0], [0.0], label="row", tag="row_profile")
            with dpg.plot(label="Column profile (median DN per column)", height=330, width=330):
                dpg.add_plot_axis(dpg.mvXAxis, label="column", tag="colp_x")
                with dpg.plot_axis(dpg.mvYAxis, label="DN", tag="colp_y"):
                    dpg.add_line_series([0.0], [0.0], label="column", tag="col_profile")

        dpg.add_separator()
        with dpg.plot(label="ADC transfer-curve deviation from the ideal ramp", height=300, width=-1):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="ideal code (DN)", tag="adc_x")
            with dpg.plot_axis(dpg.mvYAxis, label="deviation (LSB)", tag="adc_y"):
                dpg.add_line_series([0.0], [0.0], label="INL + DNL", tag="adc_deviation")
        dpg.add_text("", tag="defect_caption", wrap=940)


def run_app(scenario_id: str | None = None) -> None:
    ExposureApp(scenario_id=scenario_id).run()
