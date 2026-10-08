"""Dear PyGui desktop app: resolution and MTF.

Turns the PSF from a picture into a number. A slanted edge is blurred by the
pipeline's own PSF functions (or loaded from a real render), then put through
the ISO 12233 chain in ``tools/sfr_analysis.py`` via
``opencam_gui.core.mtf_engine`` to give ESF, LSF, MTF and MTF50 -- with the
theoretical diffraction and pixel-aperture curves drawn alongside so the
measurement can be checked against what physics says it should be.
"""

from __future__ import annotations

import math

import dearpygui.dearpygui as dpg
import numpy as np

from opencam_gui.core import mtf_engine as me
from opencam_gui.core.camera import optics_summary
from opencam_gui.topics.mtf.scenarios import SCENARIOS, get_scenario
from opencam_gui.ui.desktop.base import DemoApp

ROI_SIZE = 192
STAR_SIZE = 256
SOURCE_SYNTHETIC = "Synthetic edge (blurred by the PSF model)"
SOURCE_RENDERED = "Rendered slanted-edge target from out/"


def _gray_rgba_flat(gray: np.ndarray) -> list[float]:
    g = np.clip(np.asarray(gray, dtype=np.float64), 0.0, 1.0)
    rgba = np.ones(g.shape + (4,), dtype=np.float64)
    rgba[..., 0] = g
    rgba[..., 1] = g
    rgba[..., 2] = g
    return rgba.ravel().tolist()


def _fit_to(arr: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize onto a fixed texture, keeping pixels blocky."""
    a = np.asarray(arr, dtype=np.float64)
    ys = np.clip((np.arange(size) * a.shape[0] // size), 0, a.shape[0] - 1)
    xs = np.clip((np.arange(size) * a.shape[1] // size), 0, a.shape[1] - 1)
    return a[np.ix_(ys, xs)]


class MtfApp(DemoApp):
    viewport_title = "Open Cam - Resolution and MTF"
    viewport_width = 1500
    viewport_height = 980
    window_label = "Resolution / MTF"
    control_panel_width = 420
    scenarios = SCENARIOS
    default_banner_title = "Interactive slanted-edge MTF explorer"
    default_banner_body = (
        "The PSF demo showed you the blur. This one measures it: ISO 12233 slanted-edge SFR "
        "turns an edge into ESF, LSF, MTF and a single MTF50 number you can compare across cameras."
    )

    def init_state(self) -> None:
        self.source = SOURCE_SYNTHETIC
        self.mode = "chromatic_gaussian"
        self.f_number = 4.0
        self.pixel_pitch_um = 4.3
        self.sigma_geometric_px = 0.3
        self.edge_angle_deg = 5.0
        self.spokes = 72
        self.downsample = 4
        self.prefilter_sigma_px = 0.0

        self._rendered = me.rendered_edge_candidates()
        self._rendered_labels = [p.name for p in self._rendered]
        self._measured_theme: int | None = None
        self._nyquist_theme: int | None = None

    def get_scenario(self, scenario_id: str):
        return get_scenario(scenario_id)

    def apply_scenario_state(self, sc) -> None:
        self.mode = sc.mode
        self.f_number = sc.f_number
        self.pixel_pitch_um = sc.pixel_pitch_um
        self.sigma_geometric_px = sc.sigma_geometric_px
        self.edge_angle_deg = sc.edge_angle_deg
        self.spokes = sc.spokes
        self.downsample = sc.downsample
        self.prefilter_sigma_px = sc.prefilter_sigma_px

    def apply_recipe_state(self, model: dict) -> None:
        summary = optics_summary(model)
        self.f_number = summary.f_number
        self.pixel_pitch_um = summary.pixel_pitch_um
        self.sigma_geometric_px = summary.sigma_geometric_pixels
        self.mode = summary.post_psf_mode if summary.post_psf_mode in ("airy_disk", "chromatic_gaussian") else self.mode

    # --- controls --------------------------------------------------
    def read_controls(self) -> None:
        self.source = dpg.get_value("source_radio")
        self.mode = dpg.get_value("psf_mode")
        self.f_number = float(dpg.get_value("f_number"))
        self.pixel_pitch_um = float(dpg.get_value("pixel_pitch_um"))
        self.sigma_geometric_px = float(dpg.get_value("sigma_geom"))
        self.edge_angle_deg = float(dpg.get_value("edge_angle"))
        self.spokes = int(dpg.get_value("spokes"))
        self.downsample = int(dpg.get_value("downsample"))
        self.prefilter_sigma_px = float(dpg.get_value("prefilter_sigma"))

    def push_controls(self) -> None:
        dpg.set_value("source_radio", self.source)
        dpg.set_value("psf_mode", self.mode)
        dpg.set_value("f_number", self.f_number)
        dpg.set_value("pixel_pitch_um", self.pixel_pitch_um)
        dpg.set_value("sigma_geom", self.sigma_geometric_px)
        dpg.set_value("edge_angle", self.edge_angle_deg)
        dpg.set_value("spokes", self.spokes)
        dpg.set_value("downsample", self.downsample)
        dpg.set_value("prefilter_sigma", self.prefilter_sigma_px)

    # --- compute + draw ---------------------------------------------
    def _current_roi(self) -> tuple[np.ndarray, str]:
        if self.source == SOURCE_RENDERED and self._rendered:
            label = dpg.get_value("rendered_combo")
            path = next((p for p in self._rendered if p.name == label), self._rendered[0])
            return me.load_rendered_edge(path, ROI_SIZE), f"rendered: {path.name}"
        return (
            me.synthetic_edge(
                size=ROI_SIZE,
                angle_deg=self.edge_angle_deg,
                mode=self.mode,
                f_number=self.f_number,
                pixel_pitch_um=self.pixel_pitch_um,
                sigma_geometric_px=self.sigma_geometric_px,
            ),
            "synthetic edge blurred by the PSF model",
        )

    def refresh(self) -> None:
        self.read_controls()
        roi, source_label = self._current_roi()

        try:
            m = me.measure(roi, self.pixel_pitch_um)
        except ValueError as exc:
            dpg.set_value("status_text", f"Could not measure this ROI: {exc}")
            return

        dpg.set_value("roi_texture", _gray_rgba_flat(_fit_to(m.roi, ROI_SIZE)))
        dpg.set_value("esf_series", [m.esf_position_px.tolist(), m.esf.tolist()])
        dpg.set_axis_limits("esf_x", float(m.esf_position_px[0]), float(m.esf_position_px[-1]))
        dpg.set_value("lsf_series", [m.lsf_position_px.tolist(), m.lsf.tolist()])
        dpg.set_axis_limits("lsf_x", -12.0, 12.0)

        dpg.set_value("mtf_series", [m.frequency_cy_per_px.tolist(), m.mtf.tolist()])

        theory = me.theory_curves(f_number=self.f_number, pixel_pitch_um=self.pixel_pitch_um, max_frequency=1.0)
        dpg.set_value("mtf_diffraction", [theory.frequency_cy_per_px.tolist(), theory.diffraction.tolist()])
        dpg.set_value("mtf_pixel", [theory.frequency_cy_per_px.tolist(), theory.pixel_aperture.tolist()])
        dpg.set_value("mtf_system", [theory.frequency_cy_per_px.tolist(), theory.system.tolist()])
        dpg.set_value("mtf_nyquist", [[me.NYQUIST_CY_PER_PX, me.NYQUIST_CY_PER_PX], [0.0, 1.05]])
        cutoff = theory.diffraction_cutoff_cy_per_px
        dpg.set_value("mtf_cutoff", [[cutoff, cutoff], [0.0, 1.05]])
        if math.isfinite(m.mtf50_cy_per_px):
            dpg.set_value("mtf50_marker", [[m.mtf50_cy_per_px], [0.5]])
        else:
            dpg.set_value("mtf50_marker", [[], []])
        dpg.set_axis_limits("mtf_x", 0.0, 1.0)
        dpg.set_axis_limits("mtf_y", 0.0, 1.05)

        self._draw_aliasing()
        self._draw_status(m, source_label, cutoff)

    def _draw_aliasing(self) -> None:
        preview = me.aliasing_preview(
            size=STAR_SIZE,
            spokes=self.spokes,
            downsample=self.downsample,
            prefilter_sigma_px=self.prefilter_sigma_px,
        )
        dpg.set_value("star_texture", _gray_rgba_flat(preview.reference))
        dpg.set_value(
            "sampled_texture",
            _gray_rgba_flat(_fit_to(preview.sampled, STAR_SIZE)),
        )
        dpg.set_value(
            "aliasing_caption",
            (
                f"{preview.spokes} spokes, point-sampled every {preview.downsample} px. "
                f"Inside r = {preview.nyquist_radius_px:.0f} px (in reference-image units) the "
                f"spokes are past Nyquist and fold back as moire.\n"
                + (
                    f"Prefilter sigma {self.prefilter_sigma_px:.2f} px is suppressing them before "
                    "sampling -- detail is lost, but nothing is invented."
                    if self.prefilter_sigma_px > 0
                    else "No prefilter: every frequency above Nyquist aliases."
                )
            ),
        )

    def _draw_status(self, m: me.MtfMeasurement, source_label: str, cutoff: float) -> None:
        mtf50_px = f"{m.mtf50_cy_per_px:.4f}" if math.isfinite(m.mtf50_cy_per_px) else "not reached"
        mtf50_mm = f"{m.mtf50_cy_per_mm:.1f}" if math.isfinite(m.mtf50_cy_per_mm) else "-"
        limited_by = "diffraction" if cutoff <= me.NYQUIST_CY_PER_PX else "the sensor (Nyquist)"
        dpg.set_value(
            "status_text",
            (
                f"Source: {source_label}\n"
                f"Edge angle {m.angle_deg:.2f} deg   pixel pitch {m.pixel_pitch_um:.2f} um\n"
                f"MTF50 = {mtf50_px} cycles/px = {mtf50_mm} cycles/mm (lp/mm)\n"
                f"MTF10 = {m.mtf10_cy_per_px:.4f} cycles/px   "
                f"MTF at Nyquist = {m.mtf_at_nyquist:.3f}\n"
                f"Diffraction cutoff 1/(lambda N) = {cutoff:.3f} cycles/px "
                f"({cutoff * 1000.0 / m.pixel_pitch_um:.0f} cycles/mm)\n"
                f"Resolution is limited by {limited_by}."
            ),
        )

    # --- layout -----------------------------------------------------
    def register_themes(self) -> None:
        with dpg.theme() as measured, dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Line, (110, 200, 255), category=dpg.mvThemeCat_Plots)
        self._measured_theme = measured

        with dpg.theme() as nyquist, dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Line, (240, 110, 110), category=dpg.mvThemeCat_Plots)
        self._nyquist_theme = nyquist

    def register_textures(self) -> None:
        roi_blank = [0.0, 0.0, 0.0, 1.0] * (ROI_SIZE * ROI_SIZE)
        dpg.add_dynamic_texture(ROI_SIZE, ROI_SIZE, roi_blank, tag="roi_texture")
        star_blank = [0.5, 0.5, 0.5, 1.0] * (STAR_SIZE * STAR_SIZE)
        dpg.add_dynamic_texture(STAR_SIZE, STAR_SIZE, star_blank, tag="star_texture")
        dpg.add_dynamic_texture(STAR_SIZE, STAR_SIZE, list(star_blank), tag="sampled_texture")

    def build_controls(self) -> None:
        dpg.add_text("Edge source")
        dpg.add_radio_button(
            tag="source_radio",
            items=[SOURCE_SYNTHETIC, SOURCE_RENDERED],
            default_value=self.source,
            callback=self.on_control_change,
        )
        dpg.add_combo(
            tag="rendered_combo",
            items=self._rendered_labels or ["(no rendered targets in out/)"],
            default_value=(self._rendered_labels or ["(no rendered targets in out/)"])[0],
            callback=self.on_control_change,
        )
        dpg.add_separator()

        dpg.add_text("Optics under test")
        dpg.add_combo(
            tag="psf_mode",
            label="PSF mode",
            items=["chromatic_gaussian", "airy_disk"],
            default_value=self.mode,
            callback=self.on_control_change,
        )
        dpg.add_slider_float(
            tag="f_number",
            label="f-number (N)",
            default_value=self.f_number,
            min_value=1.0,
            max_value=32.0,
            callback=self.on_control_change,
        )
        dpg.add_slider_float(
            tag="pixel_pitch_um",
            label="Pixel pitch (um)",
            default_value=self.pixel_pitch_um,
            min_value=0.7,
            max_value=9.0,
            callback=self.on_control_change,
        )
        dpg.add_slider_float(
            tag="sigma_geom",
            label="Geometric aberration sigma (px)",
            default_value=self.sigma_geometric_px,
            min_value=0.0,
            max_value=3.0,
            callback=self.on_control_change,
        )

        dpg.add_separator()
        dpg.add_text("Aliasing (Siemens star)")
        dpg.add_slider_int(
            tag="spokes",
            label="Spokes",
            default_value=self.spokes,
            min_value=16,
            max_value=144,
            callback=self.on_control_change,
        )
        dpg.add_slider_int(
            tag="downsample",
            label="Sample every N px",
            default_value=self.downsample,
            min_value=1,
            max_value=8,
            callback=self.on_control_change,
        )
        dpg.add_slider_float(
            tag="prefilter_sigma",
            label="OLPF prefilter sigma (px)",
            default_value=self.prefilter_sigma_px,
            min_value=0.0,
            max_value=3.0,
            callback=self.on_control_change,
        )

        with dpg.group(tag="advanced_controls"):
            dpg.add_separator()
            dpg.add_text("Advanced")
            dpg.add_slider_float(
                tag="edge_angle",
                label="Edge slant (deg)",
                default_value=self.edge_angle_deg,
                min_value=1.0,
                max_value=15.0,
                callback=self.on_control_change,
            )

    def build_content(self) -> None:
        with dpg.tab_bar():
            with dpg.tab(label="Slanted-edge SFR"):
                self._build_sfr_tab()
            with dpg.tab(label="Aliasing past Nyquist"):
                self._build_aliasing_tab()

    def _build_sfr_tab(self) -> None:
        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text("Edge ROI under test")
                dpg.add_image("roi_texture", width=250, height=250)
            with dpg.plot(label="Edge spread function (4x oversampled)", height=250, width=380):
                dpg.add_plot_axis(dpg.mvXAxis, label="distance from edge (px)", tag="esf_x")
                with dpg.plot_axis(dpg.mvYAxis, label="value", tag="esf_y"):
                    dpg.add_line_series([0.0], [0.0], label="ESF", tag="esf_series")
            with dpg.plot(label="Line spread function", height=250, width=380):
                dpg.add_plot_axis(dpg.mvXAxis, label="distance from edge (px)", tag="lsf_x")
                with dpg.plot_axis(dpg.mvYAxis, label="normalised", tag="lsf_y"):
                    dpg.add_line_series([0.0], [0.0], label="LSF", tag="lsf_series")

        with dpg.plot(label="Modulation transfer function", height=420, width=-1):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="spatial frequency (cycles/pixel)", tag="mtf_x")
            with dpg.plot_axis(dpg.mvYAxis, label="modulation", tag="mtf_y"):
                measured = dpg.add_line_series([0.0], [0.0], label="measured (slanted edge)", tag="mtf_series")
                dpg.bind_item_theme(measured, self._measured_theme)
                dpg.add_line_series([0.0], [0.0], label="diffraction-limited theory", tag="mtf_diffraction")
                dpg.add_line_series([0.0], [0.0], label="pixel aperture", tag="mtf_pixel")
                dpg.add_line_series([0.0], [0.0], label="diffraction x pixel", tag="mtf_system")
                nyq = dpg.add_line_series([0.0], [0.0], label="Nyquist (0.5 cy/px)", tag="mtf_nyquist")
                dpg.bind_item_theme(nyq, self._nyquist_theme)
                dpg.add_line_series([0.0], [0.0], label="diffraction cutoff", tag="mtf_cutoff")
                dpg.add_scatter_series([0.0], [0.0], label="MTF50", tag="mtf50_marker")

    def _build_aliasing_tab(self) -> None:
        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text("Siemens star (reference)")
                dpg.add_image("star_texture", width=380, height=380)
            with dpg.group():
                dpg.add_text("Point-sampled on a coarse grid")
                dpg.add_image("sampled_texture", width=380, height=380)
        dpg.add_separator()
        dpg.add_text("", tag="aliasing_caption", wrap=940)


def run_app(scenario_id: str | None = None) -> None:
    MtfApp(scenario_id=scenario_id).run()
