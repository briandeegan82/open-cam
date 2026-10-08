"""Dear PyGui desktop app: fundamental optics / imaging geometry.

Where the PSF demo asks how a point spreads, this one asks where it lands: the
thin-lens conjugate relation, field of view, depth of field, the aperture
trade-off against diffraction, cos^4 falloff and Brown-Conrady distortion. All
of it is computed by ``tools/imaging_geometry.py`` through
``opencam_gui.core.geometry_engine``.
"""

from __future__ import annotations

import math

import numpy as np
import dearpygui.dearpygui as dpg

from opencam_gui.core import geometry_engine as ge
from opencam_gui.core.camera import optics_summary
from opencam_gui.topics.geometry.scenarios import SCENARIOS, get_scenario
from opencam_gui.ui.desktop.base import DemoApp

VIGNETTE_SIZE = 120
GRID_LINES = 9
N_GRID_SERIES = GRID_LINES * 2


def _gray_rgba_flat(gray: np.ndarray) -> list[float]:
    g = np.clip(np.asarray(gray, dtype=np.float64), 0.0, 1.0)
    rgba = np.ones(g.shape + (4,), dtype=np.float64)
    rgba[..., 0] = g
    rgba[..., 1] = g
    rgba[..., 2] = g
    return rgba.ravel().tolist()


def _format_distance(mm: float) -> str:
    if not math.isfinite(mm):
        return "infinity"
    if mm >= 1000.0:
        return f"{mm / 1000.0:.2f} m"
    if mm >= 10.0:
        return f"{mm / 10.0:.1f} cm"
    return f"{mm:.1f} mm"


class GeometryApp(DemoApp):
    viewport_title = "Open Cam - Fundamental Optics / Imaging Geometry"
    viewport_width = 1500
    viewport_height = 980
    window_label = "Fundamental optics"
    control_panel_width = 420
    scenarios = SCENARIOS
    default_banner_title = "Interactive imaging-geometry explorer"
    default_banner_body = (
        "Focal length, aperture and sensor size decide the framing and the depth of field "
        "before any blur or noise enters. Every number comes from tools/imaging_geometry.py."
    )

    def init_state(self) -> None:
        self.focal_length_mm = 50.0
        self.f_number = 5.6
        self.focus_distance_mm = 3000.0
        self.format_id = "full_frame"
        self.pixel_pitch_um = 5.94
        self.object_height_mm = 1700.0
        self.use_pixel_coc = False
        self.distortion_k1 = 0.0
        self.distortion_k2 = 0.0

        self._formats = ge.sensor_formats()
        self._format_id_by_name = {f.name: fid for fid, f in self._formats.items()}
        self._ray_theme: int | None = None
        self._limit_theme: int | None = None

    def get_scenario(self, scenario_id: str):
        return get_scenario(scenario_id)

    def apply_scenario_state(self, sc) -> None:
        self.focal_length_mm = sc.focal_length_mm
        self.f_number = sc.f_number
        self.focus_distance_mm = sc.focus_distance_mm
        self.format_id = sc.format_id
        self.pixel_pitch_um = sc.pixel_pitch_um
        self.object_height_mm = sc.object_height_mm
        self.use_pixel_coc = sc.use_pixel_coc
        self.distortion_k1 = sc.distortion_k1
        self.distortion_k2 = sc.distortion_k2

    def apply_recipe_state(self, model: dict) -> None:
        """A recipe fixes the pixel pitch, aperture and distortion; the focal
        length and sensor format stay under the student's control because the
        recipes do not record a physical sensor size."""
        summary = optics_summary(model)
        self.pixel_pitch_um = summary.pixel_pitch_um
        self.f_number = summary.f_number
        k1, k2, _, _ = ge.distortion_coefficients(model)
        self.distortion_k1 = k1
        self.distortion_k2 = k2

    # --- controls --------------------------------------------------
    def read_controls(self) -> None:
        self.focal_length_mm = float(dpg.get_value("focal_length"))
        self.f_number = float(dpg.get_value("f_number"))
        self.focus_distance_mm = 10.0 ** float(dpg.get_value("focus_log10_m")) * 1000.0
        self.format_id = self._format_id_by_name[dpg.get_value("format_combo")]
        self.pixel_pitch_um = float(dpg.get_value("pixel_pitch_um"))
        self.object_height_mm = float(dpg.get_value("object_height"))
        self.use_pixel_coc = bool(dpg.get_value("use_pixel_coc"))
        self.distortion_k1 = float(dpg.get_value("distortion_k1"))
        self.distortion_k2 = float(dpg.get_value("distortion_k2"))

    def push_controls(self) -> None:
        dpg.set_value("focal_length", self.focal_length_mm)
        dpg.set_value("f_number", self.f_number)
        dpg.set_value("focus_log10_m", math.log10(max(self.focus_distance_mm, 1.0) / 1000.0))
        dpg.set_value("format_combo", self._formats[self.format_id].name)
        dpg.set_value("pixel_pitch_um", self.pixel_pitch_um)
        dpg.set_value("object_height", self.object_height_mm)
        dpg.set_value("use_pixel_coc", self.use_pixel_coc)
        dpg.set_value("distortion_k1", self.distortion_k1)
        dpg.set_value("distortion_k2", self.distortion_k2)

    # --- compute + draw ---------------------------------------------
    def refresh(self) -> None:
        self.read_controls()
        s = ge.summarize(
            focal_length_mm=self.focal_length_mm,
            f_number=self.f_number,
            focus_distance_mm=self.focus_distance_mm,
            format_id=self.format_id,
            pixel_pitch_um=self.pixel_pitch_um,
            use_pixel_coc=self.use_pixel_coc,
            k1=self.distortion_k1,
            k2=self.distortion_k2,
        )
        # The slider is in log10 metres, so keep the real distance on its label.
        dpg.configure_item("focus_log10_m", label=f"Focus distance = {_format_distance(s.focus_distance_mm)}")

        self._draw_ray_diagram()
        self._draw_field_of_view(s)
        self._draw_depth_of_field(s)
        self._draw_vignetting_and_distortion(s)
        self._draw_status(s)

    def _draw_ray_diagram(self) -> None:
        d = ge.ray_diagram(
            focal_length_mm=self.focal_length_mm,
            object_distance_mm=self.focus_distance_mm,
            object_height_mm=self.object_height_mm,
        )
        for i, (xs, ys) in enumerate(d.rays):
            dpg.set_value(f"ray_{i}", [xs, ys])
        dpg.set_value("ray_object", [list(d.object_arrow[0]), list(d.object_arrow[1])])
        dpg.set_value("ray_image", [list(d.image_arrow[0]), list(d.image_arrow[1])])
        dpg.set_value("ray_lens", [list(d.lens_outline[0]), list(d.lens_outline[1])])
        dpg.set_value("ray_axis", [[d.x_limits[0], d.x_limits[1]], [0.0, 0.0]])
        dpg.set_value("ray_focal", [list(d.focal_points[0]), list(d.focal_points[1])])
        dpg.set_axis_limits("ray_x", *d.x_limits)
        dpg.set_axis_limits("ray_y", *d.y_limits)
        dpg.set_value(
            "ray_caption",
            "No real image: the object is inside the front focal length."
            if d.image_is_virtual
            else "Object side and image side use independent scales; read magnification from the table.",
        )

    def _draw_field_of_view(self, s: ge.GeometrySummary) -> None:
        focal, fov = ge.fov_vs_focal_length(sensor_width_mm=s.sensor_width_mm, sensor_height_mm=s.sensor_height_mm)
        dpg.set_value("fov_curve", [focal.tolist(), fov.tolist()])
        dpg.set_value("fov_marker", [[s.focal_length_mm], [s.fov_diagonal_deg]])

        xs, ys = ge.subject_footprint(
            focal_length_mm=s.focal_length_mm,
            sensor_width_mm=s.sensor_width_mm,
            sensor_height_mm=s.sensor_height_mm,
            distance_mm=s.focus_distance_mm,
        )
        dpg.set_value("footprint", [xs, ys])
        span = max(max(map(abs, xs)), max(map(abs, ys))) * 1.25
        dpg.set_axis_limits("footprint_x", -span, span)
        dpg.set_axis_limits("footprint_y", -span, span)

    def _draw_depth_of_field(self, s: ge.GeometrySummary) -> None:
        lo = max(s.focal_length_mm * 1.5, 50.0)
        hi = max(s.hyperfocal_mm * 4.0, s.focus_distance_mm * 4.0)
        sweep = ge.dof_vs_focus_distance(
            focal_length_mm=s.focal_length_mm,
            f_number=s.f_number,
            coc_mm=s.coc_mm,
            min_distance_mm=lo,
            max_distance_mm=hi,
        )
        dpg.set_value("dof_near", [sweep.focus_distance_m.tolist(), sweep.near_m.tolist()])
        dpg.set_value("dof_far", [sweep.focus_distance_m.tolist(), sweep.far_m.tolist()])
        dpg.set_value(
            "dof_focus",
            [sweep.focus_distance_m.tolist(), sweep.focus_distance_m.tolist()],
        )
        dpg.set_value("dof_marker", [[s.focus_distance_mm / 1000.0], [s.focus_distance_mm / 1000.0]])
        dpg.set_axis_limits("dof_x", float(sweep.focus_distance_m[0]), float(sweep.focus_distance_m[-1]))
        dpg.set_axis_limits("dof_y", float(sweep.focus_distance_m[0]), float(sweep.far_m.max()))

        d_m, blur_um = ge.blur_vs_object_distance(
            focal_length_mm=s.focal_length_mm,
            f_number=s.f_number,
            focus_distance_mm=s.focus_distance_mm,
            min_distance_mm=lo,
            max_distance_mm=hi,
        )
        dpg.set_value("blur_curve", [d_m.tolist(), blur_um.tolist()])
        coc_um = s.coc_mm * 1000.0
        dpg.set_value("blur_coc", [[float(d_m[0]), float(d_m[-1])], [coc_um, coc_um]])
        dpg.set_axis_limits("blur_x", float(d_m[0]), float(d_m[-1]))
        dpg.set_axis_limits("blur_y", 0.0, max(coc_um * 4.0, 1e-3))

        # Aperture trade-off, evaluated at the near depth-of-field limit so
        # there is a real defocus error to trade against diffraction.
        probe = s.dof_near_mm if math.isfinite(s.dof_near_mm) else s.focus_distance_mm * 0.9
        trade = ge.aperture_tradeoff(
            focal_length_mm=s.focal_length_mm,
            focus_distance_mm=s.focus_distance_mm,
            object_distance_mm=probe,
        )
        dpg.set_value("trade_defocus", [trade.f_numbers.tolist(), trade.defocus_blur_um.tolist()])
        dpg.set_value("trade_diffraction", [trade.f_numbers.tolist(), trade.diffraction_blur_um.tolist()])
        dpg.set_value("trade_total", [trade.f_numbers.tolist(), trade.total_blur_um.tolist()])
        best = float(np.min(trade.total_blur_um))
        dpg.set_value("trade_optimum", [[trade.optimum_f_number], [best]])
        dpg.set_axis_limits("trade_x", float(trade.f_numbers[0]), float(trade.f_numbers[-1]))
        dpg.set_axis_limits("trade_y", 0.0, float(np.percentile(trade.total_blur_um, 95)) * 1.3 + 1e-6)
        dpg.set_value(
            "trade_caption",
            f"Sharpest aperture for an object at {_format_distance(probe)}: "
            f"f/{trade.optimum_f_number:.1f}  (total blur {best:.2f} um)",
        )

    def _draw_vignetting_and_distortion(self, s: ge.GeometrySummary) -> None:
        r_norm, ri = ge.relative_illumination(
            focal_length_mm=s.focal_length_mm, sensor_diagonal_mm=s.sensor_diagonal_mm
        )
        dpg.set_value("ri_curve", [r_norm.tolist(), ri.tolist()])

        preview = ge.vignetting_preview(
            focal_length_mm=s.focal_length_mm,
            sensor_width_mm=s.sensor_width_mm,
            sensor_height_mm=s.sensor_height_mm,
            size=VIGNETTE_SIZE,
        )
        dpg.set_value("vignette_texture", _gray_rgba_flat(preview))
        dpg.set_value(
            "vignette_caption",
            f"cos^4 relative illumination   corner = {float(ri[-1]) * 100.0:.1f}% "
            f"({s.corner_falloff_stops:.2f} stops down)",
        )

        lines = ge.distortion_grid(
            k1=self.distortion_k1,
            k2=self.distortion_k2,
            aspect_ratio=s.sensor_width_mm / s.sensor_height_mm,
        )
        for i in range(N_GRID_SERIES):
            if i < len(lines):
                dpg.set_value(f"grid_{i}", [lines[i][0], lines[i][1]])
            else:
                dpg.set_value(f"grid_{i}", [[], []])
        shape = "barrel" if s.distortion_percent < 0 else ("pincushion" if s.distortion_percent > 0 else "none")
        dpg.set_value(
            "distortion_caption",
            f"Corner distortion {s.distortion_percent:+.2f}%  ({shape})",
        )

    def _draw_status(self, s: ge.GeometrySummary) -> None:
        dof_far = _format_distance(s.dof_far_mm)
        dof_total = "infinite" if not math.isfinite(s.dof_total_mm) else _format_distance(s.dof_total_mm)
        coc_kind = "pixel (2 px)" if self.use_pixel_coc else "print (diag/1440)"
        dpg.set_value(
            "status_text",
            (
                f"{s.format_name}  {s.sensor_width_mm:.1f} x {s.sensor_height_mm:.1f} mm   "
                f"crop {s.crop_factor:.2f}x   {s.megapixels:.1f} MP at {self.pixel_pitch_um:.2f} um\n"
                f"{s.focal_length_mm:.1f} mm = {s.equivalent_focal_length_mm:.0f} mm equivalent   "
                f"aperture {s.aperture_diameter_mm:.1f} mm\n"
                f"FOV {s.fov_horizontal_deg:.1f} x {s.fov_vertical_deg:.1f} deg "
                f"({s.fov_diagonal_deg:.1f} diag)   covers "
                f"{s.subject_width_mm / 1000.0:.2f} x {s.subject_height_mm / 1000.0:.2f} m\n"
                f"Magnification {abs(s.magnification):.4f}x   image distance {s.image_distance_mm:.2f} mm   "
                f"bellows {s.bellows_factor:.2f}x (N_eff f/{s.effective_f_number:.1f})\n"
                f"CoC {s.coc_mm * 1000.0:.1f} um [{coc_kind}]   hyperfocal {_format_distance(s.hyperfocal_mm)}\n"
                f"Depth of field {_format_distance(s.dof_near_mm)} to {dof_far}  (total {dof_total})\n"
                f"Corner falloff {s.corner_falloff_stops:.2f} stops   "
                f"distortion {s.distortion_percent:+.2f}%"
            ),
        )

    # --- layout -----------------------------------------------------
    def register_themes(self) -> None:
        with dpg.theme() as ray_theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 200, 90), category=dpg.mvThemeCat_Plots)
        self._ray_theme = ray_theme

        with dpg.theme() as limit_theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (240, 110, 110), category=dpg.mvThemeCat_Plots)
        self._limit_theme = limit_theme

    def register_textures(self) -> None:
        blank = [0.0, 0.0, 0.0, 1.0] * (VIGNETTE_SIZE * VIGNETTE_SIZE)
        dpg.add_dynamic_texture(VIGNETTE_SIZE, VIGNETTE_SIZE, blank, tag="vignette_texture")

    def build_controls(self) -> None:
        dpg.add_text("Core controls")
        dpg.add_slider_float(
            tag="focal_length",
            label="Focal length (mm)",
            default_value=self.focal_length_mm,
            min_value=4.0,
            max_value=300.0,
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
            tag="focus_log10_m",
            label="Focus distance",
            default_value=math.log10(self.focus_distance_mm / 1000.0),
            min_value=-1.7,
            max_value=2.0,
            format="%.2f",
            callback=self.on_control_change,
        )
        dpg.add_text("Sensor format")
        dpg.add_combo(
            tag="format_combo",
            items=[f.name for f in self._formats.values()],
            default_value=self._formats[self.format_id].name,
            callback=self.on_control_change,
        )

        with dpg.group(tag="advanced_controls"):
            dpg.add_separator()
            dpg.add_text("Advanced")
            dpg.add_slider_float(
                tag="pixel_pitch_um",
                label="Pixel pitch (um)",
                default_value=self.pixel_pitch_um,
                min_value=0.7,
                max_value=9.0,
                callback=self.on_control_change,
            )
            dpg.add_checkbox(
                tag="use_pixel_coc",
                label="Pixel-level CoC (2 px) instead of print",
                default_value=self.use_pixel_coc,
                callback=self.on_control_change,
            )
            dpg.add_slider_float(
                tag="object_height",
                label="Subject height (mm)",
                default_value=self.object_height_mm,
                min_value=10.0,
                max_value=3000.0,
                callback=self.on_control_change,
            )
            dpg.add_slider_float(
                tag="distortion_k1",
                label="Distortion k1",
                default_value=self.distortion_k1,
                min_value=-0.4,
                max_value=0.4,
                format="%.3f",
                callback=self.on_control_change,
            )
            dpg.add_slider_float(
                tag="distortion_k2",
                label="Distortion k2",
                default_value=self.distortion_k2,
                min_value=-0.2,
                max_value=0.2,
                format="%.3f",
                callback=self.on_control_change,
            )

    def build_content(self) -> None:
        with dpg.tab_bar():
            self._build_geometry_tab()
            self._build_dof_tab()
            self._build_falloff_tab()

    def _build_geometry_tab(self) -> None:
        with dpg.tab(label="Imaging geometry"):
            with dpg.plot(label="Thin-lens construction (schematic)", height=330, width=-1):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="object side  |  lens  |  image side", tag="ray_x")
                with dpg.plot_axis(dpg.mvYAxis, label="height (normalised)", tag="ray_y"):
                    dpg.add_line_series([0.0], [0.0], label="optical axis", tag="ray_axis")
                    dpg.add_line_series([0.0], [0.0], label="lens", tag="ray_lens")
                    dpg.add_line_series([0.0], [0.0], label="object", tag="ray_object")
                    dpg.add_line_series([0.0], [0.0], label="image", tag="ray_image")
                    for i in range(3):
                        r = dpg.add_line_series([0.0], [0.0], label=f"ray {i + 1}", tag=f"ray_{i}")
                        dpg.bind_item_theme(r, self._ray_theme)
                    dpg.add_scatter_series([0.0], [0.0], label="focal points", tag="ray_focal")
            dpg.add_text("", tag="ray_caption", wrap=900)

            with dpg.group(horizontal=True):
                with dpg.plot(label="Diagonal field of view vs focal length", height=320, width=520):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="focal length (mm)", tag="fov_x", scale=dpg.mvPlotScale_Log10)
                    with dpg.plot_axis(dpg.mvYAxis, label="diagonal FOV (deg)", tag="fov_y"):
                        dpg.add_line_series([0.0], [0.0], label="this sensor format", tag="fov_curve")
                        dpg.add_scatter_series([0.0], [0.0], label="current lens", tag="fov_marker")
                with dpg.plot(label="Subject footprint at the focus distance (m)", height=320, width=520):
                    dpg.add_plot_axis(dpg.mvXAxis, label="width (m)", tag="footprint_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="height (m)", tag="footprint_y"):
                        dpg.add_line_series([0.0], [0.0], label="frame", tag="footprint")

    def _build_dof_tab(self) -> None:
        with dpg.tab(label="Depth of field"):
            with dpg.plot(label="Sharpness limits vs focus distance", height=320, width=-1):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="focus distance (m)", tag="dof_x", scale=dpg.mvPlotScale_Log10)
                with dpg.plot_axis(dpg.mvYAxis, label="distance (m)", tag="dof_y", scale=dpg.mvPlotScale_Log10):
                    near = dpg.add_line_series([1.0], [1.0], label="near limit", tag="dof_near")
                    far = dpg.add_line_series([1.0], [1.0], label="far limit", tag="dof_far")
                    dpg.add_line_series([1.0], [1.0], label="plane of focus", tag="dof_focus")
                    dpg.add_scatter_series([1.0], [1.0], label="current setting", tag="dof_marker")
                    dpg.bind_item_theme(near, self._limit_theme)
                    dpg.bind_item_theme(far, self._limit_theme)

            with dpg.group(horizontal=True):
                with dpg.plot(label="Defocus blur vs where the object is", height=300, width=520):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(
                        dpg.mvXAxis, label="object distance (m)", tag="blur_x", scale=dpg.mvPlotScale_Log10
                    )
                    with dpg.plot_axis(dpg.mvYAxis, label="blur disc (um)", tag="blur_y"):
                        dpg.add_line_series([1.0], [0.0], label="blur diameter", tag="blur_curve")
                        dpg.add_line_series([1.0], [0.0], label="circle of confusion", tag="blur_coc")
                with dpg.plot(label="Aperture trade-off: defocus vs diffraction", height=300, width=520):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="f-number", tag="trade_x", scale=dpg.mvPlotScale_Log10)
                    with dpg.plot_axis(dpg.mvYAxis, label="blur (um)", tag="trade_y"):
                        dpg.add_line_series([1.0], [0.0], label="defocus", tag="trade_defocus")
                        dpg.add_line_series([1.0], [0.0], label="diffraction", tag="trade_diffraction")
                        dpg.add_line_series([1.0], [0.0], label="total (quadrature)", tag="trade_total")
                        dpg.add_scatter_series([1.0], [0.0], label="optimum", tag="trade_optimum")
            dpg.add_text("", tag="trade_caption", wrap=900)

    def _build_falloff_tab(self) -> None:
        with dpg.tab(label="Vignetting / distortion"):
            with dpg.group(horizontal=True):
                with dpg.group():
                    dpg.add_text("", tag="vignette_caption")
                    dpg.add_image("vignette_texture", width=300, height=300)
                with dpg.plot(label="Relative illumination vs image height", height=320, width=640):
                    dpg.add_plot_axis(dpg.mvXAxis, label="image height (0 = centre, 1 = corner)", tag="ri_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="relative illumination", tag="ri_y"):
                        dpg.set_axis_limits("ri_y", 0.0, 1.05)
                        dpg.add_line_series([0.0], [1.0], label="cos^4", tag="ri_curve")

            dpg.add_separator()
            dpg.add_text("", tag="distortion_caption")
            with dpg.plot(label="Brown-Conrady distortion of a straight grid", height=380, width=-1):
                dpg.add_plot_axis(dpg.mvXAxis, label="normalised x", tag="grid_x")
                with dpg.plot_axis(dpg.mvYAxis, label="normalised y", tag="grid_y"):
                    dpg.set_axis_limits("grid_y", -1.0, 1.0)
                    for i in range(N_GRID_SERIES):
                        dpg.add_line_series([0.0], [0.0], tag=f"grid_{i}")


def run_app(scenario_id: str | None = None) -> None:
    GeometryApp(scenario_id=scenario_id).run()
