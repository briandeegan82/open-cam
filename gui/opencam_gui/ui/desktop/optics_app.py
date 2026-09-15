"""Dear PyGui desktop app: optics / point-spread-function explorer.

Every plotted curve or image comes from ``tools/apply_spectral_psf.py`` via
``opencam_gui.core.optics_engine`` — this file only wires sliders to that
module and draws the results.
"""

from __future__ import annotations

import numpy as np
import dearpygui.dearpygui as dpg

from opencam_gui.core import optics_engine as oe
from opencam_gui.core.camera import optics_summary
from opencam_gui.topics.optics.scenarios import SCENARIOS, get_scenario
from opencam_gui.ui.desktop.base import DemoApp

KERNEL_SIZE = 65
CHART_SIZE = 160
_RGB_HEX = {"R": (255, 100, 100), "G": (100, 220, 100), "B": (110, 160, 255)}


def _rgba_flat(rgb: np.ndarray) -> list[float]:
    h, w = rgb.shape[:2]
    rgba = np.ones((h, w, 4), dtype=np.float64)
    rgba[..., :3] = rgb
    return rgba.ravel().tolist()


def _gray_rgba_flat(gray: np.ndarray) -> list[float]:
    g = np.clip(gray, 0.0, 1.0)
    rgb = np.stack([g, g, g], axis=-1)
    return _rgba_flat(rgb)


class OpticsApp(DemoApp):
    viewport_title = "Open Cam — Optics / PSF Explorer"
    viewport_width = 1400
    viewport_height = 900
    window_label = "Optics"
    control_panel_width = 360
    scenarios = SCENARIOS
    default_banner_title = "Interactive PSF explorer"
    default_banner_body = (
        "Pick a camera recipe or drag the sliders. All curves are the real "
        "tools/apply_spectral_psf.py functions, not a re-derivation."
    )

    def init_state(self) -> None:
        self.mode = "chromatic_gaussian"
        self.f_number = 2.0
        self.pixel_pitch_um = 1.4
        self.sigma_geometric_px = 0.5
        self.lateral_ca_coefficient = 0.0
        self._line_themes: dict[str, int] = {}

    def get_scenario(self, scenario_id: str):
        return get_scenario(scenario_id)

    def apply_scenario_state(self, sc) -> None:
        self.mode = sc.mode
        self.f_number = sc.f_number
        self.pixel_pitch_um = sc.pixel_pitch_um
        self.sigma_geometric_px = sc.sigma_geometric_px
        self.lateral_ca_coefficient = sc.lateral_ca_coefficient

    def apply_recipe_state(self, model: dict) -> None:
        summary = optics_summary(model)
        self.f_number = summary.f_number
        self.pixel_pitch_um = summary.pixel_pitch_um
        self.sigma_geometric_px = summary.sigma_geometric_pixels
        self.lateral_ca_coefficient = summary.lateral_ca_coefficient

    # --- controls --------------------------------------------------
    def read_controls(self) -> None:
        self.mode = dpg.get_value("psf_mode")
        self.f_number = float(dpg.get_value("f_number"))
        self.pixel_pitch_um = float(dpg.get_value("pixel_pitch_um"))
        self.sigma_geometric_px = float(dpg.get_value("sigma_geom"))
        self.lateral_ca_coefficient = float(dpg.get_value("lca_coeff"))

    def push_controls(self) -> None:
        dpg.set_value("psf_mode", self.mode)
        dpg.set_value("f_number", self.f_number)
        dpg.set_value("pixel_pitch_um", self.pixel_pitch_um)
        dpg.set_value("sigma_geom", self.sigma_geometric_px)
        dpg.set_value("lca_coeff", self.lateral_ca_coefficient)

    # --- compute + draw ---------------------------------------------
    def refresh(self) -> None:
        self.read_controls()
        centers = oe.rgb_center_wavelengths_nm()

        results = {}
        for ch, wl in centers.items():
            results[ch] = oe.compute_psf_kernel(
                mode=self.mode,
                wavelength_nm=wl,
                f_number=self.f_number,
                pixel_pitch_um=self.pixel_pitch_um,
                sigma_geometric_px=self.sigma_geometric_px,
                size=KERNEL_SIZE,
            )

        max_r = 0.0
        for ch, res in results.items():
            radii, profile = oe.radial_profile(res.kernel)
            dpg.set_value(f"radial_{ch}", [radii.tolist(), profile.tolist()])
            max_r = max(max_r, float(radii[-1]))
        dpg.set_axis_limits("radial_x", 0.0, max(4.0, max_r))

        # 2-D PSF heatmap (green channel, representative)
        g_kernel = results["G"].kernel
        dpg.configure_item(
            "psf_heat",
            bounds_min=(-KERNEL_SIZE / 2.0, -KERNEL_SIZE / 2.0),
            bounds_max=(KERNEL_SIZE / 2.0, KERNEL_SIZE / 2.0),
        )
        dpg.set_value("psf_heat", [g_kernel.ravel().tolist()])

        # Lateral CA preview
        chart = oe.radial_test_chart(CHART_SIZE, n_rings=6)
        rgb = oe.lateral_ca_rgb_preview(chart, self.lateral_ca_coefficient, 550.0, centers)
        dpg.set_value("ca_texture", _rgba_flat(rgb))
        dpg.set_value("ca_gray_texture", _gray_rgba_flat(chart))

        for ch in ("R", "G", "B"):
            r = results[ch]
            dpg.set_value(f"tbl_{ch}_lambda", f"{centers[ch]:.0f}")
            dpg.set_value(f"tbl_{ch}_sdiff", f"{r.sigma_diff_px:.3f}")
            dpg.set_value(f"tbl_{ch}_sgeom", f"{r.sigma_geom_px:.3f}")
            dpg.set_value(f"tbl_{ch}_stotal", f"{r.sigma_total_px:.3f}")
            dpg.set_value(f"tbl_{ch}_rho0", f"{r.rho0_px:.3f}" if r.rho0_px else "-")

        diffraction_limited = results["B"].sigma_diff_px >= self.sigma_geometric_px
        dpg.set_value(
            "status_text",
            (
                f"f/{self.f_number:.2f}   pixel pitch {self.pixel_pitch_um:.2f} um   mode={self.mode}\n"
                "Diffraction-limited (blue sigma_diff >= sigma_geom): "
                + ("YES" if diffraction_limited else "no - aberration-limited")
                + (
                    f"\nLateral CA coefficient: {self.lateral_ca_coefficient:.4f} (0 = none)"
                    if self.lateral_ca_coefficient > 0
                    else ""
                )
            ),
        )

    # --- layout -----------------------------------------------------
    def register_themes(self) -> None:
        for ch, rgb in _RGB_HEX.items():
            with dpg.theme() as th:
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, rgb, category=dpg.mvThemeCat_Plots)
            self._line_themes[ch] = th

    def register_textures(self) -> None:
        blank = [0.0] * CHART_SIZE * CHART_SIZE * 4
        dpg.add_dynamic_texture(CHART_SIZE, CHART_SIZE, blank, tag="ca_texture")
        dpg.add_dynamic_texture(CHART_SIZE, CHART_SIZE, list(blank), tag="ca_gray_texture")

    def build_controls(self) -> None:
        dpg.add_text("Core controls")
        dpg.add_slider_float(
            tag="f_number", label="f-number (N)", default_value=self.f_number,
            min_value=1.0, max_value=22.0, callback=self.on_control_change,
        )
        dpg.add_slider_float(
            tag="pixel_pitch_um", label="Pixel pitch (um)", default_value=self.pixel_pitch_um,
            min_value=0.7, max_value=8.0, callback=self.on_control_change,
        )
        dpg.add_combo(
            tag="psf_mode", label="PSF mode", items=["chromatic_gaussian", "airy_disk"],
            default_value=self.mode, callback=self.on_control_change,
        )

        with dpg.group(tag="advanced_controls"):
            dpg.add_separator()
            dpg.add_text("Advanced")
            dpg.add_slider_float(
                tag="sigma_geom", label="Geometric aberration sigma (px)",
                default_value=self.sigma_geometric_px, min_value=0.0, max_value=3.0,
                callback=self.on_control_change,
            )
            dpg.add_slider_float(
                tag="lca_coeff", label="Lateral CA coefficient",
                default_value=self.lateral_ca_coefficient, min_value=0.0, max_value=0.05,
                callback=self.on_control_change,
            )

    def build_content(self) -> None:
        with dpg.table(
            header_row=True, borders_innerH=True, borders_outerH=True,
            borders_innerV=True, borders_outerV=True,
        ):
            dpg.add_table_column(label="Channel")
            dpg.add_table_column(label="lambda (nm)")
            dpg.add_table_column(label="sigma_diff (px)")
            dpg.add_table_column(label="sigma_geom (px)")
            dpg.add_table_column(label="sigma_total (px)")
            dpg.add_table_column(label="Airy rho0 (px)")
            for ch in ("R", "G", "B"):
                with dpg.table_row():
                    dpg.add_text(ch)
                    dpg.add_text("", tag=f"tbl_{ch}_lambda")
                    dpg.add_text("", tag=f"tbl_{ch}_sdiff")
                    dpg.add_text("", tag=f"tbl_{ch}_sgeom")
                    dpg.add_text("", tag=f"tbl_{ch}_stotal")
                    dpg.add_text("", tag=f"tbl_{ch}_rho0")

        with dpg.plot(label="Radial PSF profile (normalised)", height=300, width=-1, tag="radial_plot"):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="radius (px)", tag="radial_x")
            with dpg.plot_axis(dpg.mvYAxis, label="intensity (peak = 1)", tag="radial_y"):
                dpg.set_axis_limits("radial_y", -0.05, 1.05)
                for ch in ("R", "G", "B"):
                    s = dpg.add_line_series([0.0], [0.0], label=f"{ch} channel", tag=f"radial_{ch}")
                    dpg.bind_item_theme(s, self._line_themes[ch])

        with dpg.group(horizontal=True):
            with dpg.plot(label="2-D PSF kernel (G channel)", height=340, width=460):
                dpg.add_plot_axis(dpg.mvXAxis, label="x (px)")
                with dpg.plot_axis(dpg.mvYAxis, label="y (px)"):
                    dpg.add_heat_series(
                        [0.0] * (KERNEL_SIZE * KERNEL_SIZE),
                        KERNEL_SIZE,
                        KERNEL_SIZE,
                        scale_min=0.0,
                        scale_max=1.0,
                        bounds_min=(-KERNEL_SIZE / 2.0, -KERNEL_SIZE / 2.0),
                        bounds_max=(KERNEL_SIZE / 2.0, KERNEL_SIZE / 2.0),
                        format="",
                        tag="psf_heat",
                    )
            with dpg.group():
                dpg.add_text("Lateral-CA test chart (input, greyscale)")
                dpg.add_image("ca_gray_texture", width=200, height=200)
                dpg.add_text("After per-channel lateral CA (apply_lateral_ca x R/G/B)")
                dpg.add_image("ca_texture", width=200, height=200)


def run_app(scenario_id: str | None = None) -> None:
    OpticsApp(scenario_id=scenario_id).run()
