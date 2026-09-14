"""Dear PyGui desktop app: optics / point-spread-function explorer.

Every plotted curve or image comes from ``tools/apply_spectral_psf.py`` via
``opencam_gui.core.optics_engine`` — this file only wires sliders to that
module and draws the results.
"""

from __future__ import annotations

import numpy as np
import dearpygui.dearpygui as dpg

from opencam_gui.core import optics_engine as oe
from opencam_gui.core.camera import load_camera_model, optics_summary
from opencam_gui.core.catalog import list_camera_recipes
from opencam_gui.topics.optics.scenarios import SCENARIOS, get_scenario

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


class OpticsApp:
    def __init__(self, scenario_id: str | None = None) -> None:
        self.mode = "chromatic_gaussian"
        self.f_number = 2.0
        self.pixel_pitch_um = 1.4
        self.sigma_geometric_px = 0.5
        self.lateral_ca_coefficient = 0.0
        self.camera_recipe_id: str | None = None
        self._presenter = False
        self._scenario_title = "Interactive PSF explorer"
        self._scenario_note = (
            "Pick a camera recipe or drag the sliders. All curves are the real "
            "tools/apply_spectral_psf.py functions, not a re-derivation."
        )
        self._recipes = list_camera_recipes()
        if scenario_id:
            self._apply_scenario_fields(get_scenario(scenario_id))

    def _apply_scenario_fields(self, sc) -> None:
        self.mode = sc.mode
        self.f_number = sc.f_number
        self.pixel_pitch_um = sc.pixel_pitch_um
        self.sigma_geometric_px = sc.sigma_geometric_px
        self.lateral_ca_coefficient = sc.lateral_ca_coefficient
        self.camera_recipe_id = sc.camera_recipe_id
        self._scenario_title = sc.title
        self._scenario_note = f"{sc.teaching_point}\n{sc.notes}"

    # --- controls --------------------------------------------------
    def _read_controls(self) -> None:
        self.mode = dpg.get_value("psf_mode")
        self.f_number = float(dpg.get_value("f_number"))
        self.pixel_pitch_um = float(dpg.get_value("pixel_pitch_um"))
        self.sigma_geometric_px = float(dpg.get_value("sigma_geom"))
        self.lateral_ca_coefficient = float(dpg.get_value("lca_coeff"))

    def _push_controls(self) -> None:
        dpg.set_value("psf_mode", self.mode)
        dpg.set_value("f_number", self.f_number)
        dpg.set_value("pixel_pitch_um", self.pixel_pitch_um)
        dpg.set_value("sigma_geom", self.sigma_geometric_px)
        dpg.set_value("lca_coeff", self.lateral_ca_coefficient)

    def _load_recipe(self, recipe_id: str) -> None:
        recipe = next((r for r in self._recipes if r.id == recipe_id), None)
        if recipe is None:
            return
        model = load_camera_model(recipe.path)
        summary = optics_summary(model)
        self.f_number = summary.f_number
        self.pixel_pitch_um = summary.pixel_pitch_um
        self.sigma_geometric_px = summary.sigma_geometric_pixels
        self.lateral_ca_coefficient = summary.lateral_ca_coefficient
        self.camera_recipe_id = recipe_id
        self._push_controls()
        self.refresh()

    def _on_recipe_change(self, _sender=None, app_data=None, _user_data=None) -> None:
        if app_data and app_data != "Custom":
            self._load_recipe(app_data)

    def _apply_scenario(self, _sender=None, _app_data=None, user_data=None) -> None:
        sc = get_scenario(user_data)
        self._apply_scenario_fields(sc)
        self._push_controls()
        dpg.set_value("recipe_combo", sc.camera_recipe_id or "Custom")
        dpg.set_value("banner_title", self._scenario_title)
        dpg.set_value("banner_body", self._scenario_note)
        self.refresh()

    def _set_presenter(self, _sender=None, app_data=None, _user_data=None) -> None:
        self._presenter = bool(app_data)
        dpg.configure_item("advanced_controls", show=not self._presenter)

    # --- compute + draw ---------------------------------------------
    def refresh(self) -> None:
        self._read_controls()
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
        dpg.set_value("banner_title", self._scenario_title)
        dpg.set_value("banner_body", self._scenario_note)

    def run(self) -> None:
        dpg.create_context()
        dpg.create_viewport(title="Open Cam — Optics / PSF Explorer", width=1400, height=900)

        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6)
        dpg.bind_theme(global_theme)

        line_themes = {}
        for ch, rgb in _RGB_HEX.items():
            with dpg.theme() as th:
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, rgb, category=dpg.mvThemeCat_Plots)
            line_themes[ch] = th

        with dpg.texture_registry():
            dpg.add_dynamic_texture(CHART_SIZE, CHART_SIZE, [0.0] * CHART_SIZE * CHART_SIZE * 4, tag="ca_texture")
            dpg.add_dynamic_texture(
                CHART_SIZE, CHART_SIZE, [0.0] * CHART_SIZE * CHART_SIZE * 4, tag="ca_gray_texture"
            )

        with dpg.window(tag="primary", label="Optics"):
            with dpg.child_window(tag="banner_panel", height=72, border=True):
                dpg.add_text(self._scenario_title, tag="banner_title")
                dpg.add_text(self._scenario_note, tag="banner_body", wrap=1000)

            with dpg.group(horizontal=True):
                with dpg.child_window(width=360, border=True):
                    dpg.add_checkbox(
                        tag="presenter_mode",
                        label="Presenter mode (hide advanced)",
                        default_value=False,
                        callback=self._set_presenter,
                    )
                    dpg.add_separator()
                    dpg.add_text("Camera recipe")
                    dpg.add_combo(
                        tag="recipe_combo",
                        items=["Custom", *[r.id for r in self._recipes]],
                        default_value=self.camera_recipe_id or "Custom",
                        callback=self._on_recipe_change,
                    )
                    dpg.add_separator()
                    dpg.add_text("Core controls")
                    dpg.add_slider_float(
                        tag="f_number", label="f-number (N)", default_value=self.f_number,
                        min_value=1.0, max_value=22.0, callback=lambda *_: self.refresh(),
                    )
                    dpg.add_slider_float(
                        tag="pixel_pitch_um", label="Pixel pitch (um)", default_value=self.pixel_pitch_um,
                        min_value=0.7, max_value=8.0, callback=lambda *_: self.refresh(),
                    )
                    dpg.add_combo(
                        tag="psf_mode", label="PSF mode", items=["chromatic_gaussian", "airy_disk"],
                        default_value=self.mode, callback=lambda *_: self.refresh(),
                    )

                    with dpg.group(tag="advanced_controls"):
                        dpg.add_separator()
                        dpg.add_text("Advanced")
                        dpg.add_slider_float(
                            tag="sigma_geom", label="Geometric aberration sigma (px)",
                            default_value=self.sigma_geometric_px, min_value=0.0, max_value=3.0,
                            callback=lambda *_: self.refresh(),
                        )
                        dpg.add_slider_float(
                            tag="lca_coeff", label="Lateral CA coefficient",
                            default_value=self.lateral_ca_coefficient, min_value=0.0, max_value=0.05,
                            callback=lambda *_: self.refresh(),
                        )

                    dpg.add_separator()
                    dpg.add_text("Lecture scenarios")
                    for sid, sc in SCENARIOS.items():
                        dpg.add_button(label=sc.title, width=-1, user_data=sid, callback=self._apply_scenario)
                    dpg.add_separator()
                    dpg.add_text("", tag="status_text", wrap=330)

                with dpg.child_window(border=False):
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
                                dpg.bind_item_theme(s, line_themes[ch])

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
                                    tag="psf_heat",
                                )
                        with dpg.group():
                            dpg.add_text("Lateral-CA test chart (input, greyscale)")
                            dpg.add_image("ca_gray_texture", width=200, height=200)
                            dpg.add_text("After per-channel lateral CA (apply_lateral_ca x R/G/B)")
                            dpg.add_image("ca_texture", width=200, height=200)

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("primary", True)
        self._push_controls()
        dpg.set_value("recipe_combo", self.camera_recipe_id or "Custom")
        self.refresh()

        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()

        dpg.destroy_context()


def run_app(scenario_id: str | None = None) -> None:
    OpticsApp(scenario_id=scenario_id).run()
