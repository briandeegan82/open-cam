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
FLARE_SIZE = 192
STARBURST_SIZE = 128
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
    viewport_width = 1460
    viewport_height = 940
    window_label = "Optics"
    control_panel_width = 400
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

        self.stray_light_enabled = False
        self.veiling_glare_fraction = 0.0
        self.halo_sigma_pixels = 15.0
        self.halo_strength = 0.0
        self.ghost_enabled = False
        self.ghost_strength = 0.02
        self.aperture_diffraction_enabled = False
        self.n_blades = 6
        self.diffraction_strength = 0.05
        self.blade_rotation_deg = 0.0

        self._line_themes: dict[str, int] = {}

    def get_scenario(self, scenario_id: str):
        return get_scenario(scenario_id)

    def apply_scenario_state(self, sc) -> None:
        self.mode = sc.mode
        self.f_number = sc.f_number
        self.pixel_pitch_um = sc.pixel_pitch_um
        self.sigma_geometric_px = sc.sigma_geometric_px
        self.lateral_ca_coefficient = sc.lateral_ca_coefficient
        self.stray_light_enabled = sc.stray_light_enabled
        self.veiling_glare_fraction = sc.veiling_glare_fraction
        self.halo_sigma_pixels = sc.halo_sigma_pixels
        self.halo_strength = sc.halo_strength
        self.ghost_enabled = sc.ghost_enabled
        self.ghost_strength = sc.ghost_strength
        self.aperture_diffraction_enabled = sc.aperture_diffraction_enabled
        self.n_blades = sc.n_blades
        self.diffraction_strength = sc.diffraction_strength
        self.blade_rotation_deg = sc.blade_rotation_deg

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
        self.stray_light_enabled = bool(dpg.get_value("stray_enabled"))
        self.veiling_glare_fraction = float(dpg.get_value("veiling"))
        self.halo_sigma_pixels = float(dpg.get_value("halo_sigma"))
        self.halo_strength = float(dpg.get_value("halo_strength"))
        self.ghost_enabled = bool(dpg.get_value("ghost_enabled"))
        self.ghost_strength = float(dpg.get_value("ghost_strength"))
        self.aperture_diffraction_enabled = bool(dpg.get_value("diffraction_enabled"))
        self.n_blades = int(dpg.get_value("n_blades"))
        self.diffraction_strength = float(dpg.get_value("diffraction_strength"))
        self.blade_rotation_deg = float(dpg.get_value("blade_rotation"))

    def push_controls(self) -> None:
        dpg.set_value("psf_mode", self.mode)
        dpg.set_value("f_number", self.f_number)
        dpg.set_value("pixel_pitch_um", self.pixel_pitch_um)
        dpg.set_value("sigma_geom", self.sigma_geometric_px)
        dpg.set_value("lca_coeff", self.lateral_ca_coefficient)
        dpg.set_value("stray_enabled", self.stray_light_enabled)
        dpg.set_value("veiling", self.veiling_glare_fraction)
        dpg.set_value("halo_sigma", self.halo_sigma_pixels)
        dpg.set_value("halo_strength", self.halo_strength)
        dpg.set_value("ghost_enabled", self.ghost_enabled)
        dpg.set_value("ghost_strength", self.ghost_strength)
        dpg.set_value("diffraction_enabled", self.aperture_diffraction_enabled)
        dpg.set_value("n_blades", self.n_blades)
        dpg.set_value("diffraction_strength", self.diffraction_strength)
        dpg.set_value("blade_rotation", self.blade_rotation_deg)

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

        self._refresh_stray_light()

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

    def _refresh_stray_light(self) -> None:
        clean = oe.stray_light_test_image(FLARE_SIZE)
        cfg = oe.stray_light_config(
            enabled=self.stray_light_enabled,
            veiling_glare_fraction=self.veiling_glare_fraction,
            halo_sigma_pixels=self.halo_sigma_pixels,
            halo_strength=self.halo_strength,
            ghost_enabled=self.ghost_enabled,
            ghost_strength=self.ghost_strength,
            aperture_diffraction_enabled=self.aperture_diffraction_enabled,
            n_blades=self.n_blades,
            diffraction_strength=self.diffraction_strength,
            rotation_deg=self.blade_rotation_deg,
            psf_kernel_size=STARBURST_SIZE,
        )
        strayed = oe.apply_stray_light(clean, cfg)

        dpg.set_value("flare_clean_texture", _gray_rgba_flat(oe.tone_for_display(clean)))
        dpg.set_value("flare_texture", _gray_rgba_flat(oe.tone_for_display(strayed)))

        starburst = oe.aperture_diffraction_kernel(self.n_blades, STARBURST_SIZE, self.blade_rotation_deg)
        dpg.set_value("starburst_texture", _gray_rgba_flat(np.power(starburst, 0.25)))
        dpg.set_value(
            "starburst_caption",
            f"{self.n_blades}-blade iris PSF  ->  "
            f"{self.n_blades if self.n_blades % 2 == 0 else 2 * self.n_blades} spikes",
        )

        ec = oe.edge_contrast(clean, strayed)
        dpg.set_value("edge_clean", [ec.position_px.tolist(), ec.clean.tolist()])
        dpg.set_value("edge_strayed", [ec.position_px.tolist(), ec.strayed.tolist()])
        dpg.set_axis_limits("edge_x", 0.0, float(ec.position_px[-1]))
        dpg.set_axis_limits("edge_y", 0.0, max(float(ec.strayed.max()), 1.0) * 1.1)
        dpg.set_value(
            "edge_caption",
            (
                f"Michelson contrast across the step edge: {ec.clean_contrast:.3f} clean  ->  "
                f"{ec.strayed_contrast:.3f} with stray light  "
                f"({ec.contrast_loss_percent:.1f}% lost)\n"
                "Veiling glare adds the same constant everywhere: the black-to-white difference "
                "survives, but the sum grows, so contrast falls with no extra blur."
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

        flare_blank = [0.0, 0.0, 0.0, 1.0] * (FLARE_SIZE * FLARE_SIZE)
        dpg.add_dynamic_texture(FLARE_SIZE, FLARE_SIZE, flare_blank, tag="flare_clean_texture")
        dpg.add_dynamic_texture(FLARE_SIZE, FLARE_SIZE, list(flare_blank), tag="flare_texture")
        dpg.add_dynamic_texture(
            STARBURST_SIZE,
            STARBURST_SIZE,
            [0.0, 0.0, 0.0, 1.0] * (STARBURST_SIZE * STARBURST_SIZE),
            tag="starburst_texture",
        )

    def build_controls(self) -> None:
        dpg.add_text("Core controls")
        dpg.add_slider_float(
            tag="f_number",
            label="f-number (N)",
            default_value=self.f_number,
            min_value=1.0,
            max_value=22.0,
            callback=self.on_control_change,
        )
        dpg.add_slider_float(
            tag="pixel_pitch_um",
            label="Pixel pitch (um)",
            default_value=self.pixel_pitch_um,
            min_value=0.7,
            max_value=8.0,
            callback=self.on_control_change,
        )
        dpg.add_combo(
            tag="psf_mode",
            label="PSF mode",
            items=["chromatic_gaussian", "airy_disk"],
            default_value=self.mode,
            callback=self.on_control_change,
        )

        dpg.add_separator()
        dpg.add_checkbox(
            tag="stray_enabled",
            label="Stray light enabled",
            default_value=self.stray_light_enabled,
            callback=self.on_control_change,
        )
        with dpg.collapsing_header(label="Stray light terms", default_open=True):
            dpg.add_slider_float(
                tag="veiling",
                label="Veiling glare fraction",
                default_value=self.veiling_glare_fraction,
                min_value=0.0,
                max_value=0.25,
                format="%.3f",
                callback=self.on_control_change,
            )
            dpg.add_slider_float(
                tag="halo_strength",
                label="Halo strength",
                default_value=self.halo_strength,
                min_value=0.0,
                max_value=0.25,
                format="%.3f",
                callback=self.on_control_change,
            )
            dpg.add_slider_float(
                tag="halo_sigma",
                label="Halo sigma (px)",
                default_value=self.halo_sigma_pixels,
                min_value=1.0,
                max_value=60.0,
                callback=self.on_control_change,
            )
            dpg.add_checkbox(
                tag="ghost_enabled",
                label="Ghost reflection",
                default_value=self.ghost_enabled,
                callback=self.on_control_change,
            )
            dpg.add_slider_float(
                tag="ghost_strength",
                label="Ghost strength",
                default_value=self.ghost_strength,
                min_value=0.0,
                max_value=0.2,
                format="%.3f",
                callback=self.on_control_change,
            )
            dpg.add_checkbox(
                tag="diffraction_enabled",
                label="Aperture blade diffraction",
                default_value=self.aperture_diffraction_enabled,
                callback=self.on_control_change,
            )
            dpg.add_slider_int(
                tag="n_blades",
                label="Iris blades",
                default_value=self.n_blades,
                min_value=3,
                max_value=14,
                callback=self.on_control_change,
            )
            dpg.add_slider_float(
                tag="diffraction_strength",
                label="Starburst strength",
                default_value=self.diffraction_strength,
                min_value=0.0,
                max_value=0.4,
                format="%.3f",
                callback=self.on_control_change,
            )
            dpg.add_slider_float(
                tag="blade_rotation",
                label="Blade rotation (deg)",
                default_value=self.blade_rotation_deg,
                min_value=0.0,
                max_value=90.0,
                callback=self.on_control_change,
            )

        with dpg.group(tag="advanced_controls"):
            dpg.add_separator()
            dpg.add_text("Advanced")
            dpg.add_slider_float(
                tag="sigma_geom",
                label="Geometric aberration sigma (px)",
                default_value=self.sigma_geometric_px,
                min_value=0.0,
                max_value=3.0,
                callback=self.on_control_change,
            )
            dpg.add_slider_float(
                tag="lca_coeff",
                label="Lateral CA coefficient",
                default_value=self.lateral_ca_coefficient,
                min_value=0.0,
                max_value=0.05,
                callback=self.on_control_change,
            )

    def build_content(self) -> None:
        with dpg.tab_bar():
            with dpg.tab(label="PSF and aberrations"):
                self._build_psf_tab()
            with dpg.tab(label="Stray light"):
                self._build_stray_light_tab()

    def _build_stray_light_tab(self) -> None:
        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text("Clean scene (log-tone mapped)")
                dpg.add_image("flare_clean_texture", width=300, height=300)
            with dpg.group():
                dpg.add_text("With stray light (apply_stray_light)")
                dpg.add_image("flare_texture", width=300, height=300)
            with dpg.group():
                dpg.add_text("", tag="starburst_caption")
                dpg.add_image("starburst_texture", width=260, height=260)

        dpg.add_separator()
        with dpg.plot(label="High-contrast edge profile", height=280, width=-1):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="column (px)", tag="edge_x")
            with dpg.plot_axis(dpg.mvYAxis, label="scene value", tag="edge_y"):
                dpg.add_line_series([0.0], [0.0], label="clean", tag="edge_clean")
                dpg.add_line_series([0.0], [0.0], label="with stray light", tag="edge_strayed")
        dpg.add_text("", tag="edge_caption", wrap=940)

    def _build_psf_tab(self) -> None:
        with dpg.table(
            header_row=True,
            borders_innerH=True,
            borders_outerH=True,
            borders_innerV=True,
            borders_outerV=True,
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
