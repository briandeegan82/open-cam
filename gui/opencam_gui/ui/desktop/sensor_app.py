"""Dear PyGui desktop app: EMVA1288 sensor-noise / photon-transfer explorer.

Every curve comes from ``tools/emva_theory.py`` via
``opencam_gui.core.sensor_engine`` — the same functions
``tools/validate_emva_model.py`` uses to check a camera config against its
datasheet. The dark-current-vs-temperature helper is a documented read-only
mirror of the formula in ``tools/apply_emva_noise.py`` (see
``opencam_gui.core.dark_current`` for the exact citation).
"""

from __future__ import annotations

import numpy as np
import dearpygui.dearpygui as dpg

from opencam_gui.core import dark_current as dc
from opencam_gui.core import sensor_engine as se
from opencam_gui.core.camera import emva_summary, load_camera_model
from opencam_gui.core.catalog import list_camera_recipes
from opencam_gui.topics.sensor.scenarios import SCENARIOS, get_scenario


class SensorApp:
    def __init__(self, scenario_id: str | None = None) -> None:
        self.sigma_d_e = 2.6
        self.K_e_per_DN = 4.77
        self.black_level_DN = 16.0
        self.full_well_e = 4800.0
        self.use_poisson = True
        self.dark_current_e_per_s = 0.15
        self.temperature_c = 20.0
        self.dark_current_reference_temp_c = 20.0
        self.dark_current_doubling_per_c = 6.0
        self.dark_activation_energy_eV = 0.0
        self.integration_time_s = 0.01
        self.camera_recipe_id: str | None = None
        self._presenter = False
        self._recipes = list_camera_recipes()
        self._scenario_title = "Interactive photon-transfer explorer"
        self._scenario_note = (
            "Pick a camera recipe or drag the sliders. The PTC curve is computed by the "
            "real tools/emva_theory.py functions, the same ones the EMVA validator uses."
        )
        if scenario_id:
            self._apply_scenario_fields(get_scenario(scenario_id))

    def _apply_scenario_fields(self, sc) -> None:
        self.sigma_d_e = sc.sigma_d_e
        self.K_e_per_DN = sc.K_e_per_DN
        self.black_level_DN = sc.black_level_DN
        self.full_well_e = sc.full_well_e
        self.use_poisson = sc.use_poisson
        self.dark_current_e_per_s = sc.dark_current_e_per_s
        self.temperature_c = sc.temperature_c
        self.dark_current_reference_temp_c = sc.dark_current_reference_temp_c
        self.dark_current_doubling_per_c = sc.dark_current_doubling_per_c
        self.dark_activation_energy_eV = sc.dark_activation_energy_eV
        self.integration_time_s = sc.integration_time_s
        self.camera_recipe_id = sc.camera_recipe_id
        self._scenario_title = sc.title
        self._scenario_note = f"{sc.teaching_point}\n{sc.notes}"

    # --- controls --------------------------------------------------
    def _read_controls(self) -> None:
        self.sigma_d_e = float(dpg.get_value("sigma_d_e"))
        self.K_e_per_DN = float(dpg.get_value("k_gain"))
        self.black_level_DN = float(dpg.get_value("black_level"))
        self.full_well_e = float(dpg.get_value("full_well"))
        self.use_poisson = bool(dpg.get_value("use_poisson"))
        self.dark_current_e_per_s = float(dpg.get_value("dark_rate"))
        self.temperature_c = float(dpg.get_value("temperature"))
        self.dark_activation_energy_eV = float(dpg.get_value("activation_energy"))
        self.integration_time_s = float(dpg.get_value("integration_time"))

    def _push_controls(self) -> None:
        dpg.set_value("sigma_d_e", self.sigma_d_e)
        dpg.set_value("k_gain", self.K_e_per_DN)
        dpg.set_value("black_level", self.black_level_DN)
        dpg.set_value("full_well", self.full_well_e)
        dpg.set_value("use_poisson", self.use_poisson)
        dpg.set_value("dark_rate", self.dark_current_e_per_s)
        dpg.set_value("temperature", self.temperature_c)
        dpg.set_value("activation_energy", self.dark_activation_energy_eV)
        dpg.set_value("integration_time", self.integration_time_s)
        dpg.configure_item("verify_mu_e", max_value=self.full_well_e)

    def _load_recipe(self, recipe_id: str) -> None:
        recipe = next((r for r in self._recipes if r.id == recipe_id), None)
        if recipe is None:
            return
        model = load_camera_model(recipe.path)
        s = emva_summary(model)
        self.sigma_d_e = s.sigma_d_e
        self.K_e_per_DN = s.K_e_per_DN
        self.black_level_DN = s.black_level_DN
        self.full_well_e = s.full_well_e
        self.use_poisson = s.use_poisson
        self.dark_current_e_per_s = s.dark_current_e_per_s
        self.temperature_c = s.temperature_c
        self.dark_current_reference_temp_c = s.dark_current_reference_temp_c
        self.dark_current_doubling_per_c = s.dark_current_doubling_per_c
        self.dark_activation_energy_eV = s.dark_activation_energy_eV
        self.integration_time_s = s.integration_time_s
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

    def _run_verify(self, _sender=None, _app_data=None, _user_data=None) -> None:
        mu_dark_e = dc.dark_current_electrons_per_s(
            self.dark_current_e_per_s,
            self.temperature_c,
            self.dark_current_reference_temp_c,
            self.dark_current_doubling_per_c,
            self.dark_activation_energy_eV,
        ) * self.integration_time_s
        mu_e = float(dpg.get_value("verify_mu_e"))
        v = se.verify_against_monte_carlo(
            mu_e=mu_e,
            sigma_d_e=self.sigma_d_e,
            K_e_per_DN=self.K_e_per_DN,
            black_level_DN=self.black_level_DN,
            full_well_e=self.full_well_e,
            use_poisson=self.use_poisson,
            mu_dark_e=mu_dark_e,
            n_trials=20000,
            seed=0,
        )
        dpg.set_value(
            "verify_text",
            (
                f"mu_e = {v.mu_e:.1f} e-  (20000-trial Monte Carlo vs closed-form theory)\n"
                f"  theory:  mean = {v.theory_mean_dn:8.3f} DN   var = {v.theory_var_dn:8.4f} DN^2\n"
                f"  Monte Carlo: mean = {v.mc_mean_dn:8.3f} DN   var = {v.mc_var_dn:8.4f} DN^2"
            ),
        )

    # --- compute + draw ---------------------------------------------
    def refresh(self) -> None:
        self._read_controls()
        mu_dark_e = dc.dark_current_electrons_per_s(
            self.dark_current_e_per_s,
            self.temperature_c,
            self.dark_current_reference_temp_c,
            self.dark_current_doubling_per_c,
            self.dark_activation_energy_eV,
        ) * self.integration_time_s

        curve = se.photon_transfer_curve(
            sigma_d_e=self.sigma_d_e,
            K_e_per_DN=self.K_e_per_DN,
            black_level_DN=self.black_level_DN,
            full_well_e=self.full_well_e,
            use_poisson=self.use_poisson,
            mu_dark_e=mu_dark_e,
            n_points=200,
        )

        dpg.set_value("ptc_series", [curve.mean_dn.tolist(), curve.var_dn.tolist()])
        dpg.set_value("ptc_dark_point", [[curve.dark_mean_dn], [max(curve.dark_var_dn, 1e-6)]])
        dpg.set_axis_limits("ptc_x", float(curve.mean_dn.min()) * 0.8, float(curve.mean_dn.max()) * 1.2)
        dpg.set_axis_limits(
            "ptc_y",
            max(1e-4, float(min(curve.var_dn.min(), curve.dark_var_dn)) * 0.5),
            float(curve.var_dn.max()) * 2.0,
        )

        signal_dn = curve.mean_dn - self.black_level_DN
        noise_dn = np.sqrt(np.maximum(curve.var_dn, 1e-12))
        with np.errstate(divide="ignore", invalid="ignore"):
            snr_db = 20.0 * np.log10(np.clip(signal_dn, 1e-6, None) / noise_dn)
        dpg.set_value("snr_series", [curve.mean_dn.tolist(), snr_db.tolist()])
        dpg.set_axis_limits("snr_x", float(curve.mean_dn.min()) * 0.8, float(curve.mean_dn.max()) * 1.2)

        crossover_dn = se.mean_dn_linear(
            curve.shot_read_crossover_e, self.K_e_per_DN, self.black_level_DN
        )
        dpg.set_value(
            "status_text",
            (
                f"K = {self.K_e_per_DN:.3f} e-/DN   sigma_d = {self.sigma_d_e:.2f} e-   "
                f"full well = {self.full_well_e:.0f} e-   black = {self.black_level_DN:.1f} DN\n"
                f"Shot/read crossover: {curve.shot_read_crossover_e:.1f} e- "
                f"(~{crossover_dn:.1f} DN above black) -- read-noise-limited below this signal\n"
                f"Dark floor: mean {curve.dark_mean_dn:.2f} DN, var {curve.dark_var_dn:.4f} DN^2   "
                f"(mu_dark = {mu_dark_e:.3f} e- at {self.temperature_c:.0f} C, "
                f"{self.integration_time_s * 1000:.1f} ms)"
            ),
        )
        dpg.set_value("banner_title", self._scenario_title)
        dpg.set_value("banner_body", self._scenario_note)

    def run(self) -> None:
        dpg.create_context()
        dpg.create_viewport(title="Open Cam - Sensor / EMVA1288 Explorer", width=1400, height=900)

        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6)
        dpg.bind_theme(global_theme)

        with dpg.theme() as dark_point_theme:
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, (255, 190, 60), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 190, 60), category=dpg.mvThemeCat_Plots)

        with dpg.window(tag="primary", label="Sensor"):
            with dpg.child_window(tag="banner_panel", height=72, border=True):
                dpg.add_text(self._scenario_title, tag="banner_title")
                dpg.add_text(self._scenario_note, tag="banner_body", wrap=1000)

            with dpg.group(horizontal=True):
                with dpg.child_window(width=430, border=True):
                    dpg.add_checkbox(
                        tag="presenter_mode", label="Presenter mode (hide advanced)",
                        default_value=False, callback=self._set_presenter,
                    )
                    dpg.add_separator()
                    dpg.add_text("Camera recipe")
                    dpg.add_combo(
                        tag="recipe_combo", items=["Custom", *[r.id for r in self._recipes]],
                        default_value=self.camera_recipe_id or "Custom", callback=self._on_recipe_change,
                    )
                    dpg.add_separator()
                    dpg.add_text("Core controls")
                    dpg.add_slider_float(
                        tag="sigma_d_e", label="Read noise sigma_d", default_value=self.sigma_d_e,
                        min_value=0.1, max_value=15.0, callback=lambda *_: self.refresh(),
                    )
                    dpg.add_slider_float(
                        tag="k_gain", label="Gain K (e-/DN)", default_value=self.K_e_per_DN,
                        min_value=0.3, max_value=8.0, callback=lambda *_: self.refresh(),
                    )
                    dpg.add_slider_float(
                        tag="full_well", label="Full well (e-)", default_value=self.full_well_e,
                        min_value=500.0, max_value=100000.0, callback=lambda *_: self.refresh(),
                    )
                    dpg.add_slider_float(
                        tag="black_level", label="Black level (DN)", default_value=self.black_level_DN,
                        min_value=0.0, max_value=100.0, callback=lambda *_: self.refresh(),
                    )
                    dpg.add_checkbox(
                        tag="use_poisson", label="Poisson shot noise enabled",
                        default_value=self.use_poisson, callback=lambda *_: self.refresh(),
                    )

                    with dpg.group(tag="advanced_controls"):
                        dpg.add_separator()
                        dpg.add_text("Dark current / temperature")
                        dpg.add_slider_float(
                            tag="dark_rate", label="Dark current (e-/s)",
                            default_value=self.dark_current_e_per_s, min_value=0.0, max_value=5.0,
                            callback=lambda *_: self.refresh(),
                        )
                        dpg.add_slider_float(
                            tag="temperature", label="Temperature (C)", default_value=self.temperature_c,
                            min_value=-10.0, max_value=80.0, callback=lambda *_: self.refresh(),
                        )
                        dpg.add_slider_float(
                            tag="activation_energy", label="Arrhenius Ea (eV)",
                            default_value=self.dark_activation_energy_eV, min_value=0.0, max_value=1.0,
                            callback=lambda *_: self.refresh(),
                        )
                        dpg.add_slider_float(
                            tag="integration_time", label="Exposure time (s)",
                            default_value=self.integration_time_s, min_value=0.0001, max_value=2.0,
                            callback=lambda *_: self.refresh(),
                        )

                    dpg.add_separator()
                    dpg.add_text("Lecture scenarios")
                    for sid, sc in SCENARIOS.items():
                        dpg.add_button(label=sc.title, width=-1, user_data=sid, callback=self._apply_scenario)
                    dpg.add_separator()
                    dpg.add_text("", tag="status_text", wrap=350)
                    dpg.add_separator()
                    dpg.add_text("Monte Carlo verification")
                    dpg.add_slider_float(
                        tag="verify_mu_e", label="mu_e to verify", default_value=2000.0,
                        min_value=0.0, max_value=self.full_well_e,
                    )
                    dpg.add_button(label="Run Monte Carlo check", width=-1, callback=self._run_verify)
                    dpg.add_text("", tag="verify_text", wrap=350)

                with dpg.child_window(border=False):
                    with dpg.plot(label="Photon transfer curve (log-log)", height=380, width=-1, tag="ptc_plot"):
                        dpg.add_plot_legend()
                        dpg.add_plot_axis(dpg.mvXAxis, label="mean signal (DN)", tag="ptc_x", scale=dpg.mvPlotScale_Log10)
                        with dpg.plot_axis(dpg.mvYAxis, label="variance (DN^2)", tag="ptc_y", scale=dpg.mvPlotScale_Log10):
                            dpg.add_line_series([1.0], [1.0], label="Var(DN) vs mean(DN)", tag="ptc_series")
                            dark_pt = dpg.add_scatter_series([1.0], [1.0], label="dark floor (mu_e=0)", tag="ptc_dark_point")
                            dpg.bind_item_theme(dark_pt, dark_point_theme)

                    with dpg.plot(label="SNR vs mean signal", height=340, width=-1, tag="snr_plot"):
                        dpg.add_plot_axis(dpg.mvXAxis, label="mean signal (DN)", tag="snr_x", scale=dpg.mvPlotScale_Log10)
                        with dpg.plot_axis(dpg.mvYAxis, label="SNR (dB)", tag="snr_y"):
                            dpg.add_line_series([1.0], [0.0], label="SNR", tag="snr_series")

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
    SensorApp(scenario_id=scenario_id).run()
