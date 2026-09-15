"""Dear PyGui desktop app: EMVA1288 sensor-noise / photon-transfer explorer.

Every curve comes from ``tools/emva_theory.py`` via
``opencam_gui.core.sensor_engine`` — the same functions
``tools/validate_emva_model.py`` uses to check a camera config against its
datasheet. The DSNU/PRNU tab runs the EMVA1288 spatial protocol (temporal
averaging, residual-temporal correction, DSNU1288, PRNU1288) on simulated
uniform-field stacks from those same functions. The dark-current-vs-temperature
helper is a documented read-only mirror of the formula in
``tools/apply_emva_noise.py`` (see ``opencam_gui.core.dark_current``).
"""

from __future__ import annotations

import numpy as np
import dearpygui.dearpygui as dpg

from opencam_gui.core import dark_current as dc
from opencam_gui.core import sensor_engine as se
from opencam_gui.core.camera import emva_summary
from opencam_gui.topics.sensor.scenarios import SCENARIOS, get_scenario
from opencam_gui.ui.desktop.base import DemoApp

MAP_SIZE = se.FPN_MAP_SIZE
_DSNU_MODELS = ("gaussian", "lognormal")
# Fixed display scales so slider amplitude is visible (auto-normalizing hid it).
DSNU_SCALE_E = 8.0  # matches the DSNU slider max
PRNU_SCALE_PCT = 5.0  # matches the PRNU slider max (0.05 fraction)
TEX_UPSAMPLE = 4
TEX_SIZE = MAP_SIZE * TEX_UPSAMPLE


def _upsample_nearest(arr: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1)


def _signed_gray_rgba(field: np.ndarray, half: float) -> list[float]:
    """Mid-gray = 0, black = -half, white = +half. Upsampled for visibility."""
    gray = np.clip(0.5 + 0.5 * np.asarray(field, dtype=np.float64) / max(half, 1e-12), 0.0, 1.0)
    gray = _upsample_nearest(gray, TEX_UPSAMPLE)
    rgba = np.ones(gray.shape + (4,), dtype=np.float64)
    rgba[..., 0] = gray
    rgba[..., 1] = gray
    rgba[..., 2] = gray
    return rgba.ravel().tolist()


def _histogram_series(
    values: np.ndarray, lo: float, hi: float, n_bins: int = 40
) -> tuple[list[float], list[float]]:
    finite = values[np.isfinite(values)].ravel()
    if finite.size == 0:
        return [0.0], [0.0]
    counts, edges = np.histogram(finite, bins=n_bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers.tolist(), counts.astype(np.float64).tolist()


class SensorApp(DemoApp):
    viewport_title = "Open Cam - Sensor / EMVA1288 Explorer"
    viewport_width = 1480
    viewport_height = 960
    window_label = "Sensor"
    control_panel_width = 430
    scenarios = SCENARIOS
    default_banner_title = "Interactive photon-transfer explorer"
    default_banner_body = (
        "Pick a camera recipe or drag the sliders. The PTC curve is computed by the "
        "real tools/emva_theory.py functions, the same ones the EMVA validator uses."
    )

    def init_state(self) -> None:
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
        self.prnu_std_fraction = 0.01
        self.dsnu_std_e = 0.3
        self.dsnu_model = "gaussian"
        self.n_measure_frames = 50
        self._measure_key: tuple | None = None
        self._unit_prnu: np.ndarray | None = None
        self._unit_dsnu: np.ndarray | None = None
        self._dark_point_theme: int | None = None
        self._measured_theme: int | None = None

    def get_scenario(self, scenario_id: str):
        return get_scenario(scenario_id)

    def apply_scenario_state(self, sc) -> None:
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
        self.prnu_std_fraction = sc.prnu_std_fraction
        self.dsnu_std_e = sc.dsnu_std_e
        self.dsnu_model = sc.dsnu_model
        self.n_measure_frames = sc.n_measure_frames

    def apply_recipe_state(self, model: dict) -> None:
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
        self.prnu_std_fraction = s.prnu_std_fraction
        self.dsnu_std_e = s.dsnu_std_e

    # --- controls --------------------------------------------------
    def read_controls(self) -> None:
        self.sigma_d_e = float(dpg.get_value("sigma_d_e"))
        self.K_e_per_DN = float(dpg.get_value("k_gain"))
        self.black_level_DN = float(dpg.get_value("black_level"))
        self.full_well_e = float(dpg.get_value("full_well"))
        self.use_poisson = bool(dpg.get_value("use_poisson"))
        self.dark_current_e_per_s = float(dpg.get_value("dark_rate"))
        self.temperature_c = float(dpg.get_value("temperature"))
        self.dark_activation_energy_eV = float(dpg.get_value("activation_energy"))
        self.integration_time_s = float(dpg.get_value("integration_time"))
        self.prnu_std_fraction = float(dpg.get_value("prnu_std"))
        self.dsnu_std_e = float(dpg.get_value("dsnu_std"))
        self.dsnu_model = str(dpg.get_value("dsnu_model"))
        self.n_measure_frames = int(dpg.get_value("n_frames"))

    def push_controls(self) -> None:
        dpg.set_value("sigma_d_e", self.sigma_d_e)
        dpg.set_value("k_gain", self.K_e_per_DN)
        dpg.set_value("black_level", self.black_level_DN)
        dpg.set_value("full_well", self.full_well_e)
        dpg.set_value("use_poisson", self.use_poisson)
        dpg.set_value("dark_rate", self.dark_current_e_per_s)
        dpg.set_value("temperature", self.temperature_c)
        dpg.set_value("activation_energy", self.dark_activation_energy_eV)
        dpg.set_value("integration_time", self.integration_time_s)
        dpg.set_value("prnu_std", self.prnu_std_fraction)
        dpg.set_value("dsnu_std", self.dsnu_std_e)
        dpg.set_value("dsnu_model", self.dsnu_model)
        dpg.set_value("n_frames", self.n_measure_frames)
        dpg.configure_item("verify_mu_e", max_value=self.full_well_e)

    def _mu_dark_e(self) -> float:
        return dc.dark_current_electrons_per_s(
            self.dark_current_e_per_s,
            self.temperature_c,
            self.dark_current_reference_temp_c,
            self.dark_current_doubling_per_c,
            self.dark_activation_energy_eV,
        ) * self.integration_time_s

    def _unit_maps(self) -> tuple[np.ndarray, np.ndarray]:
        """Stable N(0,1) fields so slider drags scale the same speckle pattern."""
        if self._unit_prnu is None or self._unit_dsnu is None:
            rng = np.random.default_rng(0)
            self._unit_prnu = rng.normal(0.0, 1.0, size=(MAP_SIZE, MAP_SIZE))
            self._unit_dsnu = rng.normal(0.0, 1.0, size=(MAP_SIZE, MAP_SIZE))
        return self._unit_prnu, self._unit_dsnu

    def _fpn_key(self) -> tuple:
        return (
            round(self.prnu_std_fraction, 6),
            round(self.dsnu_std_e, 6),
            self.dsnu_model,
            int(self.n_measure_frames),
            round(self.sigma_d_e, 6),
            round(self.K_e_per_DN, 6),
            round(self.black_level_DN, 6),
            round(self.full_well_e, 3),
            bool(self.use_poisson),
            round(self._mu_dark_e(), 6),
        )

    def _run_verify(self, _sender=None, _app_data=None, _user_data=None) -> None:
        self.read_controls()
        mu_dark_e = self._mu_dark_e()
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

    def _run_emva_measure(self, _sender=None, _app_data=None, _user_data=None) -> None:
        self.read_controls()
        mu_dark_e = self._mu_dark_e()
        measured = se.measure_emva1288(
            prnu_std_fraction=self.prnu_std_fraction,
            dsnu_std_e=self.dsnu_std_e,
            dark_mean_e=mu_dark_e,
            dsnu_model=self.dsnu_model,
            sigma_d_e=self.sigma_d_e,
            K_e_per_DN=self.K_e_per_DN,
            black_level_DN=self.black_level_DN,
            full_well_e=self.full_well_e,
            use_poisson=self.use_poisson,
            n_frames=self.n_measure_frames,
            seed=0,
        )
        self._measure_key = self._fpn_key()
        dpg.set_value(
            "spatial_measured",
            [measured.spatial_mu_e.tolist(), measured.spatial_std_measured_e.tolist()],
        )
        dpg.set_value(
            "fpn_measure_text",
            (
                f"EMVA1288 protocol on {measured.n_frames} frames of "
                f"{measured.height}x{measured.width} (50% well = {measured.mu_50_e:.0f} e-)\n"
                f"  Uncorrected spatial std of dark mean: {measured.uncorrected_dark_std_e:.3f} e-  "
                f"(residual temporal ~ {measured.residual_temporal_std_e:.3f} e- = sigma_d/sqrt(L))\n"
                f"  DSNU1288 = {measured.dsnu1288_e:.3f} e-   (config {measured.dsnu_config_e:.3f} e-)\n"
                f"  PRNU1288 = {100.0 * measured.prnu1288:.2f}%   "
                f"(config {100.0 * measured.prnu_config:.2f}%)"
            ),
        )

    # --- compute + draw ---------------------------------------------
    def refresh(self) -> None:
        self.read_controls()
        mu_dark_e = self._mu_dark_e()

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
                f"{self.integration_time_s * 1000:.1f} ms)\n"
                f"FPN: DSNU = {self.dsnu_std_e:.2f} e- ({self.dsnu_model}), "
                f"PRNU = {100.0 * self.prnu_std_fraction:.2f}%  "
                f"-- PTC is temporal-only; open the DSNU/PRNU tab for spatial noise"
            ),
        )

        preview = se.fpn_preview(
            prnu_std_fraction=self.prnu_std_fraction,
            dsnu_std_e=self.dsnu_std_e,
            dark_mean_e=mu_dark_e,
            dsnu_model=self.dsnu_model,
            full_well_e=self.full_well_e,
            seed=0,
        )
        unit_prnu, unit_dsnu = self._unit_maps()
        # Scale a frozen speckle field so dragging PRNU/DSNU changes contrast,
        # not the pattern. Lognormal DSNU is not a linear scale of N(0,1).
        prnu_pct = unit_prnu * (self.prnu_std_fraction * 100.0)
        if self.dsnu_model == "lognormal":
            dsnu_e = preview.dsnu_e
        else:
            dsnu_e = unit_dsnu * self.dsnu_std_e
        dpg.set_value("dsnu_texture", _signed_gray_rgba(dsnu_e, DSNU_SCALE_E))
        dpg.set_value("prnu_texture", _signed_gray_rgba(prnu_pct, PRNU_SCALE_PCT))
        dpg.set_value(
            "dsnu_caption",
            (
                f"DSNU map  (fixed scale +/-{DSNU_SCALE_E:.0f} e-; mid-gray = 0)   "
                f"RMS = {float(np.std(dsnu_e)):.2f} e-"
            ),
        )
        dpg.set_value(
            "prnu_caption",
            (
                f"PRNU map  (fixed scale +/-{PRNU_SCALE_PCT:.0f}%; mid-gray = 0)   "
                f"RMS = {float(np.std(prnu_pct)):.2f}%"
            ),
        )

        dpg.set_value("spatial_total", [preview.mu_e.tolist(), preview.spatial_std_e.tolist()])
        dpg.set_value(
            "spatial_dsnu",
            [preview.mu_e.tolist(), np.full_like(preview.mu_e, self.dsnu_std_e).tolist()],
        )
        dpg.set_value("spatial_prnu", [preview.mu_e.tolist(), preview.prnu_term_e.tolist()])
        y_max = max(float(preview.spatial_std_e.max()), float(self.dsnu_std_e), 1e-3) * 1.25
        dpg.set_axis_limits("spatial_x", 0.0, float(self.full_well_e))
        dpg.set_axis_limits("spatial_y", 0.0, y_max)

        dsnu_hx, dsnu_hy = _histogram_series(dsnu_e, -DSNU_SCALE_E, DSNU_SCALE_E)
        prnu_hx, prnu_hy = _histogram_series(prnu_pct, -PRNU_SCALE_PCT, PRNU_SCALE_PCT)
        dpg.set_value("dsnu_hist", [dsnu_hx, dsnu_hy])
        dpg.set_value("prnu_hist", [prnu_hx, prnu_hy])
        dpg.set_axis_limits("dsnu_hist_x", -DSNU_SCALE_E, DSNU_SCALE_E)
        dpg.set_axis_limits("prnu_hist_x", -PRNU_SCALE_PCT, PRNU_SCALE_PCT)

        if self._measure_key != self._fpn_key():
            dpg.set_value("spatial_measured", [[], []])
            dpg.set_value(
                "fpn_measure_text",
                "Click Measure DSNU1288 / PRNU1288 to run the EMVA protocol on simulated stacks.",
            )

    # --- layout -----------------------------------------------------
    def register_themes(self) -> None:
        with dpg.theme() as dark_point_theme:
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, (255, 190, 60), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 190, 60), category=dpg.mvThemeCat_Plots)
        self._dark_point_theme = dark_point_theme

        with dpg.theme() as measured_theme:
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, (90, 200, 255), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_Line, (90, 200, 255), category=dpg.mvThemeCat_Plots)
        self._measured_theme = measured_theme

    def register_textures(self) -> None:
        gray = [0.5, 0.5, 0.5, 1.0] * (TEX_SIZE * TEX_SIZE)
        dpg.add_dynamic_texture(TEX_SIZE, TEX_SIZE, gray, tag="dsnu_texture")
        dpg.add_dynamic_texture(TEX_SIZE, TEX_SIZE, list(gray), tag="prnu_texture")

    def build_controls(self) -> None:
        dpg.add_text("Core controls")
        dpg.add_slider_float(
            tag="sigma_d_e", label="Read noise sigma_d", default_value=self.sigma_d_e,
            min_value=0.1, max_value=15.0, callback=self.on_control_change,
        )
        dpg.add_slider_float(
            tag="k_gain", label="Gain K (e-/DN)", default_value=self.K_e_per_DN,
            min_value=0.3, max_value=8.0, callback=self.on_control_change,
        )
        dpg.add_slider_float(
            tag="full_well", label="Full well (e-)", default_value=self.full_well_e,
            min_value=500.0, max_value=100000.0, callback=self.on_control_change,
        )
        dpg.add_slider_float(
            tag="black_level", label="Black level (DN)", default_value=self.black_level_DN,
            min_value=0.0, max_value=100.0, callback=self.on_control_change,
        )
        dpg.add_checkbox(
            tag="use_poisson", label="Poisson shot noise enabled",
            default_value=self.use_poisson, callback=self.on_control_change,
        )
        dpg.add_separator()
        dpg.add_text("Fixed-pattern noise (EMVA1288)")
        dpg.add_slider_float(
            tag="prnu_std", label="PRNU (fraction)",
            default_value=self.prnu_std_fraction, min_value=0.0, max_value=0.05,
            format="%.3f", callback=self.on_control_change,
        )
        dpg.add_slider_float(
            tag="dsnu_std", label="DSNU (e-)",
            default_value=self.dsnu_std_e, min_value=0.0, max_value=8.0,
            callback=self.on_control_change,
        )
        dpg.add_slider_int(
            tag="n_frames", label="Frames L to average",
            default_value=self.n_measure_frames, min_value=2, max_value=100,
            callback=self.on_control_change,
        )
        dpg.add_button(
            label="Measure DSNU1288 / PRNU1288", width=-1,
            callback=self._run_emva_measure,
        )
        dpg.add_text("", tag="fpn_measure_text", wrap=350)

        with dpg.group(tag="advanced_controls"):
            dpg.add_separator()
            dpg.add_text("Dark current / temperature")
            dpg.add_slider_float(
                tag="dark_rate", label="Dark current (e-/s)",
                default_value=self.dark_current_e_per_s, min_value=0.0, max_value=5.0,
                callback=self.on_control_change,
            )
            dpg.add_slider_float(
                tag="temperature", label="Temperature (C)", default_value=self.temperature_c,
                min_value=-10.0, max_value=80.0, callback=self.on_control_change,
            )
            dpg.add_slider_float(
                tag="activation_energy", label="Arrhenius Ea (eV)",
                default_value=self.dark_activation_energy_eV, min_value=0.0, max_value=1.0,
                callback=self.on_control_change,
            )
            dpg.add_slider_float(
                tag="integration_time", label="Exposure time (s)",
                default_value=self.integration_time_s, min_value=0.0001, max_value=2.0,
                callback=self.on_control_change,
            )
            dpg.add_combo(
                tag="dsnu_model", label="DSNU model",
                items=list(_DSNU_MODELS), default_value=self.dsnu_model,
                callback=self.on_control_change,
            )

    def build_footer(self) -> None:
        dpg.add_separator()
        dpg.add_text("Monte Carlo verification")
        dpg.add_slider_float(
            tag="verify_mu_e", label="mu_e to verify", default_value=2000.0,
            min_value=0.0, max_value=self.full_well_e,
        )
        dpg.add_button(label="Run Monte Carlo check", width=-1, callback=self._run_verify)
        dpg.add_text("", tag="verify_text", wrap=350)

    def build_content(self) -> None:
        with dpg.tab_bar():
            with dpg.tab(label="Photon transfer"):
                with dpg.plot(label="Photon transfer curve (log-log)", height=380, width=-1, tag="ptc_plot"):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="mean signal (DN)", tag="ptc_x", scale=dpg.mvPlotScale_Log10)
                    with dpg.plot_axis(dpg.mvYAxis, label="variance (DN^2)", tag="ptc_y", scale=dpg.mvPlotScale_Log10):
                        dpg.add_line_series([1.0], [1.0], label="Var(DN) vs mean(DN)", tag="ptc_series")
                        dark_pt = dpg.add_scatter_series([1.0], [1.0], label="dark floor (mu_e=0)", tag="ptc_dark_point")
                        dpg.bind_item_theme(dark_pt, self._dark_point_theme)

                with dpg.plot(label="SNR vs mean signal", height=340, width=-1, tag="snr_plot"):
                    dpg.add_plot_axis(dpg.mvXAxis, label="mean signal (DN)", tag="snr_x", scale=dpg.mvPlotScale_Log10)
                    with dpg.plot_axis(dpg.mvYAxis, label="SNR (dB)", tag="snr_y"):
                        dpg.add_line_series([1.0], [0.0], label="SNR", tag="snr_series")

            with dpg.tab(label="DSNU / PRNU"):
                with dpg.group(horizontal=True):
                    with dpg.group():
                        dpg.add_text("", tag="dsnu_caption")
                        dpg.add_image("dsnu_texture", width=TEX_SIZE, height=TEX_SIZE)
                    with dpg.group():
                        dpg.add_text("", tag="prnu_caption")
                        dpg.add_image("prnu_texture", width=TEX_SIZE, height=TEX_SIZE)

                with dpg.plot(
                    label="Spatial std vs signal (after infinite averaging)",
                    height=280, width=-1, tag="spatial_plot",
                ):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="mean signal (e-)", tag="spatial_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="spatial std (e-)", tag="spatial_y"):
                        dpg.add_line_series([0.0], [0.0], label="total  sqrt(DSNU^2 + (PRNU x mu)^2)", tag="spatial_total")
                        dpg.add_line_series([0.0], [0.0], label="DSNU floor", tag="spatial_dsnu")
                        dpg.add_line_series([0.0], [0.0], label="PRNU x mu", tag="spatial_prnu")
                        meas = dpg.add_scatter_series([], [], label="EMVA1288 measured", tag="spatial_measured")
                        dpg.bind_item_theme(meas, self._measured_theme)

                with dpg.group(horizontal=True):
                    with dpg.plot(label="DSNU histogram (e-)", height=180, width=460):
                        dpg.add_plot_axis(dpg.mvXAxis, label="offset (e-)", tag="dsnu_hist_x")
                        with dpg.plot_axis(dpg.mvYAxis, label="count"):
                            dpg.add_line_series([0.0], [0.0], tag="dsnu_hist")
                    with dpg.plot(label="PRNU histogram (% from mean)", height=180, width=460):
                        dpg.add_plot_axis(dpg.mvXAxis, label="gain - 1 (%)", tag="prnu_hist_x")
                        with dpg.plot_axis(dpg.mvYAxis, label="count"):
                            dpg.add_line_series([0.0], [0.0], tag="prnu_hist")


def run_app(scenario_id: str | None = None) -> None:
    SensorApp(scenario_id=scenario_id).run()
