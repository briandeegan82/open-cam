"""Dear PyGui desktop app: colour and the ISP.

Three tabs. The pipeline tab walks a spectrally-rendered ColorChecker through
mosaic, demosaic, white balance, CCM and sRGB with every stage toggleable and
previewed. The demosaic tab puts bilinear and Malvar side by side against known
ground truth. The spectra tab overlays the illuminant, the surface reflectance
and the camera's QE curves, which is where metamerism stops being an assertion
and becomes a picture.
"""

from __future__ import annotations

import dearpygui.dearpygui as dpg
import numpy as np

from opencam_gui.core import isp_engine as ie
from opencam_gui.topics.isp.scenarios import SCENARIOS, get_scenario
from opencam_gui.ui.desktop.base import DemoApp

STAGE_TEX = 160
COMPARE_TEX = 192


def _rgba_flat(img: np.ndarray) -> list[float]:
    a = np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    rgba = np.ones(a.shape[:2] + (4,), dtype=np.float64)
    rgba[..., :3] = a[..., :3]
    return rgba.ravel().tolist()


def _fit_to(arr: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize, so demosaic artefacts stay visible."""
    a = np.asarray(arr, dtype=np.float64)
    ys = np.clip(np.arange(size) * a.shape[0] // size, 0, a.shape[0] - 1)
    xs = np.clip(np.arange(size) * a.shape[1] // size, 0, a.shape[1] - 1)
    return a[np.ix_(ys, xs)] if a.ndim == 2 else a[np.ix_(ys, xs)][:, :, :3]


def _heat(gray: np.ndarray) -> np.ndarray:
    """Black-red-yellow-white ramp for error maps."""
    g = np.clip(np.asarray(gray, dtype=np.float64), 0.0, 1.0)
    return np.stack(
        [
            np.clip(g * 3.0, 0.0, 1.0),
            np.clip(g * 3.0 - 1.0, 0.0, 1.0),
            np.clip(g * 3.0 - 2.0, 0.0, 1.0),
        ],
        axis=2,
    )


class IspApp(DemoApp):
    viewport_title = "Open Cam - Colour and the ISP"
    viewport_width = 1560
    viewport_height = 1000
    window_label = "Colour / ISP"
    control_panel_width = 430
    scenarios = SCENARIOS
    default_banner_title = "From spectra to a picture"
    default_banner_body = (
        "The sensor measures three numbers per pixel in a basis nobody chose for its colour "
        "properties. Turning that into a picture takes a mosaic, a reconstruction, a white "
        "balance, a matrix and a transfer function  -  each with its own failure mode."
    )

    def init_state(self) -> None:
        self.illuminant_id = "D65"
        self.demosaic_method = "bilinear"
        self.wb_method = "white_patch"
        self.bayer_pattern = "RGGB"
        self.patch_index = 21
        self.stages: set[str] = set(ie.STAGES)

        self.qe_paths = dict(ie.DEFAULT_QE_PATHS)

        self._illuminants = ie.list_illuminants()
        self._chart: ie.Chart | None = None
        self._curve_themes: dict[str, int] = {}

    def get_scenario(self, scenario_id: str):
        return get_scenario(scenario_id)

    def apply_recipe_state(self, model: dict) -> None:
        """Take the camera's own QE curves.

        Different sensors miss the Luther condition by different amounts, so
        swapping the recipe genuinely changes how accurate the colour can be.
        """
        self.qe_paths = ie.qe_paths_from_model(model)

    def apply_scenario_state(self, sc) -> None:
        self.illuminant_id = sc.illuminant_id
        self.stages = set(sc.stages)
        self.demosaic_method = sc.demosaic_method
        self.wb_method = sc.wb_method
        self.bayer_pattern = sc.bayer_pattern
        self.patch_index = sc.patch_index
        # Alternative CFAs (RCCB, CMY) are different QE curves, so a scenario
        # that names a recipe has to actually load it -- otherwise the combo
        # would change and the spectra would not.
        self.qe_paths = ie.qe_paths_for_recipe(sc.camera_recipe_id)

    # --- controls --------------------------------------------------
    def read_controls(self) -> None:
        self.illuminant_id = dpg.get_value("illuminant")
        self.demosaic_method = dpg.get_value("demosaic_method")
        self.wb_method = dpg.get_value("wb_method")
        self.bayer_pattern = dpg.get_value("bayer_pattern")
        self.patch_index = int(dpg.get_value("patch_index"))
        self.stages = {s for s in ie.STAGES if dpg.get_value(f"stage_{s}")}

    def push_controls(self) -> None:
        dpg.set_value("illuminant", self.illuminant_id)
        dpg.set_value("demosaic_method", self.demosaic_method)
        dpg.set_value("wb_method", self.wb_method)
        dpg.set_value("bayer_pattern", self.bayer_pattern)
        dpg.set_value("patch_index", self.patch_index)
        for s in ie.STAGES:
            dpg.set_value(f"stage_{s}", s in self.stages)

    # --- compute + draw ---------------------------------------------
    def refresh(self) -> None:
        self.read_controls()
        chart = ie.load_chart(illuminant_id=self.illuminant_id, qe_paths=self.qe_paths)
        self._chart = chart

        result = ie.run_isp(
            chart,
            ie.IspConfig(
                enabled=set(self.stages),
                bayer_pattern=self.bayer_pattern,
                demosaic_method=self.demosaic_method,
                wb_method=self.wb_method,
            ),
        )
        accuracy = ie.colour_accuracy(chart, result)

        self._draw_stages(result)
        self._draw_accuracy(chart, result, accuracy)
        self._draw_demosaic()
        self._draw_spectra(chart)
        self._draw_status(chart, result, accuracy)

    def _draw_stages(self, result: ie.IspResult) -> None:
        for stage in result.stages:
            dpg.set_value(f"tex_{stage.id}", _rgba_flat(_fit_to(stage.image, STAGE_TEX)))
            dpg.set_value(f"note_{stage.id}", stage.note)
            label = stage.label if stage.applied else f"{stage.label}  (off)"
            dpg.set_value(f"label_{stage.id}", label)
        dpg.set_value("tex_reference", _rgba_flat(_fit_to(result.reference_display, STAGE_TEX)))

    def _draw_accuracy(self, chart: ie.Chart, result: ie.IspResult, accuracy: ie.ColourAccuracy) -> None:
        idx = list(range(1, accuracy.delta_e_2000.size + 1))
        dpg.set_value("de_series", [idx, accuracy.delta_e_2000.tolist()])
        dpg.set_value("de_good", [[0, len(idx) + 1], [2.0, 2.0]])
        dpg.set_axis_limits("de_y", 0.0, max(6.0, float(accuracy.max_delta_e) * 1.15))

        ccm = result.ccm
        dpg.set_value("ccm_text", "\n".join("   ".join(f"{v:+7.3f}" for v in row) for row in ccm))
        dpg.set_value("ccm_colsums", "column sums   " + "   ".join(f"{v:+7.3f}" for v in ccm.sum(axis=0)))

    def _draw_demosaic(self) -> None:
        cmp_ = ie.compare_demosaic(pattern=self.bayer_pattern, size=COMPARE_TEX)
        dpg.set_value("tex_bilinear", _rgba_flat(_fit_to(cmp_.bilinear, COMPARE_TEX)))
        dpg.set_value("tex_malvar", _rgba_flat(_fit_to(cmp_.malvar, COMPARE_TEX)))
        peak = max(float(cmp_.difference.max()), 1e-6)
        dpg.set_value("tex_diff", _rgba_flat(_heat(_fit_to(cmp_.difference / peak, COMPARE_TEX))))

        ratio = cmp_.edge_bilinear_rmse / max(cmp_.edge_malvar_rmse, 1e-9)
        dpg.set_value(
            "demosaic_caption",
            (
                f"RMSE against ground truth  -  bilinear {cmp_.bilinear_rmse:.4f}, "
                f"Malvar {cmp_.malvar_rmse:.4f}.\n"
                f"On edges only: bilinear {cmp_.edge_bilinear_rmse:.4f}, "
                f"Malvar {cmp_.edge_malvar_rmse:.4f}  -  Malvar is {ratio:.2f}x better where "
                f"it counts.\n"
                "Malvar earns that by assuming chroma varies more slowly than luminance, "
                "which is true of real scenes. Feed it saturated, independently varying "
                "channels and bilinear wins instead: the advantage is a statement about "
                "images, not about arithmetic.\n"
                f"The difference map is scaled to its own peak of {peak:.3f}; the errors sit "
                "on edges, which is exactly where the eye looks."
            ),
        )

    def _draw_spectra(self, chart: ie.Chart) -> None:
        overlay = ie.spectral_overlay(chart, self.patch_index)
        wl = overlay.wavelength_nm.tolist()
        dpg.set_value("spd_series", [wl, overlay.illuminant.tolist()])
        dpg.set_value("refl_series", [wl, overlay.reflectance.tolist()])
        for i, name in enumerate(("r", "g", "b")):
            dpg.set_value(f"qe_{name}", [wl, overlay.qe_rgb[i].tolist()])
            dpg.set_value(f"prod_{name}", [wl, overlay.product_rgb[i].tolist()])

        dpg.set_value(
            "spectra_caption",
            (
                f"Patch: {overlay.patch_name}   illuminant {chart.illuminant_id} "
                f"({overlay.cct_k:.0f} K correlated colour temperature).\n"
                "The top plot is the three curves whose product decides the answer; the "
                "bottom plot is that product, one curve per channel. A channel can only "
                "respond where all three overlap  -  which is why a lamp with gaps in its "
                "spectrum renders some surfaces so badly, and why no amount of downstream "
                "processing recovers them.\n"
                f"This camera's QE curves miss the Luther condition by "
                f"{chart.luther_error * 100:.1f}%. That residual is the part of the colour "
                "error no 3x3 matrix can fit away."
            ),
        )

    def _draw_status(self, chart: ie.Chart, result: ie.IspResult, accuracy: ie.ColourAccuracy) -> None:
        g = result.wb_gains
        cast = accuracy.neutral_cast_rgb
        verdict = (
            "excellent" if accuracy.mean_delta_e < 2 else "usable" if accuracy.mean_delta_e < 5 else "clearly wrong"
        )
        dpg.set_value(
            "status_text",
            (
                f"Illuminant {chart.illuminant_id}\n"
                f"Stages on: {', '.join(sorted(self.stages)) or 'none'}\n"
                f"White balance gains  R {g[0]:.2f}  G {g[1]:.2f}  B {g[2]:.2f}\n"
                f"Neutral patches land at  {cast[0]:.3f}, {cast[1]:.3f}, {cast[2]:.3f}\n"
                f"  (1.000, 1.000, 1.000 means the greys really are grey)\n"
                f"Mean delta-E 2000 = {accuracy.mean_delta_e:.2f}  ({verdict})\n"
                f"Worst patch: {accuracy.worst_patch} at {accuracy.max_delta_e:.2f}"
            ),
        )

    # --- layout -----------------------------------------------------
    def register_themes(self) -> None:
        for name, colour in (
            ("r", (235, 90, 90)),
            ("g", (90, 205, 110)),
            ("b", (100, 140, 245)),
            ("spd", (240, 200, 90)),
            ("refl", (200, 200, 200)),
        ):
            with dpg.theme() as theme, dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, colour, category=dpg.mvThemeCat_Plots)
            self._curve_themes[name] = theme

    def register_textures(self) -> None:
        blank = [0.0, 0.0, 0.0, 1.0] * (STAGE_TEX * STAGE_TEX)
        for stage_id in ("scene", *ie.STAGES, "reference"):
            dpg.add_dynamic_texture(STAGE_TEX, STAGE_TEX, list(blank), tag=f"tex_{stage_id}")
        big = [0.0, 0.0, 0.0, 1.0] * (COMPARE_TEX * COMPARE_TEX)
        for tag in ("tex_bilinear", "tex_malvar", "tex_diff"):
            dpg.add_dynamic_texture(COMPARE_TEX, COMPARE_TEX, list(big), tag=tag)

    def build_controls(self) -> None:
        dpg.add_text("Scene")
        dpg.add_combo(
            tag="illuminant",
            label="Illuminant",
            items=self._illuminants,
            default_value=self.illuminant_id,
            callback=self.on_control_change,
        )

        dpg.add_separator()
        dpg.add_text("ISP stages")
        for s in ie.STAGES:
            dpg.add_checkbox(
                tag=f"stage_{s}",
                label=ie.STAGE_LABELS[s],
                default_value=s in self.stages,
                callback=self.on_control_change,
            )

        dpg.add_separator()
        dpg.add_combo(
            tag="demosaic_method",
            label="Demosaic",
            items=list(ie.DEMOSAIC_METHODS),
            default_value=self.demosaic_method,
            callback=self.on_control_change,
        )
        dpg.add_combo(
            tag="wb_method",
            label="White balance",
            items=list(ie.WB_METHODS),
            default_value=self.wb_method,
            callback=self.on_control_change,
        )

        with dpg.group(tag="advanced_controls"):
            dpg.add_separator()
            dpg.add_text("Advanced")
            dpg.add_combo(
                tag="bayer_pattern",
                label="CFA pattern",
                items=["RGGB", "BGGR", "GRBG", "GBRG"],
                default_value=self.bayer_pattern,
                callback=self.on_control_change,
            )
            dpg.add_slider_int(
                tag="patch_index",
                label="Spectral overlay patch",
                default_value=self.patch_index,
                min_value=0,
                max_value=23,
                callback=self.on_control_change,
            )

    def build_content(self) -> None:
        with dpg.tab_bar():
            with dpg.tab(label="ISP pipeline"):
                self._build_pipeline_tab()
            with dpg.tab(label="Demosaic comparison"):
                self._build_demosaic_tab()
            with dpg.tab(label="Spectra and metamerism"):
                self._build_spectra_tab()

    def _build_pipeline_tab(self) -> None:
        dpg.add_text("Every stage, in order, with the image as it leaves that stage")
        order = ("scene", *ie.STAGES)
        for row in (order[:3], order[3:]):
            with dpg.group(horizontal=True):
                for stage_id in row:
                    with dpg.group():
                        dpg.add_text("", tag=f"label_{stage_id}")
                        dpg.add_image(f"tex_{stage_id}", width=STAGE_TEX, height=STAGE_TEX)
                        dpg.add_text("", tag=f"note_{stage_id}", wrap=STAGE_TEX + 40)

        dpg.add_separator()
        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text("Colorimetric reference (what it should look like)")
                dpg.add_image("tex_reference", width=STAGE_TEX, height=STAGE_TEX)
            with dpg.group():
                dpg.add_text("Fitted colour correction matrix")
                dpg.add_text("", tag="ccm_text")
                dpg.add_text("", tag="ccm_colsums")
                dpg.add_text(
                    "Column sums of 1.000 would mean neutrals pass through untouched;\n"
                    "anything else means the matrix is also doing white balance's job.",
                    wrap=420,
                )
            with dpg.plot(label="Per-patch colour error (CIEDE2000)", height=240, width=-1):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="ColorChecker patch", tag="de_x")
                with dpg.plot_axis(dpg.mvYAxis, label="delta-E 2000", tag="de_y"):
                    dpg.add_bar_series([0], [0], label="measured", tag="de_series", weight=0.7)
                    dpg.add_line_series([0], [0], label="delta-E 2 (good camera)", tag="de_good")

    def _build_demosaic_tab(self) -> None:
        with dpg.group(horizontal=True):
            for tag, label in (
                ("tex_bilinear", "Bilinear"),
                ("tex_malvar", "Malvar-He-Cutler"),
                ("tex_diff", "Where they disagree"),
            ):
                with dpg.group():
                    dpg.add_text(label)
                    dpg.add_image(tag, width=380, height=380)
        dpg.add_separator()
        dpg.add_text("", tag="demosaic_caption", wrap=1180)

    def _build_spectra_tab(self) -> None:
        with dpg.plot(label="Illuminant, reflectance and QE (each normalised)", height=320, width=-1):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="wavelength (nm)", tag="spec_x")
            with dpg.plot_axis(dpg.mvYAxis, label="relative", tag="spec_y"):
                spd = dpg.add_line_series([0], [0], label="illuminant SPD", tag="spd_series")
                dpg.bind_item_theme(spd, self._curve_themes["spd"])
                refl = dpg.add_line_series([0], [0], label="patch reflectance", tag="refl_series")
                dpg.bind_item_theme(refl, self._curve_themes["refl"])
                for name, label in (("r", "QE red"), ("g", "QE green"), ("b", "QE blue")):
                    s = dpg.add_line_series([0], [0], label=label, tag=f"qe_{name}")
                    dpg.bind_item_theme(s, self._curve_themes[name])

        with dpg.plot(label="Their product: what each channel actually collects", height=320, width=-1):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="wavelength (nm)", tag="prod_x")
            with dpg.plot_axis(dpg.mvYAxis, label="response", tag="prod_y"):
                for name, label in (("r", "red"), ("g", "green"), ("b", "blue")):
                    s = dpg.add_line_series([0], [0], label=label, tag=f"prod_{name}")
                    dpg.bind_item_theme(s, self._curve_themes[name])

        dpg.add_separator()
        dpg.add_text("", tag="spectra_caption", wrap=1180)


def run_app(scenario_id: str | None = None) -> None:
    IspApp(scenario_id=scenario_id).run()
