"""Dear PyGui desktop app: run one IQ-lab test scene through one camera recipe.

A thin front end over ``tools/run_recipe_scorecard.py``: "Run" renders the
scene with pbrt for the recipe's optics, runs the recipe's full sensor chain
(QE, IRCF, EMVA noise, CFA, HDR, demosaic, white balance, CCM) and scores it,
in a background thread. The preview shows the processed image with the
measurement ROIs drawn on top; the table lists the scorecard metrics.
"""

from __future__ import annotations

import queue
import tempfile
import threading
from pathlib import Path

import dearpygui.dearpygui as dpg
import numpy as np
from PIL import Image

from opencam_gui.core.camera import load_camera_model
from opencam_gui.core.repo import import_tool, pbrt_available
from opencam_gui.topics.iq_lab.scenarios import SCENARIOS, get_scenario
from opencam_gui.ui.desktop.base import DemoApp

PREVIEW_W, PREVIEW_H = 720, 480

SCENE_LABELS = {
    "diorama": "Diorama (slanted edge, dead leaves, ColorChecker)",
    "hdr": "HDR chart (SNR, DR, CDP)",
    "skin": "Skin-tone chart, D65 (ΔE00)",
    "flare": "Black-hole flare target (veiling glare)",
}
_SCENE_BY_LABEL = {v: k for k, v in SCENE_LABELS.items()}

#: Scorecard metric keys shown for each scene.
SCENE_METRICS = {
    "hdr": ("dr_snr1_db", "dr_snr10_db", "snr_at_18pct_db", "cdp_min_level_db_at_0p9"),
    "diorama": ("edge_mtf50_cy_px", "edge_acutance", "texture_acutance"),
    "skin": ("skin_de00_mean", "skin_de00_max"),
    "flare": ("veiling_glare_pct",),
}

ROI_COLOURS = {
    "slanted_edge": (255, 170, 40, 255),
    "dead_leaves": (90, 200, 255, 255),
    "white": (255, 255, 255, 255),
    "patch": (120, 255, 120, 255),
    "low": (90, 200, 255, 255),
    "high": (255, 170, 40, 255),
    "hole": (255, 80, 80, 255),
}


def _blank_rgba(w: int, h: int) -> list[float]:
    return [0.08, 0.08, 0.09, 1.0] * (w * h)


def display_rgba(img: np.ndarray, w: int = PREVIEW_W, h: int = PREVIEW_H) -> list[float]:
    """Flat RGBA for a dynamic texture: linear RGB -> exposure to the 99.5th percentile + sRGB-ish gamma;
    a 2-D HDR electron map (relative to saturation) -> log scale over 120 dB."""
    if img.ndim == 2:
        v = np.clip(1.0 + np.log10(np.clip(img, 1e-6, None)) / 6.0, 0.0, 1.0)
        rgb = np.repeat(v[..., None], 3, axis=2)
    else:
        y = img @ np.array([0.2126, 0.7152, 0.0722])
        scale = float(np.percentile(y, 99.5)) or 1.0
        rgb = np.clip(img / scale, 0.0, 1.0) ** (1 / 2.2)
    pil = Image.fromarray((rgb * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    rgba = np.ones((h, w, 4))
    rgba[..., :3] = np.asarray(pil, dtype=np.float64) / 255.0
    return rgba.ravel().tolist()


def scene_rois(scene: str, render: dict, row: dict | None = None, image: np.ndarray | None = None) -> list:
    """``[(kind, (x0, y0, x1, y1)), ...]`` in render-raster pixels for the overlay."""
    rois = render.get("rois", {})
    if scene == "diorama":
        return [(k, rois[k]) for k in ("slanted_edge", "dead_leaves", "white") if rois.get(k)]
    if scene == "skin":
        return [("patch", r) for r in rois.get("patches", []) if r]
    if scene == "hdr":
        out = []
        for p in render["meta"]["patches"]:
            out += [("low", p["low"]["roi_xyxy"]), ("high", p["high"]["roi_xyxy"])]
        return out
    if scene == "flare" and image is not None:
        flare = import_tool("iqlab.flare")
        y = image @ np.array([0.2126, 0.7152, 0.0722])
        return [("hole", (h.x - 4, h.y - 4, h.x + 4, h.y + 4)) for h in flare.black_hole_glare(y)]
    return []


class IqLabApp(DemoApp):
    viewport_title = "Open Cam - IQ lab"
    viewport_width = 1500
    viewport_height = 980
    window_label = "IQ lab"
    control_panel_width = 400
    banner_wrap = 1100
    presenter_mode = False
    recipe_custom_entry = False
    recipe_autoload = False
    scenarios = SCENARIOS
    default_banner_title = "Image-quality lab"
    default_banner_body = (
        "Pick a test scene and a camera recipe, then Run: pbrt renders the scene through the recipe's "
        "optics, the full sensor chain processes it, and the IQ-lab metrics are computed on the ROIs shown."
    )

    def init_state(self) -> None:
        self.scene = "diorama"
        self.camera_recipe_id = "default"
        self.xres = 360
        self.yres = 240
        self.pixelsamples = 16
        self.seed = 0
        self._queue: queue.Queue = queue.Queue()
        self._running = False
        self._workdir = Path(tempfile.mkdtemp(prefix="opencam_iqlab_"))
        self.last_row: dict | None = None

    def get_scenario(self, scenario_id: str):
        return get_scenario(scenario_id)

    def apply_scenario_state(self, sc) -> None:
        self.scene = sc.scene
        self.xres, self.yres, self.pixelsamples = sc.xres, sc.yres, sc.pixelsamples

    # --- controls ----------------------------------------------------
    def read_controls(self) -> None:
        self.scene = _SCENE_BY_LABEL[dpg.get_value("iq_scene_combo")]
        self.camera_recipe_id = dpg.get_value("recipe_combo")
        self.xres = int(dpg.get_value("iq_xres"))
        self.yres = int(dpg.get_value("iq_yres"))
        self.pixelsamples = int(dpg.get_value("iq_spp"))
        self.seed = int(dpg.get_value("iq_seed"))

    def push_controls(self) -> None:
        dpg.set_value("iq_scene_combo", SCENE_LABELS[self.scene])
        dpg.set_value("recipe_combo", self.camera_recipe_id)
        dpg.set_value("iq_xres", self.xres)
        dpg.set_value("iq_yres", self.yres)
        dpg.set_value("iq_spp", self.pixelsamples)
        dpg.set_value("iq_seed", self.seed)

    def refresh(self) -> None:
        note = "" if pbrt_available() else "PBRT binary not found -- build it (tools/build_pbrt.sh) to run scenes."
        dpg.set_value("status_text", note)
        dpg.configure_item("iq_run_button", enabled=pbrt_available() and not self._running)

    # --- run ---------------------------------------------------------
    def _run(self, _sender=None, _app_data=None, _user_data=None) -> None:
        if self._running:
            return
        self.read_controls()
        self._running = True
        dpg.configure_item("iq_run_button", enabled=False)
        dpg.set_value("iq_result", f"Running {self.scene} with {self.camera_recipe_id} ...")
        scene, recipe = self.scene, self.camera_recipe_id
        res, spp, seed = (self.xres, self.yres), self.pixelsamples, self.seed
        path = next(r.path for r in self._recipes if r.id == recipe)
        out = self._workdir / recipe / scene

        def worker():
            try:
                sc = import_tool("run_recipe_scorecard")
                optics = sc.optics_of(load_camera_model(path))
                render = sc.render_scene(scene, optics, out / "render", res[0], res[1], spp)
                images: dict = {}
                row = sc.score_recipe(recipe, {scene: render}, out / "sensor", seed, images=images)
                row["optics"] = sc.optics_key(optics)
                self._queue.put(("ok", scene, render, row, images.get(scene)))
            except Exception as exc:  # surfaced in the result text, not swallowed
                self._queue.put(("error", f"{type(exc).__name__}: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def tick(self) -> None:
        try:
            while True:
                self.show_result(*self._queue.get_nowait())
        except queue.Empty:
            pass

    def show_result(self, status: str, *payload) -> None:
        self._running = False
        dpg.configure_item("iq_run_button", enabled=pbrt_available())
        if status != "ok":
            dpg.set_value("iq_result", f"FAILED: {payload[0]}")
            return
        scene, render, row, image = payload
        self.last_row = row
        if image is not None:
            dpg.set_value("iq_texture", display_rgba(image))
        self._draw_rois(scene, render, image, row)
        self._fill_table(scene, row)
        dpg.set_value(
            "iq_result",
            f"{row['recipe']}  ({row.get('optics', '')}, CFA {row.get('cfa')}, {row.get('hdr')})  "
            f"render {render.get('render_s', 0):.1f} s",
        )

    def _draw_rois(self, scene: str, render: dict, image, row: dict) -> None:
        dpg.delete_item("iq_rois", children_only=True)
        h, w = render["shape"]
        sx, sy = PREVIEW_W / w, PREVIEW_H / h
        for kind, (x0, y0, x1, y1) in scene_rois(scene, render, row, image):
            dpg.draw_rectangle(
                (x0 * sx, y0 * sy), (x1 * sx, y1 * sy), color=ROI_COLOURS[kind], thickness=2, parent="iq_rois"
            )

    def _fill_table(self, scene: str, row: dict) -> None:
        sc = import_tool("run_recipe_scorecard")
        labels = {k: (h, f) for k, h, f in sc.METRICS}
        dpg.delete_item("iq_table", children_only=True, slot=1)
        for k in SCENE_METRICS[scene]:
            h, f = labels[k]
            with dpg.table_row(parent="iq_table"):
                dpg.add_text(h)
                dpg.add_text(sc._fmt(row.get(k), f))

    # --- layout ------------------------------------------------------
    def register_textures(self) -> None:
        dpg.add_dynamic_texture(PREVIEW_W, PREVIEW_H, _blank_rgba(PREVIEW_W, PREVIEW_H), tag="iq_texture")

    def build_pre_controls(self) -> None:
        dpg.add_text("Test scene")
        dpg.add_combo(tag="iq_scene_combo", items=list(SCENE_LABELS.values()), width=-1)
        dpg.add_separator()

    def build_controls(self) -> None:
        dpg.add_slider_int(tag="iq_xres", label="Width (px)", default_value=self.xres, min_value=120, max_value=960)
        dpg.add_slider_int(tag="iq_yres", label="Height (px)", default_value=self.yres, min_value=80, max_value=640)
        dpg.add_slider_int(
            tag="iq_spp", label="PBRT samples/px", default_value=self.pixelsamples, min_value=4, max_value=256
        )
        dpg.add_input_int(tag="iq_seed", label="Noise seed", default_value=self.seed)

    def build_footer(self) -> None:
        dpg.add_separator()
        dpg.add_button(tag="iq_run_button", label="Run", width=-1, callback=self._run)
        dpg.add_text("", tag="iq_result", wrap=370)

    def build_content(self) -> None:
        dpg.add_text("Processed image (demosaic, white balance, CCM) with measurement ROIs")
        with dpg.drawlist(width=PREVIEW_W, height=PREVIEW_H, tag="iq_drawlist"):
            dpg.draw_image("iq_texture", (0, 0), (PREVIEW_W, PREVIEW_H))
            dpg.add_draw_layer(tag="iq_rois")
        dpg.add_separator()
        dpg.add_text("Metrics")
        with dpg.table(tag="iq_table", header_row=True, borders_innerH=True, width=PREVIEW_W):
            dpg.add_table_column(label="Metric")
            dpg.add_table_column(label="Value")


def run_app(scenario_id: str | None = None) -> None:
    IqLabApp(scenario_id=scenario_id).run()
