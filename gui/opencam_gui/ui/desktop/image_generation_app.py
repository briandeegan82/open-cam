"""Dear PyGui desktop app: scene -> sensor -> image generation.

This is a thin front end over the real Open Cam pipeline: every "Generate"
click builds a command plan with ``opencam_gui.core.pipeline`` and runs the
actual ``tools/*.py`` scripts (or ``tools/run_pipeline.py``) as subprocesses,
in a background thread so the UI stays responsive. Nothing here renders an
image itself.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path

import numpy as np
import dearpygui.dearpygui as dpg
from PIL import Image

from opencam_gui.core import pipeline as pl
from opencam_gui.core.catalog import illuminant_label, list_camera_recipes, list_illuminants
from opencam_gui.core.repo import pbrt_available
from opencam_gui.topics.image_generation.scenarios import SCENARIOS, get_scenario

PREVIEW_W, PREVIEW_H = 420, 280

_LUX_PRESETS = [
    ("Moonlight (~0.3 lux)", 0.3),
    ("Twilight (~10 lux)", 10.0),
    ("Dim indoor (~50 lux)", 50.0),
    ("Office (~400 lux)", 400.0),
    ("Overcast day (~5000 lux)", 5000.0),
    ("Direct sun (~50000 lux)", 50000.0),
]


def _blank_rgba(w: int, h: int) -> list[float]:
    return [0.08, 0.08, 0.09, 1.0] * (w * h)


def _load_png_rgba_flat(path: Path, w: int, h: int) -> list[float] | None:
    if not path.is_file():
        return None
    img = Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float64) / 255.0
    rgba = np.ones((h, w, 4), dtype=np.float64)
    rgba[..., :3] = arr
    return rgba.ravel().tolist()


class ImageGenerationApp:
    def __init__(self, scenario_id: str | None = None) -> None:
        self._recipes = list_camera_recipes()
        self._illuminants = list_illuminants()
        self._illum_by_id = {i.id: i for i in self._illuminants}
        self._scenes = pl.list_scenes()

        self.scene_id = "colorchecker"
        self.camera_recipe_id = "nikon_z6"
        self.mode = "fast_analytic"
        self.illuminant_id = "D65"
        self.target_lux = 1000.0
        self.exposure_time_s = 0.01
        self.seed = 0
        self.xres = 320
        self.yres = 240
        self.pixelsamples = 32
        self.dry_run = False

        self._log_queue: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._running = False
        self._log_lines: list[str] = []

        self._scenario_title = "Scene -> sensor -> image"
        self._scenario_note = (
            "Pick a scene, camera and illumination, then Generate. Every step below is a "
            "real tools/*.py subprocess call, streamed into the log panel as it runs."
        )
        if scenario_id:
            self._apply_scenario_fields(get_scenario(scenario_id))

    def _apply_scenario_fields(self, sc) -> None:
        self.scene_id = sc.scene_id
        self.camera_recipe_id = sc.camera_recipe_id
        self.illuminant_id = sc.illuminant_id or self.illuminant_id
        self.target_lux = sc.target_illuminance_lux
        self.exposure_time_s = sc.exposure_time_s
        self.mode = sc.mode
        self._scenario_title = sc.title
        self._scenario_note = f"{sc.teaching_point}\n{sc.notes}"

    # --- controls --------------------------------------------------
    def _current_scene(self) -> pl.Scene:
        return pl.SCENES[self.scene_id]

    def _read_controls(self) -> None:
        scene_label = dpg.get_value("scene_combo")
        self.scene_id = self._scene_id_by_label[scene_label]
        self.camera_recipe_id = dpg.get_value("recipe_combo")
        self.mode = "fast_analytic" if dpg.get_value("mode_radio") == "Fast preview (analytic, no PBRT)" else "pbrt_accurate"
        illum_label = dpg.get_value("illuminant_combo")
        self.illuminant_id = self._illum_id_by_label.get(illum_label)
        self.target_lux = float(dpg.get_value("lux_slider"))
        self.exposure_time_s = float(dpg.get_value("exposure_slider"))
        self.seed = int(dpg.get_value("seed_input"))
        self.xres = int(dpg.get_value("xres_slider"))
        self.yres = int(dpg.get_value("yres_slider"))
        self.pixelsamples = int(dpg.get_value("pixelsamples_slider"))
        self.dry_run = bool(dpg.get_value("dry_run_checkbox"))

    def _push_controls(self) -> None:
        scene = self._current_scene()
        dpg.set_value("scene_combo", scene.label)
        dpg.set_value("recipe_combo", self.camera_recipe_id)
        dpg.set_value(
            "mode_radio",
            "Fast preview (analytic, no PBRT)" if self.mode == "fast_analytic" else "Physically accurate (PBRT render)",
        )
        if self.illuminant_id and self.illuminant_id in self._illum_by_id:
            dpg.set_value("illuminant_combo", illuminant_label(self._illum_by_id[self.illuminant_id]))
        dpg.set_value("lux_slider", self.target_lux)
        dpg.set_value("exposure_slider", self.exposure_time_s)
        dpg.set_value("seed_input", self.seed)
        dpg.set_value("xres_slider", self.xres)
        dpg.set_value("yres_slider", self.yres)
        dpg.set_value("pixelsamples_slider", self.pixelsamples)
        self._update_mode_availability()

    def _update_mode_availability(self) -> None:
        scene = self._current_scene()
        pbrt_ok = pbrt_available()
        note_lines = [scene.description]
        if not scene.supports_fast_analytic:
            note_lines.append("This scene requires the physically-accurate PBRT path (no fast preview available).")
            if self.mode == "fast_analytic":
                self.mode = "pbrt_accurate"
                dpg.set_value("mode_radio", "Physically accurate (PBRT render)")
        if not scene.supports_illuminant_spectrum:
            note_lines.append("Illuminant spectrum is fixed by the scene builder for this target.")
        if not pbrt_ok:
            note_lines.append("PBRT binary not found -- 'Physically accurate' will only work as a Dry run.")
        dpg.configure_item("illuminant_combo", enabled=scene.supports_illuminant_spectrum)
        dpg.set_value("scene_note", "\n".join(note_lines))

    def _apply_scenario(self, _sender=None, _app_data=None, user_data=None) -> None:
        sc = get_scenario(user_data)
        self._apply_scenario_fields(sc)
        self._push_controls()
        dpg.set_value("banner_title", self._scenario_title)
        dpg.set_value("banner_body", self._scenario_note)

    def _on_scene_change(self, *_):
        self._read_controls()
        self._update_mode_availability()

    def _set_lux_preset(self, _sender=None, _app_data=None, user_data=None) -> None:
        dpg.set_value("lux_slider", user_data)

    # --- run ---------------------------------------------------------
    def _append_log(self, line: str) -> None:
        self._log_lines.append(line)
        if len(self._log_lines) > 400:
            self._log_lines = self._log_lines[-400:]
        dpg.set_value("log_text", "\n".join(self._log_lines))
        dpg.set_y_scroll("log_window", dpg.get_y_scroll_max("log_window"))

    def _generate(self, _sender=None, _app_data=None, _user_data=None) -> None:
        if self._running:
            return
        self._read_controls()
        self._log_lines = []
        dpg.set_value("log_text", "")
        dpg.configure_item("generate_button", enabled=False)
        dpg.set_value("result_text", "")
        self._running = True

        req = pl.GenerationRequest(
            scene_id=self.scene_id,
            camera_model_config=next(r.path for r in self._recipes if r.id == self.camera_recipe_id),
            mode=self.mode,
            illuminant_csv=self._illum_by_id[self.illuminant_id].repo_relative if self.illuminant_id in self._illum_by_id else None,
            target_illuminance_lux=self.target_lux,
            exposure_time_s=self.exposure_time_s,
            seed=self.seed,
            xres=self.xres,
            yres=self.yres,
            pixelsamples=self.pixelsamples,
            run_label="gui_session",
        )
        dry_run = self.dry_run

        def worker():
            result = pl.run_plan(req, dry_run=dry_run, on_output=self._log_queue.put)
            self._log_queue.put(("__done__", result))

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    def _drain_queue(self) -> None:
        try:
            while True:
                item = self._log_queue.get_nowait()
                if isinstance(item, tuple) and item and item[0] == "__done__":
                    self._on_finished(item[1])
                else:
                    self._append_log(str(item))
        except queue.Empty:
            pass

    def _on_finished(self, result: pl.RunResult) -> None:
        self._running = False
        dpg.configure_item("generate_button", enabled=True)
        if not result.ok:
            dpg.set_value("result_text", f"FAILED: {result.error}")
            return
        if self.dry_run:
            dpg.set_value("result_text", "Dry run complete -- no files were written.")
            return

        out_dir = pl.preview_output_dir()
        clean = _load_png_rgba_flat(out_dir / "clean_demosaic_rgb8.png", PREVIEW_W, PREVIEW_H)
        noisy = _load_png_rgba_flat(out_dir / "noisy_demosaic_rgb8.png", PREVIEW_W, PREVIEW_H)
        if clean:
            dpg.set_value("clean_texture", clean)
        if noisy:
            dpg.set_value("noisy_texture", noisy)

        stats = pl.load_run_stats()
        lines = [f"Done. Preview PNGs written to {out_dir}"]
        if stats:
            lines.append(
                f"bit depth {stats.get('bit_depth')}   K_eff {stats.get('K_effective_e_per_DN'):.3f} e-/DN   "
                f"full well {stats.get('full_well_effective_e'):.0f} e-   black {stats.get('black_level_DN'):.1f} DN"
            )
            if "signal_e_mean_mono" in stats:
                lines.append(
                    f"mean signal (mono) {stats.get('signal_e_mean_mono'):.1f} e-   "
                    f"DN range [{stats.get('dn_noisy_min_mono')}, {stats.get('dn_noisy_max_mono')}]"
                )
        dpg.set_value("result_text", "\n".join(lines))

    def _tick(self) -> None:
        self._drain_queue()

    def run(self) -> None:
        dpg.create_context()
        dpg.create_viewport(title="Open Cam - Image Generation", width=1500, height=980)

        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6)
        dpg.bind_theme(global_theme)

        with dpg.texture_registry():
            dpg.add_dynamic_texture(PREVIEW_W, PREVIEW_H, _blank_rgba(PREVIEW_W, PREVIEW_H), tag="clean_texture")
            dpg.add_dynamic_texture(PREVIEW_W, PREVIEW_H, _blank_rgba(PREVIEW_W, PREVIEW_H), tag="noisy_texture")

        scene_labels = [s.label for s in self._scenes]
        self._scene_id_by_label = {s.label: s.id for s in self._scenes}
        illum_labels = [illuminant_label(i) for i in self._illuminants]
        self._illum_id_by_label = {illuminant_label(i): i.id for i in self._illuminants}

        with dpg.window(tag="primary", label="Image Generation"):
            with dpg.child_window(tag="banner_panel", height=72, border=True):
                dpg.add_text(self._scenario_title, tag="banner_title")
                dpg.add_text(self._scenario_note, tag="banner_body", wrap=1100)

            with dpg.group(horizontal=True):
                with dpg.child_window(width=400, border=True):
                    dpg.add_text("Scene")
                    dpg.add_combo(tag="scene_combo", items=scene_labels, callback=self._on_scene_change)
                    dpg.add_text("", tag="scene_note", wrap=370)
                    dpg.add_separator()

                    dpg.add_text("Camera recipe")
                    dpg.add_combo(
                        tag="recipe_combo", items=[r.id for r in self._recipes],
                        callback=lambda *_: None,
                    )
                    dpg.add_separator()

                    dpg.add_text("Generation mode")
                    dpg.add_radio_button(
                        tag="mode_radio",
                        items=["Fast preview (analytic, no PBRT)", "Physically accurate (PBRT render)"],
                        callback=lambda *_: None,
                    )
                    dpg.add_separator()

                    dpg.add_text("Illumination spectrum")
                    dpg.add_combo(tag="illuminant_combo", items=illum_labels, callback=lambda *_: None)
                    dpg.add_text("Illumination intensity (lux)")
                    dpg.add_slider_float(
                        tag="lux_slider", default_value=self.target_lux, min_value=0.1, max_value=50000.0,
                        format="%.1f lux",
                    )
                    with dpg.group(horizontal=True):
                        for label, val in _LUX_PRESETS:
                            dpg.add_button(label=label.split(" (")[0], user_data=val, callback=self._set_lux_preset)
                    dpg.add_slider_float(
                        tag="exposure_slider", label="Exposure time (s)", default_value=self.exposure_time_s,
                        min_value=0.0005, max_value=1.0,
                    )
                    dpg.add_separator()

                    dpg.add_text("Lecture scenarios")
                    for sid, sc in SCENARIOS.items():
                        dpg.add_button(label=sc.title, width=-1, user_data=sid, callback=self._apply_scenario)
                    dpg.add_separator()

                    dpg.add_text("Advanced")
                    dpg.add_input_int(tag="seed_input", label="Seed", default_value=self.seed)
                    dpg.add_slider_int(tag="xres_slider", label="Width (px)", default_value=self.xres, min_value=80, max_value=960)
                    dpg.add_slider_int(tag="yres_slider", label="Height (px)", default_value=self.yres, min_value=60, max_value=640)
                    dpg.add_slider_int(
                        tag="pixelsamples_slider", label="PBRT samples/px", default_value=self.pixelsamples,
                        min_value=4, max_value=1024,
                    )
                    dpg.add_checkbox(tag="dry_run_checkbox", label="Dry run (print commands only)", default_value=False)
                    dpg.add_button(tag="generate_button", label="Generate", width=-1, callback=self._generate)
                    dpg.add_text("", tag="result_text", wrap=370)

                with dpg.child_window(border=False):
                    with dpg.group(horizontal=True):
                        with dpg.group():
                            dpg.add_text("Clean (no sensor noise)")
                            dpg.add_image("clean_texture", width=PREVIEW_W, height=PREVIEW_H)
                        with dpg.group():
                            dpg.add_text("Noisy (full EMVA + Bayer + demosaic)")
                            dpg.add_image("noisy_texture", width=PREVIEW_W, height=PREVIEW_H)
                    dpg.add_separator()
                    dpg.add_text("Command log")
                    with dpg.child_window(tag="log_window", height=-1, border=True):
                        dpg.add_text("", tag="log_text", wrap=1000)

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("primary", True)
        self._push_controls()
        dpg.set_value("banner_title", self._scenario_title)
        dpg.set_value("banner_body", self._scenario_note)

        while dpg.is_dearpygui_running():
            self._tick()
            dpg.render_dearpygui_frame()

        dpg.destroy_context()


def run_app(scenario_id: str | None = None) -> None:
    ImageGenerationApp(scenario_id=scenario_id).run()
