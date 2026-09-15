"""Shared Dear PyGui scaffolding for the Open Cam teaching demos.

Every demo is the same shape: a banner carrying the current lecture
scenario's teaching point, a left control panel (camera recipe dropdown,
demo-specific sliders, scenario buttons, status readout) and a right content
panel of plots and previews. :class:`DemoApp` owns that shape plus the
window/viewport lifecycle; subclasses supply only the physics wiring and the
widgets that are actually specific to their topic.

Subclasses must implement :meth:`init_state`, :meth:`apply_scenario_state`,
:meth:`build_controls` and :meth:`build_content`, and will usually implement
:meth:`read_controls`, :meth:`push_controls` and :meth:`refresh`.
"""

from __future__ import annotations

from typing import Any, Mapping

import dearpygui.dearpygui as dpg

from opencam_gui.core.camera import load_camera_model
from opencam_gui.core.catalog import list_camera_recipes

CUSTOM_RECIPE = "Custom"


class DemoApp:
    """Base class for the desktop teaching demos."""

    # --- viewport / layout -------------------------------------------
    viewport_title = "Open Cam"
    viewport_width = 1400
    viewport_height = 900
    window_label = "Demo"
    control_panel_width = 380
    banner_height = 72
    banner_wrap = 1000

    # --- behaviour toggles -------------------------------------------
    #: Show the "Presenter mode" checkbox that hides the ``advanced_controls`` group.
    presenter_mode = True
    #: Show the camera recipe dropdown at the top of the control panel.
    show_recipe_combo = True
    #: Prepend a "Custom" entry meaning "no recipe loaded, sliders are free".
    recipe_custom_entry = True
    #: Load the recipe immediately when the dropdown changes.
    recipe_autoload = True
    #: Render a ``status_text`` widget under the scenario buttons.
    show_status_text = True

    # --- scenario wiring (override in subclass) ----------------------
    scenarios: Mapping[str, Any] = {}
    default_banner_title = "Open Cam demo"
    default_banner_body = ""

    def __init__(self, scenario_id: str | None = None) -> None:
        self._recipes = list_camera_recipes()
        self._presenter = False
        self._scenario_title = self.default_banner_title
        self._scenario_note = self.default_banner_body
        self.camera_recipe_id: str | None = None
        self.init_state()
        if scenario_id:
            self.apply_scenario_fields(self.get_scenario(scenario_id))

    # =================================================================
    # Subclass hooks
    # =================================================================
    def init_state(self) -> None:
        """Set the demo's default parameter values as instance attributes."""
        raise NotImplementedError

    def get_scenario(self, scenario_id: str) -> Any:
        """Return the scenario object for *scenario_id*."""
        raise NotImplementedError

    def apply_scenario_state(self, sc: Any) -> None:
        """Copy the demo-specific fields of *sc* onto ``self``."""
        raise NotImplementedError

    def apply_recipe_state(self, model: dict) -> None:
        """Copy the relevant fields of a loaded camera model onto ``self``."""

    def build_pre_controls(self) -> None:
        """Widgets above the camera recipe dropdown (e.g. a scene selector)."""

    def build_controls(self) -> None:
        """The demo's own sliders, between the recipe combo and the scenarios."""
        raise NotImplementedError

    def build_footer(self) -> None:
        """Widgets below the status readout (e.g. a Generate button)."""

    def build_content(self) -> None:
        """The right-hand panel: plots, tables, image previews."""
        raise NotImplementedError

    def register_textures(self) -> None:
        """Create dynamic textures inside the texture registry."""

    def register_themes(self) -> None:
        """Create any item themes the content panel binds to."""

    def read_controls(self) -> None:
        """Pull widget values into instance attributes."""

    def push_controls(self) -> None:
        """Push instance attributes back into the widgets."""

    def refresh(self) -> None:
        """Recompute and redraw everything from the current control values."""

    def tick(self) -> None:
        """Called once per frame before rendering (e.g. to drain a log queue)."""

    # =================================================================
    # Shared behaviour
    # =================================================================
    def apply_scenario_fields(self, sc: Any) -> None:
        self._scenario_title = sc.title
        self._scenario_note = f"{sc.teaching_point}\n{sc.notes}"
        self.camera_recipe_id = getattr(sc, "camera_recipe_id", None)
        self.apply_scenario_state(sc)

    def load_recipe(self, recipe_id: str) -> None:
        recipe = next((r for r in self._recipes if r.id == recipe_id), None)
        if recipe is None:
            return
        self.apply_recipe_state(load_camera_model(recipe.path))
        self.camera_recipe_id = recipe_id
        self.push_controls()
        self._sync_recipe_combo()
        self.refresh()

    def set_banner(self) -> None:
        dpg.set_value("banner_title", self._scenario_title)
        dpg.set_value("banner_body", self._scenario_note)

    def status_wrap(self) -> int:
        return max(200, self.control_panel_width - 30)

    # --- callbacks ---------------------------------------------------
    def _on_recipe_change(self, _sender=None, app_data=None, _user_data=None) -> None:
        if not app_data or not self.recipe_autoload:
            return
        if self.recipe_custom_entry and app_data == CUSTOM_RECIPE:
            return
        self.load_recipe(app_data)

    def _on_scenario_click(self, _sender=None, _app_data=None, user_data=None) -> None:
        self.apply_scenario_fields(self.get_scenario(user_data))
        self.push_controls()
        self._sync_recipe_combo()
        self.set_banner()
        self.refresh()

    def _on_presenter_toggle(self, _sender=None, app_data=None, _user_data=None) -> None:
        self._presenter = bool(app_data)
        if dpg.does_item_exist("advanced_controls"):
            dpg.configure_item("advanced_controls", show=not self._presenter)

    def on_control_change(self, *_args) -> None:
        """Generic slider callback: recompute everything."""
        self.refresh()

    # --- layout helpers ----------------------------------------------
    def _sync_recipe_combo(self) -> None:
        if not (self.show_recipe_combo and dpg.does_item_exist("recipe_combo")):
            return
        if self.camera_recipe_id:
            dpg.set_value("recipe_combo", self.camera_recipe_id)
        elif self.recipe_custom_entry:
            dpg.set_value("recipe_combo", CUSTOM_RECIPE)

    def _build_recipe_combo(self) -> None:
        ids = [r.id for r in self._recipes]
        items = [CUSTOM_RECIPE, *ids] if self.recipe_custom_entry else ids
        default = self.camera_recipe_id or (CUSTOM_RECIPE if self.recipe_custom_entry else (ids[0] if ids else ""))
        dpg.add_text("Camera recipe")
        dpg.add_combo(
            tag="recipe_combo",
            items=items,
            default_value=default,
            callback=self._on_recipe_change,
        )

    def _build_scenario_buttons(self) -> None:
        if not self.scenarios:
            return
        dpg.add_text("Lecture scenarios")
        for sid, sc in self.scenarios.items():
            dpg.add_button(label=sc.title, width=-1, user_data=sid, callback=self._on_scenario_click)

    def _build_control_panel(self) -> None:
        if self.presenter_mode:
            dpg.add_checkbox(
                tag="presenter_mode",
                label="Presenter mode (hide advanced)",
                default_value=False,
                callback=self._on_presenter_toggle,
            )
            dpg.add_separator()
        self.build_pre_controls()
        if self.show_recipe_combo:
            self._build_recipe_combo()
            dpg.add_separator()
        self.build_controls()
        dpg.add_separator()
        self._build_scenario_buttons()
        if self.show_status_text:
            dpg.add_separator()
            dpg.add_text("", tag="status_text", wrap=self.status_wrap())
        self.build_footer()

    def _bind_global_theme(self) -> None:
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6)
        dpg.bind_theme(global_theme)

    # --- lifecycle ---------------------------------------------------
    def build_ui(self) -> None:
        """Create every widget. Requires a Dear PyGui context but no viewport,
        so tests can build the whole interface headlessly."""
        self._bind_global_theme()
        self.register_themes()
        with dpg.texture_registry():
            self.register_textures()

        with dpg.window(tag="primary", label=self.window_label):
            with dpg.child_window(tag="banner_panel", height=self.banner_height, border=True):
                dpg.add_text(self._scenario_title, tag="banner_title")
                dpg.add_text(self._scenario_note, tag="banner_body", wrap=self.banner_wrap)

            with dpg.group(horizontal=True):
                with dpg.child_window(width=self.control_panel_width, border=True):
                    self._build_control_panel()
                with dpg.child_window(border=False):
                    self.build_content()

    def start(self) -> None:
        """Seed every widget from the current state and draw the first frame's data."""
        self.push_controls()
        self._sync_recipe_combo()
        self.set_banner()
        self.refresh()

    def run(self) -> None:
        dpg.create_context()
        dpg.create_viewport(
            title=self.viewport_title, width=self.viewport_width, height=self.viewport_height
        )
        self.build_ui()
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("primary", True)
        self.start()

        while dpg.is_dearpygui_running():
            self.tick()
            dpg.render_dearpygui_frame()

        dpg.destroy_context()
