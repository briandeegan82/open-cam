"""Headless smoke tests for the Dear PyGui demo apps.

Dear PyGui builds its whole widget tree inside a context, and only needs a
viewport to actually *display* it. So we can create a context, call
``build_ui()`` + ``start()`` and drive callbacks exactly as a user would --
catching missing tags, mismatched scenario fields and broken refresh paths
without ever opening a window.
"""

from __future__ import annotations

import dearpygui.dearpygui as dpg
import pytest

from opencam_gui.ui.desktop.base import CUSTOM_RECIPE
from opencam_gui.ui.desktop.exposure_app import ExposureApp
from opencam_gui.ui.desktop.geometry_app import GeometryApp
from opencam_gui.ui.desktop.image_generation_app import ImageGenerationApp
from opencam_gui.ui.desktop.isp_app import IspApp
from opencam_gui.ui.desktop.mtf_app import MtfApp
from opencam_gui.ui.desktop.optics_app import OpticsApp
from opencam_gui.ui.desktop.sensor_app import SensorApp

ALL_APPS = [GeometryApp, OpticsApp, MtfApp, SensorApp, ExposureApp, IspApp, ImageGenerationApp]

#: Apps with a "Custom" recipe entry that auto-loads a camera config on change.
RECIPE_APPS = [GeometryApp, OpticsApp, MtfApp, SensorApp, ExposureApp, IspApp]


@pytest.fixture
def dpg_context():
    dpg.create_context()
    try:
        yield
    finally:
        dpg.destroy_context()


def _built(app_cls, scenario_id=None):
    app = app_cls(scenario_id=scenario_id)
    app.build_ui()
    app.start()
    return app


@pytest.mark.parametrize("app_cls", ALL_APPS, ids=lambda c: c.__name__)
def test_app_builds_and_starts(dpg_context, app_cls):
    app = _built(app_cls)
    assert dpg.does_item_exist("primary")
    assert dpg.does_item_exist("banner_title")
    assert dpg.get_value("banner_title") == app.default_banner_title


@pytest.mark.parametrize("app_cls", ALL_APPS, ids=lambda c: c.__name__)
def test_every_scenario_applies_cleanly(dpg_context, app_cls):
    app = _built(app_cls)
    assert app_cls.scenarios, f"{app_cls.__name__} has no lecture scenarios"
    for sid, sc in app_cls.scenarios.items():
        app._on_scenario_click(user_data=sid)
        assert dpg.get_value("banner_title") == sc.title
        assert sc.teaching_point in dpg.get_value("banner_body")


@pytest.mark.parametrize("app_cls", ALL_APPS, ids=lambda c: c.__name__)
def test_controls_round_trip(dpg_context, app_cls):
    """push_controls() then read_controls() must preserve state, otherwise a
    scenario or recipe load would be silently reverted on the next refresh."""
    app = _built(app_cls)
    before = {k: v for k, v in vars(app).items() if isinstance(v, (int, float, str, bool))}
    app.push_controls()
    app.read_controls()
    after = {k: v for k, v in vars(app).items() if isinstance(v, (int, float, str, bool))}
    for key, old in before.items():
        if isinstance(old, float):
            assert after[key] == pytest.approx(old, rel=1e-5), key
        else:
            assert after[key] == old, key


@pytest.mark.parametrize("app_cls", RECIPE_APPS, ids=lambda c: c.__name__)
def test_recipe_dropdown_loads_real_configs(dpg_context, app_cls):
    app = _built(app_cls)
    for recipe_id in ("nikon_z6", "iphone_8"):
        app._on_recipe_change(app_data=recipe_id)
        assert app.camera_recipe_id == recipe_id
        assert dpg.get_value("recipe_combo") == recipe_id


@pytest.mark.parametrize("app_cls", RECIPE_APPS, ids=lambda c: c.__name__)
def test_custom_recipe_entry_does_not_load(dpg_context, app_cls):
    app = _built(app_cls)
    app._on_recipe_change(app_data="nikon_z6")
    app._on_recipe_change(app_data=CUSTOM_RECIPE)
    assert app.camera_recipe_id == "nikon_z6"


@pytest.mark.parametrize("app_cls", RECIPE_APPS, ids=lambda c: c.__name__)
def test_presenter_mode_hides_advanced_controls(dpg_context, app_cls):
    app = _built(app_cls)
    assert dpg.does_item_exist("advanced_controls")
    app._on_presenter_toggle(app_data=True)
    assert dpg.get_item_configuration("advanced_controls")["show"] is False
    app._on_presenter_toggle(app_data=False)
    assert dpg.get_item_configuration("advanced_controls")["show"] is True


@pytest.mark.parametrize("app_cls", ALL_APPS, ids=lambda c: c.__name__)
def test_constructing_with_scenario_id_matches_clicking_it(dpg_context, app_cls):
    """``--scenario`` on the CLI must land in the same state as the button."""
    sid = next(iter(app_cls.scenarios))
    from_arg = _built(app_cls, scenario_id=sid)
    assert dpg.get_value("banner_title") == app_cls.scenarios[sid].title
    scalars = {k: v for k, v in vars(from_arg).items() if isinstance(v, (int, float, str, bool))}

    dpg.destroy_context()
    dpg.create_context()
    from_click = _built(app_cls)
    from_click._on_scenario_click(user_data=sid)
    assert {k: v for k, v in vars(from_click).items() if isinstance(v, (int, float, str, bool))} == scalars


def test_isp_cfa_scenarios_load_the_named_qe_curves(dpg_context):
    """RCCB/CMY scenarios have to change the spectral basis, not just the combo."""
    app = _built(IspApp)
    app._on_scenario_click(user_data="rccb")
    assert app.camera_recipe_id == "default_rccb"
    assert app.qe_paths["green"].endswith("QE_cyan.csv")
    app._on_scenario_click(user_data="cmy")
    assert app.camera_recipe_id == "default_cmy"
    assert app.qe_paths["green"].endswith("QE_magenta.csv")
    assert app.qe_paths["red"].endswith("QE_cyan.csv")
    assert app.qe_paths["blue"].endswith("QE_yellow.csv")
