from opencam_gui.core.camera import emva_summary, load_camera_model, optics_summary
from opencam_gui.core.catalog import find_recipe


def test_optics_summary_nikon_z6():
    recipe = find_recipe("nikon_z6")
    model = load_camera_model(recipe.path)
    s = optics_summary(model)
    assert s.f_number > 0
    assert s.pixel_pitch_um > 0
    assert s.camera_type in ("pinhole", "thinlens", "realistic")


def test_emva_summary_nikon_z6_matches_yaml_ballpark():
    recipe = find_recipe("nikon_z6")
    model = load_camera_model(recipe.path)
    s = emva_summary(model)
    # Full-frame ILC: large full well relative to a phone.
    assert s.full_well_e > 20000
    assert s.K_e_per_DN > 0
    assert s.sigma_d_e > 0


def test_emva_summary_iphone_8_small_full_well():
    recipe = find_recipe("iphone_8")
    model = load_camera_model(recipe.path)
    nikon = emva_summary(load_camera_model(find_recipe("nikon_z6").path))
    iphone = emva_summary(model)
    assert iphone.full_well_e < nikon.full_well_e
