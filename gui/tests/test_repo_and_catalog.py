
from opencam_gui.core import catalog
from opencam_gui.core.repo import config_dir, repo_root, tools_dir


def test_repo_root_points_at_open_cam():
    root = repo_root()
    assert (root / "tools" / "camera_model.py").is_file()
    assert (root / "config" / "pipeline.yaml").is_file()


def test_tools_dir_and_config_dir():
    assert tools_dir().name == "tools"
    assert config_dir().name == "config"


def test_list_camera_recipes_nonempty_and_has_nikon_z6():
    recipes = catalog.list_camera_recipes()
    assert len(recipes) > 10
    ids = {r.id for r in recipes}
    assert "nikon_z6" in ids
    assert "iphone_8" in ids


def test_find_recipe_display_name():
    recipe = catalog.find_recipe("nikon_z6")
    assert recipe.display_name == "Nikon Z6"
    assert recipe.path.is_file()


def test_list_illuminants_has_d65():
    illuminants = catalog.list_illuminants()
    ids = {i.id for i in illuminants}
    assert "D65" in ids
    d65 = next(i for i in illuminants if i.id == "D65")
    assert d65.repo_relative.endswith("D65.csv")


def test_load_spectrum_csv_matches_apply_emva_noise():
    illuminants = catalog.list_illuminants()
    d65 = next(i for i in illuminants if i.id == "D65")
    wl, val = catalog.load_spectrum_csv(d65.path)
    assert len(wl) == len(val)
    assert len(wl) > 100
    assert 380.0 <= wl[0] <= 400.0
