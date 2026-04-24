from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from tools.camera_model import load_camera_model


class TestCameraModelLoader(unittest.TestCase):
    def test_recipe_composes_lens_and_sensor_models(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / "config" / "camera_recipes").mkdir(parents=True, exist_ok=True)
            (repo / "config" / "lens_models").mkdir(parents=True, exist_ok=True)
            (repo / "config" / "sensor_models").mkdir(parents=True, exist_ok=True)

            (repo / "config" / "lens_models" / "base.yaml").write_text(
                yaml.safe_dump({"schema_version": 1, "lens": {"camera": "thinlens"}}, sort_keys=False)
            )
            (repo / "config" / "sensor_models" / "base.yaml").write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "sensor": {"integration_time_s": 0.01},
                        "noise": {"emva": {}},
                        "cfa": {"enabled": True},
                        "sensor_forward": {"model": {}},
                    },
                    sort_keys=False,
                )
            )
            recipe_path = repo / "config" / "camera_recipes" / "combo.yaml"
            recipe_path.write_text(
                yaml.safe_dump(
                    {"schema_version": 1, "lens_model": "base", "sensor_model": "base"},
                    sort_keys=False,
                )
            )

            out = load_camera_model(recipe_path)
            self.assertEqual(out["lens"]["camera"], "thinlens")
            self.assertTrue(out["cfa"]["enabled"])

    def test_recipe_missing_ref_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / "config" / "camera_recipes").mkdir(parents=True, exist_ok=True)
            recipe_path = repo / "config" / "camera_recipes" / "broken.yaml"
            recipe_path.write_text(
                yaml.safe_dump(
                    {"schema_version": 1, "lens_model": "nope", "sensor_model": "missing"},
                    sort_keys=False,
                )
            )
            with self.assertRaises(FileNotFoundError):
                load_camera_model(recipe_path)


if __name__ == "__main__":
    unittest.main()
