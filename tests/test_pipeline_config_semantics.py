from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from tools.pipeline_shell_env import resolve_camera_model_path as resolve_shell_camera_model_path
from tools.run_pipeline import resolve_camera_model_path as resolve_run_pipeline_camera_model_path


def _minimal_camera_model_yaml() -> dict:
    return {
        "lens": {},
        "sensor": {},
        "noise": {},
        "cfa": {},
        "sensor_forward": {},
    }


def _minimal_recipe_yaml() -> dict:
    return {"schema_version": 1, "lens_model": "default", "sensor_model": "default"}


class TestPipelineConfigSemantics(unittest.TestCase):
    def test_camera_model_fields_conflict_rejected_in_python_runner(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            paths = {"camera_model_name": "iphone_8", "camera_model_config": "config/camera_models/iphone_8.yaml"}
            with self.assertRaises(ValueError):
                resolve_run_pipeline_camera_model_path(repo, paths, cli_path=None)

    def test_camera_model_fields_conflict_rejected_in_shell_exporter(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            paths = {"camera_model_name": "iphone_8", "camera_model_config": "config/camera_models/iphone_8.yaml"}
            with self.assertRaises(ValueError):
                resolve_shell_camera_model_path(repo, paths)

    def test_shell_export_defaults_match_python_runner(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / "config" / "camera_recipes").mkdir(parents=True, exist_ok=True)
            (repo / "config" / "lens_models").mkdir(parents=True, exist_ok=True)
            (repo / "config" / "sensor_models").mkdir(parents=True, exist_ok=True)
            (repo / "config" / "camera_recipes" / "default.yaml").write_text(
                yaml.safe_dump(_minimal_recipe_yaml(), sort_keys=False)
            )
            minimal = _minimal_camera_model_yaml()
            (repo / "config" / "lens_models" / "default.yaml").write_text(
                yaml.safe_dump({"schema_version": 1, "lens": minimal["lens"]}, sort_keys=False)
            )
            (repo / "config" / "sensor_models" / "default.yaml").write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "sensor": minimal["sensor"],
                        "noise": minimal["noise"],
                        "cfa": minimal["cfa"],
                        "sensor_forward": minimal["sensor_forward"],
                    },
                    sort_keys=False,
                )
            )
            cfg_path = repo / "config" / "pipeline.yaml"
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text(yaml.safe_dump({"paths": {}}, sort_keys=False))

            script = Path(__file__).resolve().parent.parent / "tools" / "pipeline_shell_env.py"
            out = subprocess.run(
                [sys.executable, str(script), str(repo), str(cfg_path)],
                capture_output=True,
                text=True,
                check=True,
            ).stdout

            self.assertIn("export SENSOR_FORWARD_ENABLED=0", out)
            self.assertIn("export VALIDATE_DEMOSAIC=0", out)
            self.assertIn("export VALIDATE_EMVA=0", out)


if __name__ == "__main__":
    unittest.main()
