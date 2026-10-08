"""Module tests for batch_validate_emva, validate_colorchecker and the Munsell / IQ-target scene builders."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import batch_validate_emva as bve
import build_image_quality_targets as iq
import build_munsell_scenes as munsell
import numpy as np
import validate_colorchecker as vcc
import yaml
from synthetic_data import REPO, run_tool_main, write_rgb_exr


def _checks(issues: list) -> dict[str, str]:
    return {i.check: i.severity for i in issues}


class TestBatchValidateEmvaChecks(unittest.TestCase):
    def test_adc_vs_full_well_bands(self) -> None:
        # 10-bit, black 64 → 959 usable DN.
        self.assertEqual(_checks(bve.check_adc_vs_full_well(0.22, 9500, 10, 64)), {"adc_vs_full_well": bve.FAIL})
        self.assertEqual(_checks(bve.check_adc_vs_full_well(7.0, 9500, 10, 64)), {"adc_vs_full_well": bve.WARN})
        self.assertEqual(bve.check_adc_vs_full_well(9.9062, 9500, 10, 64), [])
        self.assertEqual(_checks(bve.check_adc_vs_full_well(30.0, 9500, 10, 64)), {"adc_vs_full_well": bve.WARN})
        self.assertEqual(_checks(bve.check_adc_vs_full_well(1.0, 0, 10, 64)), {"adc_vs_full_well": bve.FAIL})

    def test_full_well_fits_adc(self) -> None:
        self.assertEqual(bve.check_full_well_fits_adc(9.9062, 9500, 10, 64), [])
        self.assertEqual(_checks(bve.check_full_well_fits_adc(0.22, 9500, 10, 64)), {"full_well_fits_adc": bve.FAIL})

    def test_agg_status_takes_worst(self) -> None:
        self.assertEqual(bve._agg_status([]), "PASS")
        issues = [bve.Issue(bve.WARN, "a", ""), bve.Issue(bve.FAIL, "b", ""), bve.Issue(bve.WARN, "c", "")]
        self.assertEqual(bve._agg_status(issues), bve.FAIL)
        self.assertEqual(bve._agg_status(issues[:1]), bve.WARN)

    def test_individual_checks_flag_bad_parameters(self) -> None:
        flagged = [
            bve.check_black_level_headroom(10.0, 50.0, 1.0),
            bve.check_read_noise_visibility(10.0, 1.0),
            bve.check_dynamic_range(1000.0, 100.0),
            bve.check_prnu_vs_shot_noise(0.2, 50000.0),
            bve.check_dark_current(1e4, 1.0, 1000.0),
            bve.check_dsnu_vs_read_noise(10.0, 1.0),
        ]
        for issues in flagged:
            self.assertTrue(issues, "expected at least one issue")
        healthy = [
            bve.check_dynamic_range(30000.0, 2.0),
            bve.check_dark_current(0.1, 0.01, 30000.0),
            bve.check_dsnu_vs_read_noise(0.1, 2.0),
        ]
        for issues in healthy:
            self.assertEqual(issues, [])


class TestBatchValidateEmvaCamera(unittest.TestCase):
    def test_shipped_recipe_passes_adc_checks(self) -> None:
        r = bve.validate_camera(REPO / "config" / "camera_recipes" / "iphone_8.yaml")
        self.assertIn(r["status"], ("PASS", bve.WARN))
        checks = {i["check"] for i in r["issues"]}
        self.assertNotIn("adc_vs_full_well", checks)
        self.assertNotIn("full_well_fits_adc", checks)
        self.assertAlmostEqual(r["params"]["K_effective_e_per_DN"], 9.9062)

    def test_unloadable_recipe_is_fail(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "broken.yaml"
            bad.write_text(yaml.safe_dump({"schema_version": 1, "model": {}}))
            r = bve.validate_camera(bad)
        self.assertEqual(r["status"], bve.FAIL)
        self.assertEqual(r["issues"][0]["check"], "load")

    def test_cli_writes_json_and_csv_for_all_recipes(self) -> None:
        n_recipes = len([p for p in (REPO / "config" / "camera_recipes").glob("*.yaml") if p.stem != "default"])
        with tempfile.TemporaryDirectory() as d:
            js, cs = Path(d) / "r.json", Path(d) / "r.csv"
            argv = ["--repo-root", str(REPO), "--json-out", str(js), "--csv-out", str(cs), "--fail-only"]
            with self.assertRaises(SystemExit) as cm:
                run_tool_main(bve.main, argv)
            report = json.loads(js.read_text())
            rows = list(csv.DictReader(cs.open()))
        self.assertEqual(cm.exception.code, 0 if report["counts"].get("FAIL", 0) == 0 else 1)
        self.assertEqual(report["total"], n_recipes)
        self.assertEqual(len(rows), n_recipes)
        self.assertEqual(sum(report["counts"].values()), n_recipes)
        for row in rows:
            for col in bve._CSV_PARAM_COLUMNS:
                if col not in ("f_number", "pixel_pitch_um"):
                    self.assertNotEqual(row[col], "", f"{row['name']}: empty CSV column {col}")


def _write_reference_npz(repo: Path, neutral_reflectance: list[float]) -> None:
    lam = np.arange(380.0, 781.0, 1.0)
    refl = np.full((24, lam.size), 0.5)
    for k, r in enumerate(neutral_reflectance):
        refl[18 + k] = r
    out = repo / "scenes" / "generated"
    out.mkdir(parents=True)
    np.savez(out / "spectral_reference_1nm.npz", wavelength_nm=lam, illuminant=np.ones_like(lam), reflectance=refl)


class TestValidateColorchecker(unittest.TestCase):
    def test_neutral_ladder(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            self.assertTrue(vcc.check_neutral_luminance(repo), "missing reference should be skipped")
            _write_reference_npz(repo, [0.9, 0.6, 0.36, 0.2, 0.09, 0.03])
            self.assertTrue(vcc.check_neutral_luminance(repo))
        with tempfile.TemporaryDirectory() as d:
            _write_reference_npz(Path(d), [0.9, 0.6, 0.7, 0.2, 0.09, 0.03])
            self.assertFalse(vcc.check_neutral_luminance(Path(d)))

    def test_cli_summarises_exr_and_reports_status(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            _write_reference_npz(repo, [0.9, 0.6, 0.36, 0.2, 0.09, 0.03])
            exr = write_rgb_exr(repo / "img.exr", np.full((4, 6, 3), 0.25, dtype=np.float32))
            argv = ["--repo-root", str(repo), "--exr", str(exr), "--imgtool", str(repo / "none")]
            with self.assertRaises(SystemExit) as cm:
                run_tool_main(vcc.main, argv)
            self.assertEqual(cm.exception.code, 0)
            with self.assertRaises(SystemExit) as cm:
                run_tool_main(vcc.main, [*argv, "--render", "--pbrt", str(repo / "missing-pbrt")])
            self.assertEqual(cm.exception.code, 2)

    def test_summarize_exr_prints_stats(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            exr = write_rgb_exr(Path(d) / "img.exr", np.full((4, 6, 3), 0.25, dtype=np.float32))
            out = run_tool_main(lambda: vcc.summarize_exr(exr, None), [])
        self.assertIn("shape=(4, 6", out)


class TestImageQualityTargets(unittest.TestCase):
    def test_all_targets_outside_repo(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "iq"
            argv = ["--repo-root", str(REPO), "--out-dir", str(out), "--xres", "64", "--yres", "48"]
            run_tool_main(iq.main, argv)
            manifest = json.loads((out / "manifest.json").read_text())
            targets = {t["target"]: t for t in manifest["targets"]}
            self.assertEqual(set(targets), {"slanted_edge", "iso_noise", "siemens_star"})
            for t in targets.values():
                scene = Path(t["scene"])
                self.assertTrue(scene.is_absolute() and scene.is_file())
                text = scene.read_text()
                self.assertIn("WorldBegin", text)
                self.assertIn('"integer xresolution" [ 64 ]', text.replace("[64]", "[ 64 ]"))

    def test_geometry_builders(self) -> None:
        star = "\n".join(iq.build_siemens_star_lines(2.0, 2.0, spokes=8, radius_fraction=0.9))
        edge = "\n".join(iq.build_slanted_edge_lines(2.0, 1.5, angle_deg=5.0, edge_offset=0.1))
        iso = "\n".join(iq.build_iso_noise_lines(2.0, 1.5, 2, 3, 0.05, 0.9))
        self.assertGreaterEqual(star.count("Shape"), 8)
        self.assertIn("Shape", edge)
        self.assertGreaterEqual(iso.count("Shape"), 6)


class TestMunsellScenes(unittest.TestCase):
    def test_hue_token_helpers(self) -> None:
        self.assertEqual(munsell._hue_family_from_token("5YR"), "YR")
        self.assertEqual(munsell._hue_family_from_token(None), "N")
        self.assertEqual(munsell._hue_step_from_token("2.5R"), 2.5)
        self.assertIsNone(munsell._hue_step_from_token(None))

    def test_subsample_for_spd(self) -> None:
        wl = np.arange(380.0, 801.0, 1.0)
        swl, sval = munsell.subsample_for_spd(wl, wl / 1000.0, 10.0)
        self.assertEqual(swl[0], 380.0)
        self.assertTrue(np.allclose(np.diff(swl), 10.0))
        np.testing.assert_allclose(sval, swl / 1000.0)

    def test_unmatched_hue_writes_empty_index(self) -> None:
        # The Joensuu set has no neutral chips; this used to crash writing index.json.
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "munsell"
            run_tool_main(munsell.main, ["--repo-root", str(REPO), "--out-dir", str(out), "--hues", "N"])
            index = json.loads((out / "index.json").read_text())
        self.assertEqual(index["hues_generated"], [])

    def test_hue_scene_from_shipped_mat(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "munsell"
            argv = ["--repo-root", str(REPO), "--out-dir", str(out), "--hues", "R_2.5"]
            run_tool_main(munsell.main, [*argv, "--xres", "64", "--yres", "48", "--max-patches-per-scene", "6"])
            index = json.loads((out / "index.json").read_text())
            self.assertEqual(index["hues_generated"], ["R_2.5"])
            manifests = list(out.glob("*/munsell_*_manifest.json"))
            self.assertEqual(len(manifests), 1)
            m = json.loads(manifests[0].read_text())
            scene = Path(m["scene"])
            self.assertTrue(scene.is_file())
            spds = list(manifests[0].parent.glob("spd/patch_*.spd"))
            self.assertTrue(0 < len(spds) <= 6)
            self.assertEqual(scene.read_text().count('"spectrum reflectance"'), len(spds))


if __name__ == "__main__":
    unittest.main()
