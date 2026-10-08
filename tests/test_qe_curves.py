"""Shared spectral-curve loading and effective f-number selection."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import apply_emva_noise  # noqa: E402
import pbrt_spectral_exr_to_electrons  # noqa: E402
import spectral_sensor_forward  # noqa: E402
from camera_model import effective_f_number  # noqa: E402
from qe_curves import load_qe_curves_rgb, read_csv_curve  # noqa: E402


class TestReadCsvCurve(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, text: str) -> Path:
        p = self.dir / "curve.csv"
        p.write_text(text)
        return p

    def test_skips_comments_blank_and_malformed_lines_and_sorts(self) -> None:
        wl, v = read_csv_curve(self._write("# header\n\n500, 0.5\nnot-a-row\n400,0.1\n600,nan\n450,0.3\n"))
        np.testing.assert_array_equal(wl, [400.0, 450.0, 500.0])
        np.testing.assert_array_equal(v, [0.1, 0.3, 0.5])

    def test_empty_curve_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "no data"):
            read_csv_curve(self._write("# nothing\n"))

    def test_normalized_axis_is_remapped_unless_strict(self) -> None:
        path = self._write("0,0.1\n0.5,0.2\n1,0.3\n")
        with contextlib.redirect_stderr(io.StringIO()):
            wl, _ = read_csv_curve(path)
        np.testing.assert_allclose(wl, [380.0, 605.0, 830.0])
        with self.assertRaisesRegex(ValueError, "strict QE validation"):
            read_csv_curve(path, strict_wavelength_axis=True)

    def test_loads_rgb_triplet_from_config(self) -> None:
        for ch, val in (("r", 0.1), ("g", 0.2), ("b", 0.3)):
            (self.dir / f"{ch}.csv").write_text(f"400,{val}\n700,{val}\n")
        r, g, b = load_qe_curves_rgb(self.dir, {"red_csv": "r.csv", "green_csv": "g.csv", "blue_csv": "b.csv"})
        self.assertEqual([float(c[1][0]) for c in (r, g, b)], [0.1, 0.2, 0.3])

    def test_tools_share_one_implementation(self) -> None:
        for mod in (apply_emva_noise, spectral_sensor_forward, pbrt_spectral_exr_to_electrons):
            self.assertIs(mod.read_csv_curve, read_csv_curve)
            self.assertIs(mod.load_qe_curves_rgb, load_qe_curves_rgb)


class TestEffectiveFNumber(unittest.TestCase):
    def test_non_realistic_lens_uses_sensor_f_number(self) -> None:
        self.assertEqual(effective_f_number({"f_number": 4.0}, None), 4.0)
        self.assertEqual(effective_f_number({"f_number": 4.0}, {"camera": "thinlens", "focal_length_mm": 50}), 4.0)
        self.assertEqual(effective_f_number({}, {}), 2.8)

    def test_realistic_lens_uses_prescription(self) -> None:
        lens = {"camera": "realistic", "focal_length_mm": 50.0, "realistic_aperture_diameter_mm": 12.5}
        self.assertEqual(effective_f_number({"f_number": 2.0}, lens), 4.0)

    def test_realistic_lens_without_prescription_warns_and_falls_back(self) -> None:
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            got = effective_f_number({"f_number": 5.6}, {"camera": "Realistic"}, tag="t")
        self.assertEqual(got, 5.6)
        self.assertIn("warning [t]", err.getvalue())


if __name__ == "__main__":
    unittest.main()
