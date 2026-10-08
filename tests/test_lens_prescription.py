"""Traced f-number of pbrt lens prescriptions (tools/lens_prescription.py)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lens_prescription import load_lens_file, paraxial_focal_lengths_mm, traced_f_number
from synthetic_data import REPO

LENSES = REPO / "config" / "lenses"


class TestLensPrescription(unittest.TestCase):
    def test_paraxial_focal_length_matches_file_scaling(self) -> None:
        for name, f in (("wide_22mm.dat", 22.0), ("dgauss.50mm.dat", 50.0), ("telephoto.250mm.dat", 250.0)):
            efl, bfd = paraxial_focal_lengths_mm(load_lens_file(LENSES / name))
            self.assertAlmostEqual(efl, f, delta=0.01 * f, msg=name)
            self.assertGreater(bfd, 0.0)

    def test_traced_f_number_of_shipped_lenses(self) -> None:
        # pbrt renders (pinhole vs traced, matched framing) measured f/2.88 and f/2.07 off-axis.
        self.assertAlmostEqual(traced_f_number(str(LENSES / "wide_22mm.dat")), 2.79, delta=0.01)
        self.assertAlmostEqual(traced_f_number(str(LENSES / "dgauss.50mm.dat")), 2.02, delta=0.01)

    def test_aperture_clamped_to_stop_like_pbrt(self) -> None:
        lens = str(LENSES / "dgauss.50mm.dat")
        self.assertEqual(traced_f_number(lens, 25.0), traced_f_number(lens, 17.1))
        self.assertAlmostEqual(traced_f_number(lens, 12.34), 2.80, delta=0.01)

    def test_small_stop_approaches_paraxial_entrance_pupil(self) -> None:
        # At a small stop the marginal ray is paraxial: N -> scale * N_full_paraxial.
        lens = str(LENSES / "wide_22mm.dat")
        ratio = traced_f_number(lens, 1.0) / traced_f_number(lens, 2.0)
        self.assertAlmostEqual(ratio, 2.0, delta=0.01)

    def test_singlet_matches_closed_form(self) -> None:
        # Equiconvex singlet behind the stop (pbrt has no flat refracting surfaces): at a small
        # stop the entrance pupil is the stop, so N = EFL / D.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "singlet.dat"
            path.write_text("0 1.0 0 2.0\n100.0 3.0 1.5 20.0\n-100.0 0 1 20.0\n")
            efl, _ = paraxial_focal_lengths_mm(load_lens_file(path))
            self.assertAlmostEqual(efl, 100.5, delta=0.1)
            self.assertAlmostEqual(traced_f_number(str(path)), efl / 2.0, delta=0.002 * efl / 2.0)

    def test_lens_file_without_stop_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nostop.dat"
            path.write_text("51.5 3.0 1.515 20.0\n0 0 1 20.0\n".replace("0 0 1", "-1000 0 1"))
            with self.assertRaises(ValueError):
                load_lens_file(path)


if __name__ == "__main__":
    unittest.main()
