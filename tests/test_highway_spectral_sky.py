"""Spectral (CIE-daylight-basis) sky for the highway scene: data, Hosek-Wilkie port, validation."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import build_highway_scene as hw
import highway_spectral_sky as ss
import numpy as np
import yaml
from highway_sky import analytic_clear_sky, clear_sky_illuminance_lux, equal_area_directions, solar_direct_spectrum
from synthetic_data import REPO

SKYMODEL_SRC = REPO / "third_party" / "pbrt-v4" / "src" / "ext" / "skymodel"


def build_offline(out: Path, *extra: str) -> tuple[dict, str]:
    out.mkdir(parents=True, exist_ok=True)
    m = yaml.safe_load((REPO / "config" / "highway_assets.yaml").read_text())
    m["cache_dir"] = str(out / "empty_cache")
    (out / "assets.yaml").write_text(yaml.safe_dump(m))
    hw.main(
        ["--out-dir", str(out / "scene"), "--asset-manifest", str(out / "assets.yaml"), "--allow-missing-assets"]
        + ["--xres", "48", "--yres", "32", "--pixelsamples", "4", *extra]
    )
    scene = out / "scene"
    return json.loads((scene / "highway_manifest.json").read_text()), (scene / "highway.pbrt").read_text()


def hw_clear_sky_reference(elev: float) -> float:
    e_dn, e_dif = clear_sky_illuminance_lux(elev)
    return e_dn * np.sin(np.radians(elev)) + e_dif


def global_horizontal_cct(elev: float) -> float:
    """CCT of sun + Hosek sky on a horizontal plane at the builder's absolute illuminances."""
    wl, e_sky = ss.horizontal_spectrum(ss.hosek_weight_map(96, elev, 3.0))
    sun = solar_direct_spectrum(wl, elev)
    e_dn, e_dif = clear_sky_illuminance_lux(elev)
    y = lambda s: ss.spectrum_xyz(wl, s)[1]  # noqa: E731
    e = sun / y(sun) * e_dn * np.sin(np.radians(elev)) + e_sky / y(e_sky) * e_dif
    return ss.cct_duv(ss.spectrum_xyz(wl, e))[0]


class TestDaylightBasis(unittest.TestCase):
    def test_cie_table_pinned(self) -> None:
        digest = hashlib.sha256(ss.DAYLIGHT_CSV.read_bytes()).hexdigest()
        self.assertEqual(digest, "4056cf7c1afb23fb8820ff9cbc383a80005acdffe1c0a6945c6032616e08c6be")

    def test_d65_and_cct(self) -> None:
        wl, d65 = ss.daylight_spd(6504.0)
        xyz = ss.spectrum_xyz(wl, d65)
        np.testing.assert_allclose(xyz[:2] / xyz.sum(), [0.3127, 0.3290], atol=6e-4)
        cct, duv = ss.cct_duv(xyz)
        self.assertAlmostEqual(cct, 6504.0, delta=40.0)
        self.assertAlmostEqual(duv, 0.0032, delta=0.001)  # daylight locus sits ~0.003 above Planck
        wl = np.arange(360.0, 831.0) * 1e-9
        planck = 1.0 / (wl**5 * np.expm1(1.4388e-2 / (wl * 3000.0)))
        self.assertAlmostEqual(ss.cct_duv(ss.spectrum_xyz(wl * 1e9, planck))[0], 3000.0, delta=5.0)

    def test_basis_physical(self) -> None:
        wl, b = ss.basis_spectra()
        self.assertTrue((b[:, wl >= 360] >= -1e-6).all())
        np.testing.assert_allclose(ss.spectrum_xyz(wl, b)[:, 1], 1.0, rtol=1e-9)
        # the basis triangle contains the CIE daylight locus 4000 K - 25 kK
        for t in (4000.0, 5000.0, 6504.0, 10000.0, 25000.0):
            c = ss.xyz_to_coeffs(ss.spectrum_xyz(*ss.daylight_spd(t)))
            self.assertEqual(ss.out_of_gamut_fraction(c[None]), 0.0, t)

    def test_rgb_roundtrip(self) -> None:
        rgb = np.array([[0.3, 0.5, 1.0], [1.0, 1.0, 1.0], [1.0, 0.8, 0.6], [0.05, 0.9, 0.05]])
        w = ss.rgb_to_weights(rgb)
        xyz = ss.rgb_to_xyz(rgb)
        np.testing.assert_allclose(w.sum(-1), xyz[:, 1], rtol=1e-9)  # luminance always kept
        self.assertTrue((w >= 0).all())
        wl, spd = ss.weight_map_spectra(w[:3])
        np.testing.assert_allclose(ss.spectrum_xyz(wl, spd), xyz[:3], rtol=1e-6)  # in gamut: same XYZ
        self.assertGreater(ss.out_of_gamut_fraction(ss.xyz_to_coeffs(xyz)), 0.0)  # saturated green is not


class TestHosekWilkie(unittest.TestCase):
    @unittest.skipUnless(shutil.which("cc") and (SKYMODEL_SRC / "ArHosekSkyModel.c").is_file(), "needs cc + pbrt src")
    def test_matches_reference_c(self) -> None:
        prog = r"""
#include <stdio.h>
#include "ArHosekSkyModel.h"
int main(void) {
  double els[3] = {5, 30, 70}, turbs[3] = {1.5, 3.0, 7.3}, albs[2] = {0.1, 0.6};
  for (int e = 0; e < 3; e++) for (int t = 0; t < 3; t++) for (int a = 0; a < 2; a++) {
    ArHosekSkyModelState *s = arhosekskymodelstate_alloc_init(els[e] * 0.017453292519943295, turbs[t], albs[a]);
    for (int k = 0; k < 4; k++) for (int w = 0; w < 11; w++)
      printf("%g %g %g %.17g %.17g %.17g\n", els[e], turbs[t], albs[a], k * 0.4, 0.1 + k * 0.7,
             arhosekskymodel_radiance(s, k * 0.4, 0.1 + k * 0.7, 320 + 40 * w));
    arhosekskymodelstate_free(s);
  }
  return 0;
}
"""
        with tempfile.TemporaryDirectory() as td:
            src, exe = Path(td) / "ref.c", Path(td) / "ref"
            src.write_text(prog)
            subprocess.run(
                ["cc", "-O1", f"-I{SKYMODEL_SRC}", str(src), str(SKYMODEL_SRC / "ArHosekSkyModel.c"), "-lm", "-o"]
                + [str(exe)],
                check=True,
                capture_output=True,
            )
            ref = np.loadtxt(subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout.splitlines())
        for el, turb, alb in {tuple(r[:3]) for r in ref}:
            r = ref[(ref[:, 0] == el) & (ref[:, 1] == turb) & (ref[:, 2] == alb)]
            lum = ss.HosekWilkieSky(el, turb, alb).radiance(np.cos(r[::11, 3]), np.cos(r[::11, 4])).ravel()
            np.testing.assert_allclose(lum, r[:, 5], rtol=1e-10, atol=1e-12)

    def test_absolute_illuminance_vs_clear_sky_fit(self) -> None:
        # Hosek's own absolute diffuse horizontal illuminance vs the builder's IESNA clear-sky
        # fit (Lighting Handbook, 0.8 + 15.5 sqrt(sin h) klux): within 30 % for 10-70 deg.
        for el in (10.0, 30.0, 45.0, 70.0):
            w = ss.hosek_weight_map(96, el, 3.0)
            e_hosek = ss.KM * ss.map_horizontal_luminance_integral(w)
            self.assertAlmostEqual(e_hosek / clear_sky_illuminance_lux(el)[1], 1.0, delta=0.3, msg=el)

    def test_luminance_distribution_vs_cie_type12(self) -> None:
        # Relative sky luminance pattern vs CIE S 011 / ISO 15469 standard clear sky type 12
        # (published from sky-scanner measurements), away from the circumsolar region.
        n = 96
        d = equal_area_directions(n)
        for el in (20.0, 45.0):
            zs = np.radians(90.0 - el)
            far = d @ np.array([np.sin(zs), 0.0, np.cos(zs)]) < np.cos(np.radians(10.0))
            m = (d[..., 2] > 0.05) & far
            y_cie = analytic_clear_sky(n, el) @ np.array([0.2126, 0.7152, 0.0722])
            y_hw = ss.hosek_weight_map(n, el, 3.0).sum(-1)
            self.assertGreater(np.corrcoef(np.log(y_cie[m]), np.log(y_hw[m]))[0, 1], 0.85, el)

    def test_sky_cct(self) -> None:
        # Diffuse clear skylight is far bluer than daylight (CIE daylight locus 4000-25000 K, Judd et
        # al. 1964), zenith bluer than the hemisphere, and close to the daylight locus (Duv ~ +0.003).
        prev = 0.0
        for el in (10.0, 30.0, 60.0):
            s = ss.sky_colour_summary(ss.hosek_weight_map(96, el, 3.0))
            self.assertTrue(8000.0 < s["horizontal_cct_k"] < 25000.0, s)
            self.assertGreater(s["zenith_cct_k"], s["horizontal_cct_k"])
            self.assertLess(abs(s["horizontal_duv"] - 0.0031), 0.012)
            self.assertGreater(s["horizontal_cct_k"], prev)  # the sky whitens as the sun sets
            prev = s["horizontal_cct_k"]

    def test_global_daylight_cct(self) -> None:
        # Sun + sky on a horizontal plane: typical daylight, D55-D65 (CIE 015:2018) at mid elevations.
        for el in (20.0, 45.0, 70.0):
            self.assertTrue(5000.0 < global_horizontal_cct(el) < 6600.0, el)


class TestBuilderSpectralSky(unittest.TestCase):
    def test_hosek_scene(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            m, scene = build_offline(Path(td), "--sky", "hosek", "--sun-elevation", "40")
            m_rgb, _ = build_offline(Path(td) / "rgb", "--sky", "analytic", "--sun-elevation", "40")
            sky = m["lighting"]["sky"]
            self.assertIn('"spectrum L" ["spd/sky_basis_0.spd" "spd/sky_basis_1.spd" "spd/sky_basis_2.spd"]', scene)
            exr = Path(td) / "scene" / "textures" / "sky_hosek_e40.0_t3.00_daylight.exr"
            self.assertEqual(ss.read_weight_exr(exr).shape, (ss.HOSEK_MAP_SIZE, ss.HOSEK_MAP_SIZE, 3))
            for s in sky["spectral"]["spectra"]:
                self.assertTrue((Path(td) / "scene" / s).is_file())
            # absolute radiometry identical to the RGB analytic sky (builder clear-sky fit)
            self.assertAlmostEqual(sky["illuminance_horizontal_lux"], clear_sky_illuminance_lux(40.0)[1])
            for k in ("reference_illuminance_lux", "reference_illuminance_exr_lux"):
                self.assertAlmostEqual(m["lighting"][k], m_rgb["lighting"][k])
            self.assertAlmostEqual(sky["spectral"]["model_illuminance_horizontal_lux"] / 13227.0, 1.0, delta=0.3)
            self.assertFalse(any("RGB (HDRI or analytic)" in a for a in m["approximations"]))

    def test_hosek_with_haze(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            m, scene = build_offline(Path(td), "--sky", "hosek", "--haze", "hazy")
            self.assertIn("sky_basis_0.spd", scene)
            self.assertIn('Integrator "volpath"', scene)
            self.assertLess(m["lighting"]["reference_illuminance_lux"], hw_clear_sky_reference(45.0))

    def test_spectral_sky_refused_at_night(self) -> None:
        with tempfile.TemporaryDirectory() as td, self.assertRaises(SystemExit):
            build_offline(Path(td), "--sky", "hosek", "--time-of-day", "night")

    def test_weights_to_rgb_keeps_luminance(self) -> None:
        w = ss.hosek_weight_map(32, 30.0)
        np.testing.assert_allclose(ss.weights_to_rgb(w) @ ss.SRGB_TO_XYZ[1], w.sum(-1), rtol=1e-9, atol=1e-12)

    def test_daylight_conversion_of_rgb_map_and_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            m, scene = build_offline(Path(td), "--sky", "analytic", "--sky-spectrum", "daylight")
            self.assertIn("sky_cie12_45.0_daylight.exr", scene)
            self.assertEqual(m["lighting"]["sky"]["spectral"]["out_of_gamut_luminance_fraction"], 0.0)
            _, scene_rgb = build_offline(Path(td) / "rgb", "--sky", "analytic")
            self.assertNotIn("sky_basis", scene_rgb)  # default stays the stock RGB sky


if __name__ == "__main__":
    unittest.main()
