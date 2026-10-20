"""SWIR (to 2.5 um) data and InGaAs detector models (tools/swir_spectra.py, docs/SWIR.md)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import cfa_mosaic  # noqa: E402
import fog_mie  # noqa: E402
import swir_spectra as sw  # noqa: E402
import yaml  # noqa: E402
from qe_curves import read_csv_curve  # noqa: E402

LAM = np.arange(360.0, 2701.0, 1.0)


def test_ingaas_bandgap_matches_vurgaftman_lattice_matched_value():
    # In0.53Ga0.47As at 300 K: 0.74 eV (1.68 um), the textbook value.
    assert sw.ingaas_bandgap_ev(0.47, 300.0) == pytest.approx(0.737, abs=0.005)
    assert sw.varshni_ev("InP", 300.0) == pytest.approx(1.353, abs=0.002)  # Vurgaftman Varshni fit


@pytest.mark.parametrize(
    ("name", "temp_c", "hi_nm"),
    [
        # Hamamatsu G12180 / G12183 datasheet spectral response ranges (long-wavelength end).
        ("ingaas_standard", 25.0, 1700.0),
        ("ingaas_standard", -10.0, 1670.0),
        ("ingaas_standard", -20.0, 1650.0),
        ("ingaas_extended", 25.0, 2600.0),
        ("ingaas_extended", -10.0, 2570.0),
        ("ingaas_extended", -20.0, 2550.0),
    ],
)
def test_cutoff_and_its_temperature_shift_match_datasheets(name, temp_c, hi_nm):
    q = sw.DETECTORS[name].qe(LAM, temp_c)
    assert sw.response_range_nm(LAM, q)[1] == pytest.approx(hi_nm, abs=30.0)


def test_peak_responsivity_matches_datasheets():
    std = sw.responsivity_a_per_w(LAM, sw.DETECTORS["ingaas_standard"].qe(LAM))
    ext = sw.responsivity_a_per_w(LAM, sw.DETECTORS["ingaas_extended"].qe(LAM))
    i = {w: int(np.searchsorted(LAM, w)) for w in (900, 1300, 1550, 2300)}
    assert std[i[1550]] == pytest.approx(1.1, rel=0.1)  # G12180 typ. at lambda_p = 1.55 um
    assert std[i[1300]] == pytest.approx(0.9, rel=0.1)  # G12180 typ. at 1.3 um
    assert ext[i[2300]] == pytest.approx(1.3, rel=0.1)  # G12183 typ. at lambda_p = 2.3 um
    assert ext[i[900]] == pytest.approx(0.4, rel=0.15)  # G12183-210KA-03: 0.4 A/W at 900 nm
    assert sw.response_range_nm(LAM, sw.DETECTORS["ingaas_standard"].qe(LAM))[0] == pytest.approx(900, abs=20)


def test_visswir_responds_in_visible_and_standard_does_not():
    i = int(np.searchsorted(LAM, 550))
    assert sw.DETECTORS["ingaas_visswir"].qe(LAM)[i] > 0.5
    assert sw.DETECTORS["ingaas_standard"].qe(LAM)[i] < 0.01
    assert np.all(sw.DETECTORS["ingaas_extended"].qe(LAM) <= 1.0)


@pytest.mark.parametrize("name", list(sw.DETECTORS))
def test_committed_qe_csvs_match_model(name):
    wl, q = read_csv_curve(REPO / f"spectra/QE/swir/QE_{name}.csv")
    assert wl[0] <= 360 and wl[-1] >= 2600
    np.testing.assert_allclose(q, sw.DETECTORS[name].qe(wl), atol=2e-6)


def test_g173_sun_has_water_vapour_bands():
    wl, e = read_csv_curve(REPO / "spectra/illuminant/swir/sun_astm_g173_direct.csv")
    at = lambda w: float(np.interp(w, wl, e))  # noqa: E731
    assert at(1000) > 0.5 and at(1600) > 0.2 and at(2200) > 0.05
    assert at(1380) < 0.02 * at(1250) and at(1880) < 0.02 * at(1650)


def test_planck_tungsten_peaks_near_wien():
    wl, e = read_csv_curve(REPO / "spectra/illuminant/swir/tungsten_2856K.csv")
    assert wl[np.argmax(e)] == pytest.approx(2.897771955e6 / 2856.0, abs=2.0)


def test_water_and_ice_absorb_strongly_in_swir():
    k = {w: fog_mie.water_index(w).imag for w in (1100, 1450, 1940)}
    assert k[1450] > 50 * k[1100] and k[1940] > 3 * k[1450]
    assert fog_mie.water_index(550) == complex(1.333, 2.0e-9) or abs(fog_mie.water_index(550).imag - 2e-9) < 1e-12
    d = np.loadtxt(REPO / "spectra/ice/warren_brandt_2008.csv", delimiter=",", comments="#", skiprows=5)
    assert d[-1, 0] >= 2500 and np.interp(2000, d[:, 0], d[:, 2]) > 1e-4


def test_usgs_chart_reaches_2500_nm_with_swir_water_bands():
    wl, leaf = read_csv_curve(REPO / "spectra/usgs_nir/01_oak_leaf_fresh.csv")
    assert wl[-1] >= 2500
    r = lambda w: float(np.interp(w, wl, leaf))  # noqa: E731
    # Liquid water in a fresh leaf: reflectance dips at 1.45 and 1.94 um.
    assert r(1450) < 0.8 * r(1650) and r(1940) < 0.6 * r(2200)


@pytest.mark.parametrize("recipe", ["swir_ingaas_vga_15um", "swir_ingaas_visswir_5um", "swir_ingaas_extended_15um"])
def test_swir_sensor_models_resolve_to_mono_layout(recipe):
    rec = yaml.safe_load((REPO / f"config/camera_recipes/{recipe}.yaml").read_text())
    sm = yaml.safe_load((REPO / f"config/sensor_models/{rec['sensor_model']}.yaml").read_text())
    layout = cfa_mosaic.resolve_layout(sm["cfa"])
    assert layout.channels == ("S",) and layout.period == (1, 1)
    qe = cfa_mosaic.qe_stack_for_layout(REPO, layout, np.array([550.0, 1550.0, 2200.0]))
    assert qe[0, 1] > 0.5
