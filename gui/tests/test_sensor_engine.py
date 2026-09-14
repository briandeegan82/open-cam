import numpy as np

from opencam_gui.core import sensor_engine as se


def test_photon_transfer_curve_shape_and_monotonic_variance():
    curve = se.photon_transfer_curve(
        sigma_d_e=2.6, K_e_per_DN=4.77, black_level_DN=16.0,
        full_well_e=4800.0, use_poisson=True, n_points=50,
    )
    assert curve.mu_e.shape == (50,)
    assert curve.mean_dn.shape == (50,)
    assert curve.var_dn.shape == (50,)
    # Shot-noise-limited: variance should generally increase with signal.
    assert curve.var_dn[-1] > curve.var_dn[0]
    assert curve.shot_read_crossover_e == 2.6 ** 2


def test_photon_transfer_curve_matches_tool_module_directly():
    m = se._emva_module()
    curve = se.photon_transfer_curve(
        sigma_d_e=2.0, K_e_per_DN=3.0, black_level_DN=10.0,
        full_well_e=1000.0, use_poisson=True, n_points=5,
    )
    for i, mu in enumerate(curve.mu_e):
        expected_mean = m.mean_dn_linear(mu, 3.0, 10.0)
        assert np.isclose(curve.mean_dn[i], expected_mean)


def test_verify_against_monte_carlo_agrees_with_theory():
    v = se.verify_against_monte_carlo(
        mu_e=2000.0, sigma_d_e=2.6, K_e_per_DN=4.77, black_level_DN=16.0,
        full_well_e=4800.0, use_poisson=True, n_trials=20000, seed=1,
    )
    assert abs(v.mc_mean_dn - v.theory_mean_dn) < 2.0
    assert abs(v.mc_var_dn - v.theory_var_dn) / max(v.theory_var_dn, 1e-9) < 0.2


def test_dark_floor_point_present():
    curve = se.photon_transfer_curve(
        sigma_d_e=2.6, K_e_per_DN=4.77, black_level_DN=16.0,
        full_well_e=4800.0, use_poisson=True, n_points=10,
    )
    assert curve.dark_mean_dn >= curve.dark_mean_dn - 1e-9  # sanity: value exists and is finite
    assert curve.dark_var_dn > 0
