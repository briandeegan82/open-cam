import math

from opencam_gui.core.dark_current import dark_current_multiplier


def test_doubling_rule_one_doubling_interval():
    m = dark_current_multiplier(26.0, 20.0, 6.0, 0.0)
    assert math.isclose(m, 2.0, rel_tol=1e-9)


def test_doubling_rule_identity_at_reference_temp():
    m = dark_current_multiplier(20.0, 20.0, 6.0, 0.0)
    assert math.isclose(m, 1.0, rel_tol=1e-9)


def test_arrhenius_identity_at_reference_temp():
    m = dark_current_multiplier(20.0, 20.0, 6.0, 0.63)
    assert math.isclose(m, 1.0, rel_tol=1e-6)


def test_arrhenius_increases_with_temperature():
    m_cold = dark_current_multiplier(10.0, 20.0, 6.0, 0.63)
    m_hot = dark_current_multiplier(50.0, 20.0, 6.0, 0.63)
    assert m_hot > m_cold > 0
