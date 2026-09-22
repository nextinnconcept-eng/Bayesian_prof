"""Tests for J-V curve processing (Chunk 2.5), verified against a
synthetic ideal single-diode J-V curve. Voc/Jsc have closed-form ground
truth; FF/PCE are checked against an INDEPENDENT continuous optimization
of the diode equation (scipy.optimize.minimize_scalar), not derived from
this module's own grid-based implementation -- a non-tautological check.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq, minimize_scalar

from dssc_discovery.chemistry.jv import (
    JVProcessingError,
    compute_fill_factor,
    compute_jv_descriptors,
    compute_open_circuit_voltage,
    compute_short_circuit_current_density,
    parse_jv_csv,
)
from dssc_discovery.domain.enums import IlluminationProtocol

JSC_TRUE = 12.0  # mA/cm^2
VOC_TRUE = 0.68  # V
N_IDEALITY = 1.5
VT = 0.02585  # thermal voltage, ~300K
J0 = JSC_TRUE / (np.exp(VOC_TRUE / (N_IDEALITY * VT)) - 1)


def _diode_current(voltage: np.ndarray) -> np.ndarray:
    return JSC_TRUE - J0 * (np.exp(voltage / (N_IDEALITY * VT)) - 1)


def _synthetic_jv_curve(v_max: float = 0.75, step: float = 0.005):
    voltage = np.arange(0.0, v_max + step, step)
    current = _diode_current(voltage)
    return voltage, current


def _continuous_ground_truth():
    """Independent ground truth via continuous optimization on the
    diode equation itself -- not this module's grid-based algorithm.
    """
    voc = brentq(lambda v: _diode_current(np.array([v]))[0], 0, 1.0)

    def neg_power(v):
        return -(v * _diode_current(np.array([v]))[0])

    result = minimize_scalar(neg_power, bounds=(0, voc), method="bounded", options={"xatol": 1e-10})
    v_mp = result.x
    j_mp = _diode_current(np.array([v_mp]))[0]
    ff = (v_mp * j_mp) / (voc * JSC_TRUE)
    pce = (v_mp * j_mp) / 100.0 * 100.0  # Pin = 100 mW/cm^2 (AM1.5G 1 sun)
    return voc, ff, pce


VOC_CONTINUOUS, FF_CONTINUOUS, PCE_CONTINUOUS = _continuous_ground_truth()


# --- Voc / Jsc: closed-form / definitional ground truth ---------------------


def test_jsc_recovers_short_circuit_current():
    voltage, current = _synthetic_jv_curve()
    jsc = compute_short_circuit_current_density(voltage, current)
    assert jsc == pytest.approx(JSC_TRUE, abs=1e-6)


def test_voc_recovers_open_circuit_voltage():
    voltage, current = _synthetic_jv_curve()
    voc = compute_open_circuit_voltage(voltage, current)
    assert voc == pytest.approx(VOC_TRUE, abs=1e-3)


# --- FF / PCE: checked against an independent continuous optimum ------------


def test_fill_factor_matches_independent_continuous_optimum():
    voltage, current = _synthetic_jv_curve()
    jsc = compute_short_circuit_current_density(voltage, current)
    voc = compute_open_circuit_voltage(voltage, current)
    fill_factor, v_mp, j_mp = compute_fill_factor(voltage, current, voc, jsc)
    assert fill_factor == pytest.approx(FF_CONTINUOUS, rel=1e-3)


def test_full_descriptor_set_matches_independent_continuous_optimum():
    voltage, current = _synthetic_jv_curve()
    descriptors = compute_jv_descriptors(voltage, current, IlluminationProtocol.AM1_5G_1_SUN)
    assert descriptors.open_circuit_voltage_v == pytest.approx(VOC_CONTINUOUS, abs=1e-3)
    assert descriptors.fill_factor == pytest.approx(FF_CONTINUOUS, rel=1e-3)
    assert descriptors.power_conversion_efficiency_percent == pytest.approx(
        PCE_CONTINUOUS, rel=1e-3
    )


def test_finer_sampling_converges_closer_to_continuous_optimum():
    coarse_voltage, coarse_current = _synthetic_jv_curve(step=0.02)
    fine_voltage, fine_current = _synthetic_jv_curve(step=0.001)

    coarse = compute_jv_descriptors(
        coarse_voltage, coarse_current, IlluminationProtocol.AM1_5G_1_SUN
    )
    fine = compute_jv_descriptors(fine_voltage, fine_current, IlluminationProtocol.AM1_5G_1_SUN)

    coarse_error = abs(coarse.fill_factor - FF_CONTINUOUS)
    fine_error = abs(fine.fill_factor - FF_CONTINUOUS)
    assert fine_error < coarse_error


# --- determinism / robustness ------------------------------------------------


def test_deterministic_given_the_same_curve():
    voltage, current = _synthetic_jv_curve()
    first = compute_jv_descriptors(voltage, current, IlluminationProtocol.AM1_5G_1_SUN)
    second = compute_jv_descriptors(voltage, current, IlluminationProtocol.AM1_5G_1_SUN)
    assert first == second


def test_half_sun_illumination_doubles_reported_pce_for_same_curve():
    # same physical curve, but Pin is halved -- PCE (power out / power
    # in) must correspondingly be reported as roughly double.
    voltage, current = _synthetic_jv_curve()
    full_sun = compute_jv_descriptors(voltage, current, IlluminationProtocol.AM1_5G_1_SUN)
    half_sun = compute_jv_descriptors(voltage, current, IlluminationProtocol.AM1_5G_HALF_SUN)
    assert half_sun.power_conversion_efficiency_percent == pytest.approx(
        2 * full_sun.power_conversion_efficiency_percent, rel=1e-6
    )


def test_undefined_illumination_power_raises():
    voltage, current = _synthetic_jv_curve()
    with pytest.raises(JVProcessingError, match="incident power"):
        compute_jv_descriptors(voltage, current, IlluminationProtocol.DARK)


def test_curve_that_never_crosses_zero_raises():
    voltage = np.array([0.0, 0.1, 0.2, 0.3])
    current = np.array([5.0, 4.0, 3.0, 2.0])  # never reaches J=0
    with pytest.raises(JVProcessingError, match="Voc"):
        compute_open_circuit_voltage(voltage, current)


# --- parsing ------------------------------------------------------------------


def test_parse_jv_csv_sorts_out_of_order_rows():
    csv_text = "voltage_v,current_density_ma_cm2\n0.5,2\n0.0,12\n0.25,8\n"
    voltage, current = parse_jv_csv(csv_text.encode("utf-8"))
    assert list(voltage) == [0.0, 0.25, 0.5]


def test_parse_jv_csv_rejects_too_few_points():
    csv_text = "voltage_v,current_density_ma_cm2\n0.0,12\n0.5,2\n"
    with pytest.raises(JVProcessingError, match="at least 3 points"):
        parse_jv_csv(csv_text.encode("utf-8"))


def test_parse_jv_csv_rejects_malformed_row():
    csv_text = "voltage_v,current_density_ma_cm2\n0.0,12\n0.2,oops\n0.5,2\n"
    with pytest.raises(JVProcessingError, match="Malformed"):
        parse_jv_csv(csv_text.encode("utf-8"))
