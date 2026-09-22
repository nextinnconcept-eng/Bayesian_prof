"""Tests for EIS equivalent-circuit fitting (Chunk 2.4), verified against
synthetic impedance data generated from KNOWN circuit parameters -- and
against two empirically-confirmed non-convergence scenarios (non-finite
input data; an exhausted iteration budget), not a fabricated "this should
fail" case.
"""

from __future__ import annotations

import numpy as np
import pytest

from dssc_discovery.chemistry.eis import (
    EISProcessingError,
    fit_eis_spectrum,
    parse_eis_csv,
)

# Ground truth: realistic DSSC-scale parameters
RS_TRUE, RCT_TRUE, CCE_TRUE, RREC_TRUE, CMU_TRUE = 15.0, 50.0, 2e-6, 300.0, 5e-5
TAU_TRUE_MS = RREC_TRUE * CMU_TRUE * 1000.0


def _synthetic_impedance(
    rs=RS_TRUE, rct=RCT_TRUE, cce=CCE_TRUE, rrec=RREC_TRUE, cmu=CMU_TRUE, n_points=60
):
    frequency = np.logspace(-1, 5, n_points)  # 0.1 Hz to 100 kHz
    omega = 2 * np.pi * frequency
    z_ce = rct / (1 + 1j * omega * rct * cce)
    z_rec = rrec / (1 + 1j * omega * rrec * cmu)
    z = rs + z_ce + z_rec
    return frequency, z.real, z.imag


def _eis_csv_bytes(frequency, z_real, z_imag) -> bytes:
    lines = ["frequency_hz,z_real_ohm,z_imag_ohm"]
    lines += [f"{f},{r},{i}" for f, r, i in zip(frequency, z_real, z_imag)]
    return "\n".join(lines).encode("utf-8")


# --- fit recovers known ground truth -----------------------------------------


def test_fit_recovers_series_resistance():
    frequency, z_real, z_imag = _synthetic_impedance()
    result = fit_eis_spectrum(frequency, z_real, z_imag)
    assert result.converged
    assert result.series_resistance_ohm == pytest.approx(RS_TRUE, rel=1e-3)


def test_fit_recovers_charge_transfer_and_recombination_resistance():
    frequency, z_real, z_imag = _synthetic_impedance()
    result = fit_eis_spectrum(frequency, z_real, z_imag)
    assert result.charge_transfer_resistance_ohm == pytest.approx(RCT_TRUE, rel=1e-3)
    assert result.recombination_resistance_ohm == pytest.approx(RREC_TRUE, rel=1e-3)


def test_fit_recovers_electron_lifetime():
    frequency, z_real, z_imag = _synthetic_impedance()
    result = fit_eis_spectrum(frequency, z_real, z_imag)
    assert result.electron_lifetime_ms == pytest.approx(TAU_TRUE_MS, rel=1e-3)


def test_fit_quality_is_near_perfect_on_clean_synthetic_data():
    frequency, z_real, z_imag = _synthetic_impedance()
    result = fit_eis_spectrum(frequency, z_real, z_imag)
    assert result.fit_r_squared == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    "rs,rct,cce,rrec,cmu",
    [
        (5.0, 20.0, 1e-6, 100.0, 2e-5),
        (30.0, 100.0, 5e-6, 500.0, 1e-4),
        (10.0, 40.0, 3e-6, 200.0, 3e-5),
    ],
)
def test_fit_recovers_parameters_across_different_circuit_scales(rs, rct, cce, rrec, cmu):
    frequency, z_real, z_imag = _synthetic_impedance(rs=rs, rct=rct, cce=cce, rrec=rrec, cmu=cmu)
    result = fit_eis_spectrum(frequency, z_real, z_imag)
    assert result.converged
    assert result.series_resistance_ohm == pytest.approx(rs, rel=1e-2)
    assert result.charge_transfer_resistance_ohm == pytest.approx(rct, rel=1e-2)
    assert result.recombination_resistance_ohm == pytest.approx(rrec, rel=1e-2)


def test_fit_is_deterministic_given_the_same_spectrum():
    frequency, z_real, z_imag = _synthetic_impedance()
    first = fit_eis_spectrum(frequency, z_real, z_imag)
    second = fit_eis_spectrum(frequency, z_real, z_imag)
    assert first == second


# --- fit quality is always disclosed, not gated on optimizer "success" ------


def test_optimizer_success_is_not_a_reliable_quality_signal_on_pure_noise():
    # empirically confirmed: least_squares reports success=True even
    # fitting 5 parameters to pure random noise -- the fit is still
    # "converged" in scipy's sense, but R^2 discloses it's meaningless.
    rng = np.random.default_rng(0)
    frequency = np.logspace(-1, 5, 60)
    z_real = rng.uniform(-50, 50, size=60)
    z_imag = rng.uniform(-50, 50, size=60)

    result = fit_eis_spectrum(frequency, z_real, z_imag)

    assert result.converged is True  # optimizer terminated normally
    assert result.fit_r_squared is not None
    assert result.fit_r_squared < 0.5  # but the quality score discloses it's bad


# --- explicit, non-crashing failure path -------------------------------------


def test_non_finite_input_data_triggers_documented_failure_not_a_crash():
    frequency, z_real, z_imag = _synthetic_impedance()
    z_real = z_real.copy()
    z_real[0] = np.nan  # simulates an instrument glitch

    result = fit_eis_spectrum(frequency, z_real, z_imag)  # must not raise

    assert result.converged is False
    assert result.series_resistance_ohm is None
    assert result.charge_transfer_resistance_ohm is None
    assert result.recombination_resistance_ohm is None
    assert result.electron_lifetime_ms is None
    assert result.fit_r_squared is None
    assert result.failure_reason is not None


def test_exhausted_iteration_budget_triggers_documented_failure_not_a_crash():
    frequency, z_real, z_imag = _synthetic_impedance()

    result = fit_eis_spectrum(frequency, z_real, z_imag, max_nfev=1)  # must not raise

    assert result.converged is False
    assert result.failure_reason is not None
    assert "did not converge" in result.failure_reason.lower()


# --- parsing --------------------------------------------------------------


def test_parse_eis_csv_recovers_ground_truth_shape():
    frequency, z_real, z_imag = _synthetic_impedance()
    raw_bytes = _eis_csv_bytes(frequency, z_real, z_imag)
    parsed_frequency, parsed_z_real, parsed_z_imag = parse_eis_csv(raw_bytes)
    assert len(parsed_frequency) == len(frequency)
    assert parsed_frequency[0] <= parsed_frequency[-1]  # sorted ascending


def test_parse_eis_csv_sorts_out_of_order_rows():
    csv_text = (
        "frequency_hz,z_real_ohm,z_imag_ohm\n"
        "10000,3,-0.5\n1000,5,-1\n1,20,-8\n100,10,-5\n50,15,-6\n"
    )
    frequency, z_real, z_imag = parse_eis_csv(csv_text.encode("utf-8"))
    assert list(frequency) == [1.0, 50.0, 100.0, 1000.0, 10000.0]


def test_parse_eis_csv_rejects_too_few_points():
    csv_text = "frequency_hz,z_real_ohm,z_imag_ohm\n1,20,-8\n100,10,-5\n"
    with pytest.raises(EISProcessingError, match="at least"):
        parse_eis_csv(csv_text.encode("utf-8"))


def test_parse_eis_csv_rejects_malformed_row():
    frequency, z_real, z_imag = _synthetic_impedance()
    lines = ["frequency_hz,z_real_ohm,z_imag_ohm"]
    lines += [f"{f},{r},{i}" for f, r, i in zip(frequency, z_real, z_imag)]
    lines[3] = "100,not-a-number,-5"
    csv_text = "\n".join(lines)
    with pytest.raises(EISProcessingError, match="Malformed"):
        parse_eis_csv(csv_text.encode("utf-8"))
