"""Tests for UV-Vis descriptor derivation (Chunk 2.2), verified against a
synthetic Gaussian peak with analytically known lambda_max/FWHM/integral --
not just "some threshold trips", but the actual textbook formulas:
FWHM = 2*sqrt(2*ln(2))*sigma, integral = amplitude*sigma*sqrt(2*pi).
"""

from __future__ import annotations

import numpy as np
import pytest

from dssc_discovery.chemistry.uvvis import (
    PROCESSING_VERSION,
    UVVisProcessingError,
    compute_bandwidth,
    compute_integrated_absorption,
    compute_lambda_max,
    compute_uvvis_descriptors,
    parse_uvvis_csv,
)

AMPLITUDE, MU, SIGMA = 0.85, 520.0, 15.0
ANALYTICAL_FWHM = 2 * np.sqrt(2 * np.log(2)) * SIGMA
ANALYTICAL_INTEGRAL = AMPLITUDE * SIGMA * np.sqrt(2 * np.pi)


def _gaussian_spectrum(
    amplitude: float = AMPLITUDE,
    mu: float = MU,
    sigma: float = SIGMA,
    lo: float = 400.0,
    hi: float = 650.0,
    step: float = 0.5,
    baseline: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    wavelength = np.arange(lo, hi + step, step)
    absorbance = baseline + amplitude * np.exp(-((wavelength - mu) ** 2) / (2 * sigma**2))
    return wavelength, absorbance


def _gaussian_csv_bytes(**kwargs) -> bytes:
    wavelength, absorbance = _gaussian_spectrum(**kwargs)
    lines = ["wavelength_nm,absorbance"]
    lines += [f"{w},{a}" for w, a in zip(wavelength, absorbance)]
    return "\n".join(lines).encode("utf-8")


# --- analytical recovery ------------------------------------------------


def test_lambda_max_recovers_gaussian_mean():
    wavelength, absorbance = _gaussian_spectrum()
    result = compute_lambda_max(wavelength, absorbance)
    assert result == pytest.approx(MU, abs=0.5)  # within one grid step


def test_bandwidth_recovers_analytical_fwhm():
    wavelength, absorbance = _gaussian_spectrum()
    lambda_max = compute_lambda_max(wavelength, absorbance)
    result = compute_bandwidth(wavelength, absorbance, lambda_max)
    assert result == pytest.approx(ANALYTICAL_FWHM, abs=0.01)


def test_integrated_absorption_recovers_analytical_integral():
    wavelength, absorbance = _gaussian_spectrum()
    result = compute_integrated_absorption(wavelength, absorbance)
    assert result == pytest.approx(ANALYTICAL_INTEGRAL, rel=1e-3)


def test_all_three_descriptors_together():
    wavelength, absorbance = _gaussian_spectrum()
    descriptors = compute_uvvis_descriptors(wavelength, absorbance)
    assert descriptors.lambda_max_nm == pytest.approx(MU, abs=0.5)
    assert descriptors.bandwidth_nm == pytest.approx(ANALYTICAL_FWHM, abs=0.01)
    assert descriptors.integrated_absorption == pytest.approx(ANALYTICAL_INTEGRAL, rel=1e-3)


def test_descriptors_are_deterministic_given_the_same_spectrum():
    wavelength, absorbance = _gaussian_spectrum()
    first = compute_uvvis_descriptors(wavelength, absorbance)
    second = compute_uvvis_descriptors(wavelength, absorbance)
    assert first == second


@pytest.mark.parametrize("baseline", [0.0, 0.05, 0.1])
def test_bandwidth_and_integral_are_robust_to_baseline_offset(baseline):
    # baseline-correction (min-subtraction) should make results
    # independent of a constant offset added to the whole spectrum
    wavelength, absorbance = _gaussian_spectrum(baseline=baseline)
    lambda_max = compute_lambda_max(wavelength, absorbance)
    bandwidth = compute_bandwidth(wavelength, absorbance, lambda_max)
    integral = compute_integrated_absorption(wavelength, absorbance)
    assert bandwidth == pytest.approx(ANALYTICAL_FWHM, abs=0.01)
    assert integral == pytest.approx(ANALYTICAL_INTEGRAL, rel=1e-3)


@pytest.mark.parametrize("mu", [420.0, 480.0, 550.0, 610.0])
def test_lambda_max_recovers_mean_across_different_peak_positions(mu):
    wavelength, absorbance = _gaussian_spectrum(mu=mu)
    result = compute_lambda_max(wavelength, absorbance)
    assert result == pytest.approx(mu, abs=0.5)


# --- parsing --------------------------------------------------------------


def test_parse_uvvis_csv_recovers_gaussian_values():
    raw_bytes = _gaussian_csv_bytes()
    wavelength, absorbance = parse_uvvis_csv(raw_bytes)
    assert len(wavelength) == len(absorbance)
    assert wavelength[0] <= wavelength[-1]  # sorted


def test_parse_uvvis_csv_sorts_out_of_order_rows():
    csv_text = "wavelength_nm,absorbance\n500,0.5\n400,0.1\n450,0.3\n"
    wavelength, absorbance = parse_uvvis_csv(csv_text.encode("utf-8"))
    assert list(wavelength) == [400.0, 450.0, 500.0]
    assert list(absorbance) == [0.1, 0.3, 0.5]


def test_parse_uvvis_csv_rejects_too_few_points():
    csv_text = "wavelength_nm,absorbance\n400,0.1\n500,0.5\n"
    with pytest.raises(UVVisProcessingError, match="at least 3 points"):
        parse_uvvis_csv(csv_text.encode("utf-8"))


def test_parse_uvvis_csv_rejects_malformed_row():
    csv_text = "wavelength_nm,absorbance\n400,0.1\n500,not-a-number\n600,0.2\n"
    with pytest.raises(UVVisProcessingError, match="Malformed"):
        parse_uvvis_csv(csv_text.encode("utf-8"))


def test_parse_uvvis_csv_rejects_missing_column():
    csv_text = "wavelength_nm,wrong_column\n400,0.1\n500,0.5\n600,0.2\n"
    with pytest.raises(UVVisProcessingError, match="Malformed"):
        parse_uvvis_csv(csv_text.encode("utf-8"))


def test_processing_version_is_a_nonempty_string():
    assert isinstance(PROCESSING_VERSION, str)
    assert len(PROCESSING_VERSION) > 0
