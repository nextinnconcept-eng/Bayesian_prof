"""Tests for FTIR descriptor derivation (Chunk 2.3), verified against a
synthetic multi-peak spectrum (3 non-overlapping Gaussians at known
positions/intensities) -- recovering exact ground-truth values, not just
"some peaks were found".
"""

from __future__ import annotations

import numpy as np
import pytest

from dssc_discovery.chemistry.ftir import (
    PROCESSING_VERSION,
    FTIRProcessingError,
    find_ftir_peaks,
    parse_ftir_csv,
)

# (position_cm1, intensity, sigma) -- widely spaced, non-overlapping
GROUND_TRUTH_PEAKS = [
    (1050.0, 0.30, 15.0),
    (1715.0, 0.60, 12.0),
    (3300.0, 0.45, 25.0),
]


def _synthetic_spectrum(
    peaks=GROUND_TRUTH_PEAKS, lo: float = 800.0, hi: float = 3600.0, step: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    wavenumber = np.arange(lo, hi + step, step)
    absorbance = np.zeros_like(wavenumber)
    for position, amplitude, sigma in peaks:
        absorbance += amplitude * np.exp(-((wavenumber - position) ** 2) / (2 * sigma**2))
    return wavenumber, absorbance


# --- peak detection against ground truth ------------------------------------


def test_finds_exactly_the_ground_truth_peak_count():
    wavenumber, absorbance = _synthetic_spectrum()
    peaks = find_ftir_peaks(wavenumber, absorbance, min_prominence=0.05)
    assert len(peaks) == len(GROUND_TRUTH_PEAKS)


def test_peak_positions_and_intensities_match_ground_truth():
    wavenumber, absorbance = _synthetic_spectrum()
    peaks = find_ftir_peaks(wavenumber, absorbance, min_prominence=0.05)

    for peak, (true_position, true_intensity, _sigma) in zip(peaks, GROUND_TRUTH_PEAKS):
        assert peak.position_cm1 == pytest.approx(true_position, abs=1.0)
        assert peak.intensity == pytest.approx(true_intensity, abs=1e-6)


def test_peaks_are_sorted_by_wavenumber_ascending():
    # ground truth is deliberately not given in position order in this
    # variant, to prove sorting isn't accidental input-order passthrough
    shuffled = [GROUND_TRUTH_PEAKS[2], GROUND_TRUTH_PEAKS[0], GROUND_TRUTH_PEAKS[1]]
    wavenumber, absorbance = _synthetic_spectrum(peaks=shuffled)
    peaks = find_ftir_peaks(wavenumber, absorbance, min_prominence=0.05)
    positions = [p.position_cm1 for p in peaks]
    assert positions == sorted(positions)


def test_no_peaks_found_is_not_an_error():
    wavenumber = np.arange(800.0, 1000.0, 1.0)
    absorbance = np.zeros_like(wavenumber)  # flat, featureless
    peaks = find_ftir_peaks(wavenumber, absorbance, min_prominence=0.05)
    assert peaks == []


# --- configurable prominence threshold ---------------------------------------


def test_prominence_threshold_filters_noise_peaks():
    wavenumber, absorbance = _synthetic_spectrum()
    rng = np.random.default_rng(42)
    noisy_absorbance = absorbance + rng.normal(0, 0.005, size=absorbance.shape)

    low_threshold_peaks = find_ftir_peaks(wavenumber, noisy_absorbance, min_prominence=0.001)
    high_threshold_peaks = find_ftir_peaks(wavenumber, noisy_absorbance, min_prominence=0.1)

    assert len(low_threshold_peaks) > len(GROUND_TRUTH_PEAKS)  # noise inflates the count
    assert len(high_threshold_peaks) == len(GROUND_TRUTH_PEAKS)  # threshold filters it out


def test_higher_prominence_threshold_never_finds_more_peaks():
    wavenumber, absorbance = _synthetic_spectrum()
    loose = find_ftir_peaks(wavenumber, absorbance, min_prominence=0.01)
    strict = find_ftir_peaks(wavenumber, absorbance, min_prominence=0.5)
    assert len(strict) <= len(loose)


# --- determinism -------------------------------------------------------------


def test_deterministic_given_the_same_spectrum():
    wavenumber, absorbance = _synthetic_spectrum()
    first = find_ftir_peaks(wavenumber, absorbance)
    second = find_ftir_peaks(wavenumber, absorbance)
    assert first == second


# --- parsing ------------------------------------------------------------------


def test_parse_ftir_csv_sorts_out_of_order_rows():
    csv_text = "wavenumber_cm1,absorbance\n1500,0.5\n800,0.1\n1000,0.3\n"
    wavenumber, absorbance = parse_ftir_csv(csv_text.encode("utf-8"))
    assert list(wavenumber) == [800.0, 1000.0, 1500.0]
    assert list(absorbance) == [0.1, 0.3, 0.5]


def test_parse_ftir_csv_rejects_too_few_points():
    csv_text = "wavenumber_cm1,absorbance\n800,0.1\n1500,0.5\n"
    with pytest.raises(FTIRProcessingError, match="at least 3 points"):
        parse_ftir_csv(csv_text.encode("utf-8"))


def test_parse_ftir_csv_rejects_malformed_row():
    csv_text = "wavenumber_cm1,absorbance\n800,0.1\n1000,oops\n1500,0.5\n"
    with pytest.raises(FTIRProcessingError, match="Malformed"):
        parse_ftir_csv(csv_text.encode("utf-8"))


def test_processing_version_is_a_nonempty_string():
    assert isinstance(PROCESSING_VERSION, str)
    assert len(PROCESSING_VERSION) > 0
