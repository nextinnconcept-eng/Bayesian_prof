"""UV-Vis descriptor derivation: raw absorbance spectrum -> lambda_max,
bandwidth (FWHM), integrated absorption.

Reference implementation for "derived descriptor traceable to source +
processing version" (PLAN.txt Section 12 / Chunk 2.2) -- later modalities
(FTIR, EIS, J-V) follow the same shape: pure numerical functions on
arrays (independently testable against analytically known values), a
file parser, and an orchestration function producing DerivedDescriptor
objects that reference their source RawSpectrum.

PROCESSING_VERSION travels with every descriptor this module produces.
Bump it whenever the algorithm changes materially -- that is the whole
point: a descriptor's provenance must say not just "computed from
spectrum X" but "computed from spectrum X by this specific method."
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from uuid import UUID

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import find_peaks

from dssc_discovery.chemistry.raw_storage import resolve_raw_file
from dssc_discovery.domain.measurements import DerivedDescriptor
from dssc_discovery.logging_setup import get_logger

logger = get_logger(__name__)

PROCESSING_VERSION = "uvvis-v1"


class UVVisProcessingError(Exception):
    """Raised when a UV-Vis spectrum can't be processed -- malformed
    data, too few points, or no locatable peak -- rather than silently
    returning a nonsensical descriptor.
    """


@dataclass(frozen=True)
class UVVisDescriptors:
    lambda_max_nm: float
    bandwidth_nm: float
    integrated_absorption: float


def parse_uvvis_csv(raw_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Parse a two-column (wavelength_nm, absorbance) CSV with a header
    row into wavelength-sorted numpy arrays.
    """
    text = raw_bytes.decode("utf-8")
    wavelengths: list[float] = []
    absorbances: list[float] = []
    for row in csv.DictReader(text.splitlines()):
        try:
            wavelengths.append(float(row["wavelength_nm"]))
            absorbances.append(float(row["absorbance"]))
        except (KeyError, ValueError) as exc:
            raise UVVisProcessingError(f"Malformed UV-Vis CSV row: {row}") from exc

    if len(wavelengths) < 3:
        raise UVVisProcessingError("UV-Vis spectrum needs at least 3 points to process")

    wavelength = np.asarray(wavelengths, dtype=float)
    absorbance = np.asarray(absorbances, dtype=float)
    order = np.argsort(wavelength)
    return wavelength[order], absorbance[order]


def compute_lambda_max(wavelength: np.ndarray, absorbance: np.ndarray) -> float:
    """Wavelength of the most prominent peak (scipy.signal.find_peaks).
    Falls back to the global maximum if find_peaks locates nothing --
    its definition of a peak excludes points at the array's edges, which
    a global maximum can legitimately sit on.
    """
    peaks, properties = find_peaks(absorbance, prominence=0.0)
    if len(peaks) == 0:
        peak_index = int(np.argmax(absorbance))
    else:
        peak_index = int(peaks[np.argmax(properties["prominences"])])
    return float(wavelength[peak_index])


def _interpolate_half_max_crossing(
    wavelength: np.ndarray,
    absorbance: np.ndarray,
    target: float,
    start_index: int,
    direction: int,
) -> float | None:
    """Walk from start_index in `direction` (+1 or -1) until absorbance
    crosses `target`, then linearly interpolate the exact crossing
    wavelength between the two bracketing sample points.
    """
    i = start_index
    n = len(wavelength)
    while 0 <= i + direction < n:
        next_i = i + direction
        crosses = (absorbance[i] - target) * (absorbance[next_i] - target) <= 0
        if crosses and absorbance[i] != absorbance[next_i]:
            x0, x1 = wavelength[i], wavelength[next_i]
            y0, y1 = absorbance[i], absorbance[next_i]
            fraction = (target - y0) / (y1 - y0)
            return float(x0 + fraction * (x1 - x0))
        i = next_i
    return None


def compute_bandwidth(
    wavelength: np.ndarray, absorbance: np.ndarray, lambda_max_nm: float
) -> float:
    """Full width at half maximum (FWHM), baseline-corrected (baseline
    estimated as the spectrum's minimum absorbance -- reasonable for a
    single dominant peak that doesn't spill outside the sampled range),
    with linear interpolation across each half-max crossing for
    sub-grid-point precision.
    """
    baseline = float(np.min(absorbance))
    peak_index = int(np.argmin(np.abs(wavelength - lambda_max_nm)))
    peak_value = float(absorbance[peak_index])
    half_max = baseline + (peak_value - baseline) / 2.0

    left = _interpolate_half_max_crossing(wavelength, absorbance, half_max, peak_index, -1)
    right = _interpolate_half_max_crossing(wavelength, absorbance, half_max, peak_index, 1)

    if left is None or right is None:
        raise UVVisProcessingError(
            "Could not locate both half-maximum crossings -- the spectrum may "
            "not fully capture the peak (e.g. a truncated wavelength range)"
        )
    return right - left


def compute_integrated_absorption(wavelength: np.ndarray, absorbance: np.ndarray) -> float:
    """Baseline-corrected area under the curve (trapezoidal rule) --
    isolates the peak's contribution rather than counting a systematic
    baseline offset across the whole range as signal.
    """
    baseline = float(np.min(absorbance))
    return float(trapezoid(absorbance - baseline, wavelength))


def compute_uvvis_descriptors(wavelength: np.ndarray, absorbance: np.ndarray) -> UVVisDescriptors:
    lambda_max = compute_lambda_max(wavelength, absorbance)
    bandwidth = compute_bandwidth(wavelength, absorbance, lambda_max)
    integrated = compute_integrated_absorption(wavelength, absorbance)
    return UVVisDescriptors(
        lambda_max_nm=lambda_max, bandwidth_nm=bandwidth, integrated_absorption=integrated
    )


def derive_uvvis_descriptors(
    source_spectrum_id: UUID, raw_data_reference: str, operator: str
) -> list[DerivedDescriptor]:
    """Full pipeline: resolve the raw file (Chunk 2.1, checksum-verified),
    parse it, compute descriptors, and package them as DerivedDescriptor
    domain objects referencing their source RawSpectrum -- ready to
    persist via DerivedDescriptorRepository.
    """
    raw_bytes = resolve_raw_file(raw_data_reference)
    wavelength, absorbance = parse_uvvis_csv(raw_bytes)
    descriptors = compute_uvvis_descriptors(wavelength, absorbance)

    logger.info(
        "UV-Vis descriptors derived: lambda_max=%.2fnm bandwidth=%.2fnm integrated=%.4f",
        descriptors.lambda_max_nm,
        descriptors.bandwidth_nm,
        descriptors.integrated_absorption,
    )
    return [
        DerivedDescriptor(
            source_spectrum_id=source_spectrum_id,
            processing_version=PROCESSING_VERSION,
            operator=operator,
            descriptor_name="lambda_max_nm",
            value=descriptors.lambda_max_nm,
            unit="nm",
        ),
        DerivedDescriptor(
            source_spectrum_id=source_spectrum_id,
            processing_version=PROCESSING_VERSION,
            operator=operator,
            descriptor_name="bandwidth_nm",
            value=descriptors.bandwidth_nm,
            unit="nm",
        ),
        DerivedDescriptor(
            source_spectrum_id=source_spectrum_id,
            processing_version=PROCESSING_VERSION,
            operator=operator,
            descriptor_name="integrated_absorption",
            value=descriptors.integrated_absorption,
            unit="AU*nm",
        ),
    ]
