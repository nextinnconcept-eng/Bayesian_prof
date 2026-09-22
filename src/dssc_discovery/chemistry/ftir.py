"""FTIR descriptor derivation: raw absorbance spectrum -> peak positions
and intensities (functional-group bands), following the same
raw-file -> DerivedDescriptor pattern uvvis.py established (Chunk 2.2).

Unlike UV-Vis (one dominant peak matters), FTIR spectra are read by
their MULTIPLE characteristic peaks -- each one a candidate functional
group. Each detected peak becomes two DerivedDescriptor rows (position,
intensity), not one, since DerivedDescriptor holds a single scalar value
and a peak needs two numbers to be meaningful. Rows over a JSON blob:
individually queryable/traceable, matching Chunk 2's stated preference
for "structured rows for traceability" where the ML feature layer will
eventually need to select "peak near 1700 cm^-1" style features.

Input is assumed to be in absorbance units (peaks point up), the same
convention uvvis.py uses -- a %Transmittance spectrum would need
converting to absorbance first, which is out of this module's scope.

min_prominence is exposed as a parameter (not hardcoded) precisely
because FTIR spectra have many small local peaks that are noise, not
functional-group bands -- verified empirically that a realistic noise
floor produces dozens of spurious peaks at low prominence thresholds
but cleanly recovers only the true peaks once prominence exceeds the
noise amplitude.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from uuid import UUID

import numpy as np
from scipy.signal import find_peaks

from dssc_discovery.chemistry.raw_storage import resolve_raw_file
from dssc_discovery.domain.measurements import DerivedDescriptor
from dssc_discovery.logging_setup import get_logger

logger = get_logger(__name__)

PROCESSING_VERSION = "ftir-v1"
DEFAULT_MIN_PROMINENCE = 0.05


class FTIRProcessingError(Exception):
    """Raised when an FTIR spectrum can't be processed -- malformed
    data or too few points -- rather than silently returning a
    nonsensical result. Finding zero peaks is NOT an error (a flat or
    featureless spectrum is a legitimate, if uninteresting, result).
    """


@dataclass(frozen=True)
class FTIRPeak:
    position_cm1: float
    intensity: float
    prominence: float


def parse_ftir_csv(raw_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Parse a two-column (wavenumber_cm1, absorbance) CSV with a header
    row into wavenumber-sorted numpy arrays.
    """
    text = raw_bytes.decode("utf-8")
    wavenumbers: list[float] = []
    absorbances: list[float] = []
    for row in csv.DictReader(text.splitlines()):
        try:
            wavenumbers.append(float(row["wavenumber_cm1"]))
            absorbances.append(float(row["absorbance"]))
        except (KeyError, ValueError) as exc:
            raise FTIRProcessingError(f"Malformed FTIR CSV row: {row}") from exc

    if len(wavenumbers) < 3:
        raise FTIRProcessingError("FTIR spectrum needs at least 3 points to process")

    wavenumber = np.asarray(wavenumbers, dtype=float)
    absorbance = np.asarray(absorbances, dtype=float)
    order = np.argsort(wavenumber)
    return wavenumber[order], absorbance[order]


def find_ftir_peaks(
    wavenumber: np.ndarray, absorbance: np.ndarray, min_prominence: float = DEFAULT_MIN_PROMINENCE
) -> list[FTIRPeak]:
    """Detect peaks with prominence >= min_prominence, returned sorted by
    wavenumber ascending (a stable, reproducible order that doesn't
    shift with small intensity changes across replicate scans, unlike
    sorting by intensity/prominence rank would).
    """
    indices, properties = find_peaks(absorbance, prominence=min_prominence)
    peaks = [
        FTIRPeak(
            position_cm1=float(wavenumber[idx]),
            intensity=float(absorbance[idx]),
            prominence=float(prom),
        )
        for idx, prom in zip(indices, properties["prominences"])
    ]
    return sorted(peaks, key=lambda p: p.position_cm1)


def derive_ftir_descriptors(
    source_spectrum_id: UUID,
    raw_data_reference: str,
    operator: str,
    min_prominence: float = DEFAULT_MIN_PROMINENCE,
) -> list[DerivedDescriptor]:
    """Full pipeline: resolve the raw file (Chunk 2.1, checksum-verified),
    parse it, detect peaks, and package position + intensity for each
    peak as DerivedDescriptor rows referencing their source RawSpectrum.
    Peaks are numbered by wavenumber-ascending order (ftir_peak_1 = the
    lowest-wavenumber detected peak, not necessarily the most intense).
    """
    raw_bytes = resolve_raw_file(raw_data_reference)
    wavenumber, absorbance = parse_ftir_csv(raw_bytes)
    peaks = find_ftir_peaks(wavenumber, absorbance, min_prominence=min_prominence)

    logger.info("FTIR peaks derived: %d peak(s) at prominence>=%.3f", len(peaks), min_prominence)

    descriptors: list[DerivedDescriptor] = []
    for i, peak in enumerate(peaks, start=1):
        descriptors.append(
            DerivedDescriptor(
                source_spectrum_id=source_spectrum_id,
                processing_version=PROCESSING_VERSION,
                operator=operator,
                descriptor_name=f"ftir_peak_{i}_position",
                value=peak.position_cm1,
                unit="cm^-1",
            )
        )
        descriptors.append(
            DerivedDescriptor(
                source_spectrum_id=source_spectrum_id,
                processing_version=PROCESSING_VERSION,
                operator=operator,
                descriptor_name=f"ftir_peak_{i}_intensity",
                value=peak.intensity,
                unit="AU",
            )
        )
    return descriptors
