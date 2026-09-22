"""J-V curve processing: raw current-voltage sweep -> Voc, Jsc, FF, PCE.

Voc/Jsc via linear interpolation at the curve's zero-crossings (V=0 for
Jsc, J=0 for Voc) -- the same sub-grid-point interpolation pattern
uvvis.py established for its half-max crossings (Chunk 2.2). FF/PCE come
from locating the maximum power point P(V)=V*J(V) on the curve within
[0, Voc], by grid search rather than continuous refinement -- consistent
with the grid-point-precision choice already made for UV-Vis/FTIR peak
location, and verified empirically sufficient (FF/PCE recovered to
~0.01% of the true continuous optimum at realistic 5mV sampling).

Sign convention: J is positive in the power-generating quadrant
(0 <= V <= Voc), the standard solar-cell J-V curve presentation.

PCE is computed via the SAME formula PLAN.txt Section 1.6's
check_pce_consistency() checks against, reusing its
ILLUMINATION_POWER_MW_CM2 table rather than a second hardcoded copy --
and this module runs that check automatically on its own output before
returning (Chunk 2.5's own requirement), so a future drift between the
two independently-maintained formula implementations can never silently
produce an internally-inconsistent PhotovoltaicMeasurement.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from uuid import UUID

import numpy as np

from dssc_discovery.chemistry.raw_storage import resolve_raw_file
from dssc_discovery.domain.enums import IlluminationProtocol
from dssc_discovery.domain.measurements import PhotovoltaicMeasurement
from dssc_discovery.logging_setup import get_logger
from dssc_discovery.validation.rules import ILLUMINATION_POWER_MW_CM2, check_pce_consistency

logger = get_logger(__name__)

PROCESSING_VERSION = "jv-v1"


class JVProcessingError(Exception):
    """Raised for malformed input data, too few points, a curve with no
    locatable Voc (never crosses zero within the swept range), or an
    illumination protocol with no defined incident power (PCE has no
    physical meaning without one, and PhotovoltaicMeasurement.
    power_conversion_efficiency_percent is a required field -- Chunk
    1.3 -- so there is no partial result to return in that case).
    """


@dataclass(frozen=True)
class JVDescriptors:
    open_circuit_voltage_v: float
    short_circuit_current_density_ma_cm2: float
    fill_factor: float
    power_conversion_efficiency_percent: float
    v_mp: float
    j_mp: float


def parse_jv_csv(raw_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Parse a two-column (voltage_v, current_density_ma_cm2) CSV with a
    header row into voltage-sorted numpy arrays.
    """
    text = raw_bytes.decode("utf-8")
    voltages: list[float] = []
    currents: list[float] = []
    for row in csv.DictReader(text.splitlines()):
        try:
            voltages.append(float(row["voltage_v"]))
            currents.append(float(row["current_density_ma_cm2"]))
        except (KeyError, ValueError) as exc:
            raise JVProcessingError(f"Malformed J-V CSV row: {row}") from exc

    if len(voltages) < 3:
        raise JVProcessingError("J-V curve needs at least 3 points to process")

    voltage = np.asarray(voltages, dtype=float)
    current = np.asarray(currents, dtype=float)
    order = np.argsort(voltage)
    return voltage[order], current[order]


def compute_short_circuit_current_density(voltage: np.ndarray, current: np.ndarray) -> float:
    """Jsc = J at V=0, via linear interpolation."""
    return float(np.interp(0.0, voltage, current))


def compute_open_circuit_voltage(voltage: np.ndarray, current: np.ndarray) -> float:
    """Voc = V where J crosses zero (positive to negative, since J
    declines monotonically through the operating region), via linear
    interpolation between the bracketing sample points.
    """
    signs = np.sign(current)
    crossings = np.where(np.diff(signs) != 0)[0]
    if len(crossings) == 0:
        raise JVProcessingError(
            "Curve never crosses J=0 within the swept voltage range -- cannot locate Voc"
        )
    i = int(crossings[0])
    v0, v1 = voltage[i], voltage[i + 1]
    j0, j1 = current[i], current[i + 1]
    fraction = (0.0 - j0) / (j1 - j0)
    return float(v0 + fraction * (v1 - v0))


def compute_fill_factor(
    voltage: np.ndarray, current: np.ndarray, voc: float, jsc: float
) -> tuple[float, float, float]:
    """Locate the maximum power point within [0, Voc] and return
    (fill_factor, v_mp, j_mp). FF = (Vmp*Jmp)/(Voc*Jsc).
    """
    mask = (voltage >= 0) & (voltage <= voc)
    if not np.any(mask):
        raise JVProcessingError("No sampled points fall within [0, Voc] to locate the MPP")

    power = voltage[mask] * current[mask]
    mp_index = int(np.argmax(power))
    v_mp = float(voltage[mask][mp_index])
    j_mp = float(current[mask][mp_index])

    if voc <= 0 or jsc <= 0:
        raise JVProcessingError(f"Voc={voc} and Jsc={jsc} must both be positive to compute FF")

    fill_factor = (v_mp * j_mp) / (voc * jsc)
    return fill_factor, v_mp, j_mp


def compute_jv_descriptors(
    voltage: np.ndarray, current: np.ndarray, illumination_protocol: IlluminationProtocol
) -> JVDescriptors:
    power_density = ILLUMINATION_POWER_MW_CM2.get(illumination_protocol)
    if power_density is None:
        raise JVProcessingError(
            f"No defined incident power density for {illumination_protocol} -- PCE cannot be "
            f"computed (see validation.rules.ILLUMINATION_POWER_MW_CM2)"
        )

    jsc = compute_short_circuit_current_density(voltage, current)
    voc = compute_open_circuit_voltage(voltage, current)
    fill_factor, v_mp, j_mp = compute_fill_factor(voltage, current, voc, jsc)
    pce = (voc * jsc * fill_factor) / power_density * 100.0

    return JVDescriptors(
        open_circuit_voltage_v=voc,
        short_circuit_current_density_ma_cm2=jsc,
        fill_factor=fill_factor,
        power_conversion_efficiency_percent=pce,
        v_mp=v_mp,
        j_mp=j_mp,
    )


def derive_jv_measurement(
    device_id: UUID,
    raw_data_reference: str,
    illumination_protocol: IlluminationProtocol,
    measurement_date: date,
    operator: str,
    instrument: str,
    raw_curve_id: UUID | None = None,
) -> PhotovoltaicMeasurement:
    """Full pipeline: resolve the raw file (Chunk 2.1), parse it,
    extract Voc/Jsc/FF/PCE, and package the result as a
    PhotovoltaicMeasurement -- running Chunk 1.6's check_pce_consistency
    on it automatically before returning.
    """
    raw_bytes = resolve_raw_file(raw_data_reference)
    voltage, current = parse_jv_csv(raw_bytes)
    descriptors = compute_jv_descriptors(voltage, current, illumination_protocol)

    logger.info(
        "J-V descriptors derived: Voc=%.4fV Jsc=%.4fmA/cm^2 FF=%.4f PCE=%.4f%%",
        descriptors.open_circuit_voltage_v,
        descriptors.short_circuit_current_density_ma_cm2,
        descriptors.fill_factor,
        descriptors.power_conversion_efficiency_percent,
    )

    measurement = PhotovoltaicMeasurement(
        device_id=device_id,
        raw_curve_id=raw_curve_id,
        illumination_protocol=illumination_protocol,
        measurement_date=measurement_date,
        open_circuit_voltage_v=descriptors.open_circuit_voltage_v,
        short_circuit_current_density_ma_cm2=descriptors.short_circuit_current_density_ma_cm2,
        fill_factor=descriptors.fill_factor,
        power_conversion_efficiency_percent=descriptors.power_conversion_efficiency_percent,
        operator=operator,
        instrument=instrument,
    )

    # Chunk 2.5's own requirement: run the Section 1.6 consistency check
    # automatically. Protects against this module's PCE formula silently
    # drifting from validation/rules.py's independently-maintained copy.
    check_pce_consistency(measurement)

    return measurement
