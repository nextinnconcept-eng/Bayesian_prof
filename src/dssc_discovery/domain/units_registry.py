"""Canonical units per domain field, with Pint-based conversion at the
ingestion boundary only.

Everywhere else in the system (database, features, ML) stores and consumes
plain floats already in the canonical unit below -- Pint Quantity objects
never travel through pandas/sklearn pipelines. Call to_canonical() once,
at the point a value enters the system (CSV/manual ingestion), and store
the float it returns.

pH is deliberately NOT included here: it's a defined logarithmic activity
scale, not a physical unit Pint understands. It's validated by numeric
range (0-14) in the validation layer instead.
"""

from __future__ import annotations

import pint

ureg: pint.UnitRegistry = pint.UnitRegistry()

# field name -> canonical unit (Pint-parseable string). Field names match
# the domain model attributes that will consume them (see PLAN.txt Section
# G/E) so this dict is the single source of truth both the domain models
# and the ingestion layer read from.
CANONICAL_UNITS: dict[str, str] = {
    # Extraction
    "extraction_temperature_c": "degC",
    "extraction_time_min": "minute",
    "pigment_concentration_mm": "millimolar",
    "extraction_yield_percent": "percent",
    # Sensitization
    "adsorption_time_min": "minute",
    "sensitization_concentration_mm": "millimolar",
    "sensitization_temperature_c": "degC",
    # Photoanode
    "tio2_thickness_um": "micrometer",
    "sintering_temperature_c": "degC",
    "sintering_time_min": "minute",
    "active_area_cm2": "centimeter ** 2",
    # Optical
    "lambda_max_nm": "nanometer",
    "absorption_bandwidth_nm": "nanometer",
    # Electrochemical
    "series_resistance_ohm": "ohm",
    "charge_transfer_resistance_ohm": "ohm",
    "recombination_resistance_ohm": "ohm",
    "electron_lifetime_ms": "millisecond",
    # Photovoltaic
    "open_circuit_voltage_v": "volt",
    "short_circuit_current_density_ma_cm2": "milliampere / centimeter ** 2",
    "fill_factor": "dimensionless",
    "power_conversion_efficiency_percent": "percent",
    # Stability
    "ageing_duration_h": "hour",
    "retention_percent": "percent",
}


class UnitConversionError(ValueError):
    """Raised for an unregistered field, an unrecognized unit, or an
    incompatible unit conversion -- wraps Pint's own exception types so
    callers only need to catch one thing at this boundary."""


def to_canonical(value: float, unit: str, field: str) -> float:
    """Convert `value` (given in `unit`) to the canonical unit registered
    for `field`, returning a plain float. Raises UnitConversionError if
    `field` isn't registered, `unit` isn't a recognized unit, or `unit`
    isn't dimensionally compatible with the field's canonical unit."""
    try:
        canonical_unit = CANONICAL_UNITS[field]
    except KeyError as exc:
        raise UnitConversionError(
            f"Unknown field {field!r}: no canonical unit registered in CANONICAL_UNITS."
        ) from exc

    try:
        quantity = ureg.Quantity(value, unit)
    except pint.UndefinedUnitError as exc:
        raise UnitConversionError(f"Unrecognized unit {unit!r} for field {field!r}: {exc}") from exc

    try:
        converted = quantity.to(canonical_unit)
    except pint.DimensionalityError as exc:
        raise UnitConversionError(
            f"Unit {unit!r} is not compatible with canonical unit "
            f"{canonical_unit!r} for field {field!r}: {exc}"
        ) from exc

    return float(converted.magnitude)
