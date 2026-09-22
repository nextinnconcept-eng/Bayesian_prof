"""Cross-entity business-rule validation: checks that need more than one
field, or more than one record, to verify -- duplicate detection, PCE/Voc/
Jsc/FF consistency, stability retention consistency, and missing-value-
reason enforcement (deferred here from Chunk 1.3's measurements.py, which
deliberately only validates what a single record can check about itself).

Every function here raises; none silently correct a value (PLAN.txt
Section 4/11: flag mismatches, don't auto-fix them). Ingestion (Chunk 1.7)
must call the relevant checks before persisting -- this module provides
the checks, not automatic enforcement at another layer.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from dssc_discovery.database.repositories import ExperimentalConditionRepository
from dssc_discovery.domain.enums import IlluminationProtocol
from dssc_discovery.domain.experimental_condition import (
    ExperimentalCondition,
    ExperimentalConditionVector,
)
from dssc_discovery.domain.measurements import PhotovoltaicMeasurement, StabilityMeasurement
from dssc_discovery.validation.exceptions import (
    DuplicateConditionError,
    MissingProvenanceError,
    PhysicallyImplausibleError,
)

# Incident power density per illumination protocol, mW/cm^2. Only
# protocols with a well-defined standard power density are listed --
# DARK/OTHER have no defined incident power, so check_pce_consistency
# skips silently for those (nothing to check against).
ILLUMINATION_POWER_MW_CM2: dict[IlluminationProtocol, float] = {
    IlluminationProtocol.AM1_5G_1_SUN: 100.0,
    IlluminationProtocol.AM1_5G_HALF_SUN: 50.0,
}

_VECTOR_FIELD_REL_TOL = 1e-6


def vectors_match(a: ExperimentalConditionVector, b: ExperimentalConditionVector) -> bool:
    """True if two parameter vectors match within floating-point
    tolerance -- exported (not just used internally) because ingestion
    (Chunk 1.7) needs the identical check for its own get-or-create
    dedup logic (reusing an existing condition_code vs. rejecting a
    conflicting reuse of one).
    """
    a_data, b_data = a.model_dump(), b.model_dump()
    for field, a_value in a_data.items():
        b_value = b_data[field]
        if isinstance(a_value, float):
            if not math.isclose(a_value, b_value, rel_tol=_VECTOR_FIELD_REL_TOL):
                return False
        elif a_value != b_value:
            return False
    return True


def check_duplicate_condition(
    repo: ExperimentalConditionRepository, condition: ExperimentalCondition
) -> None:
    """Raise DuplicateConditionError if an existing condition for the same
    species has a parameter vector matching within floating-point
    tolerance -- catches the same recipe re-entered under a different
    condition_code, which a UNIQUE constraint on condition_code alone
    can't.
    """
    for existing in repo.list():
        if existing.id == condition.id:
            continue
        if existing.species_code != condition.species_code:
            continue
        if vectors_match(existing.target_parameters, condition.target_parameters):
            raise DuplicateConditionError(
                f"ExperimentalCondition {condition.condition_code!r} matches "
                f"existing condition {existing.condition_code!r} "
                f"({condition.species_code}) within tolerance"
            )


def check_pce_consistency(pv: PhotovoltaicMeasurement, relative_tolerance: float = 0.10) -> None:
    """Raise PhysicallyImplausibleError if recorded PCE deviates from
    Voc * Jsc * FF / Pin by more than `relative_tolerance` (default 10%,
    generous enough for measurement/rounding noise while still catching
    gross errors like a misplaced decimal or wrong units). Skipped
    silently for illumination protocols with no defined incident power.
    """
    power_density = ILLUMINATION_POWER_MW_CM2.get(pv.illumination_protocol)
    if power_density is None:
        return

    computed_pce = (
        (pv.open_circuit_voltage_v * pv.short_circuit_current_density_ma_cm2 * pv.fill_factor)
        / power_density
        * 100.0
    )

    if not math.isclose(
        computed_pce, pv.power_conversion_efficiency_percent, rel_tol=relative_tolerance
    ):
        raise PhysicallyImplausibleError(
            f"Recorded PCE {pv.power_conversion_efficiency_percent:.2f}% does not "
            f"match Voc*Jsc*FF/Pin ({computed_pce:.2f}%) within "
            f"{relative_tolerance:.0%} for device {pv.device_id}"
        )


def check_stability_retention_consistency(
    stability: StabilityMeasurement,
    baseline_pv: PhotovoltaicMeasurement,
    timepoint_pv: PhotovoltaicMeasurement,
    relative_tolerance: float = 0.05,
) -> None:
    """Raise PhysicallyImplausibleError if retention_percent deviates from
    the baseline/timepoint PCE ratio by more than `relative_tolerance`
    (default 5%).
    """
    if baseline_pv.power_conversion_efficiency_percent == 0:
        return  # ratio undefined; nothing to check against

    computed_retention = (
        timepoint_pv.power_conversion_efficiency_percent
        / baseline_pv.power_conversion_efficiency_percent
        * 100.0
    )
    if not math.isclose(
        computed_retention, stability.retention_percent, rel_tol=relative_tolerance
    ):
        raise PhysicallyImplausibleError(
            f"Recorded retention {stability.retention_percent:.1f}% does not match "
            f"baseline/timepoint PCE ratio ({computed_retention:.1f}%) within "
            f"{relative_tolerance:.0%} for device {stability.device_id}"
        )


def check_missing_value_reason(obj: Any, numeric_field_names: Iterable[str]) -> None:
    """Raise MissingProvenanceError if any of the given optional numeric
    fields on `obj` is None but obj.missing_value_reason isn't set to
    explain why.
    """
    any_missing = any(getattr(obj, name) is None for name in numeric_field_names)
    if any_missing and obj.missing_value_reason is None:
        raise MissingProvenanceError(
            f"{type(obj).__name__} has a missing numeric field but no missing_value_reason set"
        )
