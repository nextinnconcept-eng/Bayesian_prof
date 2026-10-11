"""ML-ready dataset assembly: one row per Device that has at least one
PhotovoltaicMeasurement, joining condition -> device -> measurements ->
descriptors into a flat pandas DataFrame.

Grain and target selection, explicit rather than implicit:
- One row per Device. A device with no PhotovoltaicMeasurement yet isn't
  part of the ML-ready dataset at all (it hasn't reached MEASURED in the
  closed-loop state machine, PLAN.txt Section K) -- there's no target to
  train on, so it's excluded, not included with a NaN target.
- Where a device has multiple PhotovoltaicMeasurements (repeat scans,
  or later stability-tracking measurements), the one with the earliest
  (measurement_date, measurement_replicate_number) is used as this row's
  target -- the device's baseline, as-fabricated performance. Stability
  measurements at later timepoints are a different scientific question
  (Papers 6 / Chunk 9.x) and stay out of this table.
- Where a device has multiple ElectrochemicalMeasurements, the most
  recent (by created_at) is used.
- A device's pigment batch (needed for extraction_yield_percent and the
  UV-Vis descriptors) is found via its SensitizationRun, if one exists;
  a device with no SensitizationRun on record simply has those
  characterization features as NaN, never silently zero (PLAN.txt
  Section 11).

group_id = experimental_condition_id: the key every leakage-safe CV
strategy (Chunk 4.4) groups on, since replicate devices of the same
condition must never straddle a train/test split.

Known simplification, not silently assumed: ExperimentalCondition.
target_parameters is used as the source for the 13 design-time vector
features, not a separately-reconciled "as-executed" record from
PhotoanodeFabricationBatch. For data ingested via Chunk 1.7 these are
guaranteed identical (the ingestion pipeline populates both from the
same executed recipe). Once Chunk 8.x's real BO-driven fabrication can
produce genuine target-vs-actual divergence, this will need revisiting
-- flagged here rather than silently assumed away.

No raw sqlalchemy query/ORM-class use here (Section D): only Session as
an opaque handle (the same pattern ingestion/manual_entry.py already
established) passed to the repository layer, which does all the actual
querying.
"""

from __future__ import annotations

from uuid import UUID

import pandas as pd
from sqlalchemy.orm import Session

from dssc_discovery.database.repositories import (
    DerivedDescriptorRepository,
    DeviceRepository,
    ElectrochemicalMeasurementRepository,
    ExperimentalConditionRepository,
    PhotovoltaicMeasurementRepository,
    PigmentExtractBatchRepository,
    RawSpectrumRepository,
    SensitizationRunRepository,
)
from dssc_discovery.domain.enums import MeasurementType
from dssc_discovery.domain.experimental_condition import ExperimentalCondition
from dssc_discovery.domain.measurements import PhotovoltaicMeasurement
from dssc_discovery.features.registry import feature_names

_UVVIS_DESCRIPTOR_NAMES = ("lambda_max_nm", "bandwidth_nm", "integrated_absorption")
_EIS_FIELD_NAMES = (
    "series_resistance_ohm",
    "charge_transfer_resistance_ohm",
    "recombination_resistance_ohm",
    "electron_lifetime_ms",
)

# Explicit, in-order column list -- used to give an empty result (no
# devices measured yet) the correct schema rather than zero columns, so
# `df["power_conversion_efficiency_percent"]` never KeyErrors just
# because the dataset happens to be empty right now.
DATASET_COLUMNS: tuple[str, ...] = (
    "device_id",
    "group_id",
    "replicate_id",
    "condition_code",
    "design_source",  # provenance, not a feature -- strategy B's split key (Chunk 4.5)
    "device_code",
    *feature_names(stage="design"),  # species_code is first among these (Chunk 3.1's ordering)
    *feature_names(stage="characterization"),
    "open_circuit_voltage_v",
    "short_circuit_current_density_ma_cm2",
    "fill_factor",
    "power_conversion_efficiency_percent",
    "is_synthetic",
)


def _vector_row(condition: ExperimentalCondition) -> dict[str, object]:
    vector = condition.target_parameters
    return {
        "solvent": vector.solvent.value,
        "solvent_ratio": vector.solvent_ratio,
        "ph": vector.ph,
        "extraction_temperature_c": vector.extraction_temperature_c,
        "extraction_time_min": vector.extraction_time_min,
        "pigment_concentration_mm": vector.pigment_concentration_mm,
        "adsorption_time_min": vector.adsorption_time_min,
        "sensitization_concentration_mm": vector.sensitization_concentration_mm,
        "sensitization_temperature_c": vector.sensitization_temperature_c,
        "tio2_thickness_um": vector.tio2_thickness_um,
        "sintering_temperature_c": vector.sintering_temperature_c,
        "sintering_time_min": vector.sintering_time_min,
        "surface_treatment": vector.surface_treatment.value,
    }


def _select_primary_pv_measurement(
    measurements: list[PhotovoltaicMeasurement],
) -> PhotovoltaicMeasurement:
    return min(measurements, key=lambda m: (m.measurement_date, m.measurement_replicate_number))


def _pigment_batch_id_for_device(
    device_id: UUID, sensitization_repo: SensitizationRunRepository
) -> UUID | None:
    runs = sensitization_repo.filter(device_id=device_id)
    if not runs:
        return None
    return max(runs, key=lambda r: r.created_at).pigment_extract_batch_id


def _characterization_row(
    device_id: UUID,
    pigment_batch_id: UUID | None,
    pigment_batch_repo: PigmentExtractBatchRepository,
    raw_spectrum_repo: RawSpectrumRepository,
    descriptor_repo: DerivedDescriptorRepository,
    eis_repo: ElectrochemicalMeasurementRepository,
) -> dict[str, float]:
    row: dict[str, float] = {name: float("nan") for name in feature_names(stage="characterization")}

    if pigment_batch_id is not None:
        batch = pigment_batch_repo.get(pigment_batch_id)
        if batch is not None and batch.extraction_yield_percent is not None:
            row["extraction_yield_percent"] = batch.extraction_yield_percent

        uvvis_spectra = [
            spectrum
            for spectrum in raw_spectrum_repo.filter(pigment_extract_batch_id=pigment_batch_id)
            if spectrum.measurement_type == MeasurementType.UV_VIS
        ]
        if uvvis_spectra:
            latest_spectrum = max(uvvis_spectra, key=lambda s: s.created_at)
            descriptors_by_name = {
                d.descriptor_name: d.value
                for d in descriptor_repo.filter(source_spectrum_id=latest_spectrum.id)
            }
            for name in _UVVIS_DESCRIPTOR_NAMES:
                if name in descriptors_by_name:
                    row[name] = descriptors_by_name[name]

    eis_measurements = eis_repo.filter(device_id=device_id)
    if eis_measurements:
        latest_eis = max(eis_measurements, key=lambda m: m.created_at)
        for name in _EIS_FIELD_NAMES:
            value = getattr(latest_eis, name)
            if value is not None:
                row[name] = value

    return row


def build_ml_dataset(session: Session) -> pd.DataFrame:
    """Assemble the flat, ML-ready dataset from the current database
    state. Deterministic: identical DB state always produces an
    identical DataFrame (rows sorted by device_code).
    """
    condition_repo = ExperimentalConditionRepository(session)
    device_repo = DeviceRepository(session)
    sensitization_repo = SensitizationRunRepository(session)
    pigment_batch_repo = PigmentExtractBatchRepository(session)
    raw_spectrum_repo = RawSpectrumRepository(session)
    descriptor_repo = DerivedDescriptorRepository(session)
    eis_repo = ElectrochemicalMeasurementRepository(session)
    pv_repo = PhotovoltaicMeasurementRepository(session)

    conditions_by_id = {c.id: c for c in condition_repo.list()}
    rows: list[dict[str, object]] = []

    for device in device_repo.list():
        condition = conditions_by_id.get(device.experimental_condition_id)
        if condition is None:
            continue  # defensive: FK integrity guarantees this in practice

        pv_measurements = pv_repo.filter(device_id=device.id)
        if not pv_measurements:
            continue  # not yet measured -- not part of the ML-ready dataset
        primary_pv = _select_primary_pv_measurement(pv_measurements)

        pigment_batch_id = _pigment_batch_id_for_device(device.id, sensitization_repo)
        characterization = _characterization_row(
            device.id,
            pigment_batch_id,
            pigment_batch_repo,
            raw_spectrum_repo,
            descriptor_repo,
            eis_repo,
        )

        row: dict[str, object] = {
            "device_id": str(device.id),
            "group_id": str(condition.id),
            "replicate_id": device.replicate_number,
            "condition_code": condition.condition_code,
            "design_source": condition.design_source.value,
            "device_code": device.device_code,
            "species_code": condition.species_code.value,
            **_vector_row(condition),
            "active_area_cm2": device.active_area_cm2,
            "electrolyte_type": device.electrolyte_type.value,
            "counter_electrode_type": device.counter_electrode_type.value,
            **characterization,
            "open_circuit_voltage_v": primary_pv.open_circuit_voltage_v,
            "short_circuit_current_density_ma_cm2": primary_pv.short_circuit_current_density_ma_cm2,
            "fill_factor": primary_pv.fill_factor,
            "power_conversion_efficiency_percent": primary_pv.power_conversion_efficiency_percent,
            "is_synthetic": device.is_synthetic,
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=list(DATASET_COLUMNS))

    dataframe = pd.DataFrame(rows, columns=list(DATASET_COLUMNS))
    return dataframe.sort_values("device_code").reset_index(drop=True)
