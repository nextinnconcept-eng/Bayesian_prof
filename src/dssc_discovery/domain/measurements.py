"""Domain models for raw instrument data and everything derived from it
(PLAN.txt Section E: RawSpectrum, DerivedDescriptor,
ElectrochemicalMeasurement, PhotovoltaicMeasurement, StabilityMeasurement).

Raw data is never overwritten (PLAN.txt Section 12): a RawSpectrum row is
immutable once created, and DerivedDescriptor/ElectrochemicalMeasurement
always carry a source reference back to it, tagged with the code version
that produced them (processing_version / experiment_version).

Cross-entity consistency checks that need more than one field alone to
verify (e.g. PCE roughly matching Voc x Jsc x FF, or retention_percent
matching the baseline/timepoint PCE ratio) are deliberately NOT enforced
here -- that's the validation rules engine, Chunk 1.6. This module only
enforces what a single record can check about itself.
"""

from __future__ import annotations

from datetime import date
from uuid import UUID

from pydantic import Field, model_validator

from dssc_discovery.domain.base import MeasurementProvenancedModel, ProvenancedModel
from dssc_discovery.domain.enums import (
    EquivalentCircuitModel,
    IlluminationProtocol,
    MeasurementType,
)

_CHEMICAL_OPTICAL_TYPES = {
    MeasurementType.UV_VIS,
    MeasurementType.FTIR,
    MeasurementType.HPLC,
    MeasurementType.LC_MS,
}
_DEVICE_TYPES = {MeasurementType.EIS, MeasurementType.JV_CURVE}


class RawSpectrum(MeasurementProvenancedModel):
    """One instrument acquisition. Source is polymorphic: chemical/optical
    modalities (UV-Vis/FTIR/HPLC/LC-MS) come from a PigmentExtractBatch;
    electrical modalities (EIS/J-V) come from a Device. Exactly one of
    pigment_extract_batch_id / device_id must be set, consistent with
    measurement_type.
    """

    measurement_type: MeasurementType
    pigment_extract_batch_id: UUID | None = None
    device_id: UUID | None = None
    raw_data_reference: str = Field(min_length=1)  # required here, unlike the mixin default
    acquisition_date: date

    @model_validator(mode="after")
    def _exactly_one_source_matching_measurement_type(self) -> RawSpectrum:
        has_batch = self.pigment_extract_batch_id is not None
        has_device = self.device_id is not None
        if has_batch == has_device:  # both set, or neither
            raise ValueError(
                "RawSpectrum must reference exactly one of "
                "pigment_extract_batch_id or device_id, not both or neither"
            )
        if self.measurement_type in _CHEMICAL_OPTICAL_TYPES and not has_batch:
            raise ValueError(
                f"{self.measurement_type} spectra must reference pigment_extract_batch_id"
            )
        if self.measurement_type in _DEVICE_TYPES and not has_device:
            raise ValueError(f"{self.measurement_type} spectra must reference device_id")
        return self

    @model_validator(mode="after")
    def _acquisition_date_not_in_future(self) -> RawSpectrum:
        if self.acquisition_date > date.today():
            raise ValueError(f"acquisition_date {self.acquisition_date} is in the future")
        return self


class DerivedDescriptor(ProvenancedModel):
    """A single computed value traceable to the RawSpectrum it came from
    and the processing version that computed it. Failed derivations
    (e.g. peak-finding that finds nothing) produce no row -- absence means
    "not computed", logged separately, rather than a null-valued row.
    """

    source_spectrum_id: UUID
    descriptor_name: str = Field(min_length=1)
    value: float
    unit: str = ""
    processing_version: str = Field(min_length=1)
    algorithm: str = ""


class ElectrochemicalMeasurement(MeasurementProvenancedModel):
    """One EIS equivalent-circuit fit result. Result fields are Optional
    with missing_value_reason explaining absence -- a non-convergent fit
    (PLAN.txt Section 16) still produces a row documenting the attempt,
    never a silently-dropped measurement.
    """

    device_id: UUID
    raw_spectrum_id: UUID
    equivalent_circuit_model: EquivalentCircuitModel
    series_resistance_ohm: float | None = Field(default=None, ge=0)
    charge_transfer_resistance_ohm: float | None = Field(default=None, ge=0)
    recombination_resistance_ohm: float | None = Field(default=None, ge=0)
    electron_lifetime_ms: float | None = Field(default=None, ge=0)
    fit_r_squared: float | None = Field(default=None, ge=0, le=1)


class PhotovoltaicMeasurement(MeasurementProvenancedModel):
    device_id: UUID
    raw_curve_id: UUID | None = None  # a RawSpectrum with measurement_type=JV_CURVE, if stored
    illumination_protocol: IlluminationProtocol
    measurement_date: date
    measurement_replicate_number: int = Field(default=1, gt=0)
    open_circuit_voltage_v: float = Field(ge=0)
    short_circuit_current_density_ma_cm2: float = Field(ge=0)
    fill_factor: float = Field(ge=0, le=1)
    power_conversion_efficiency_percent: float = Field(ge=0, le=100)


class StabilityMeasurement(MeasurementProvenancedModel):
    device_id: UUID
    baseline_measurement_id: UUID
    timepoint_measurement_id: UUID
    ageing_duration_h: float = Field(ge=0)
    ageing_conditions: str = ""
    # Generous upper bound (not capped at 100): ideal retention can't
    # exceed 100%, but measurement noise can plausibly push a real
    # reading slightly above it -- this catches clear data-entry errors
    # (e.g. a stray zero) without rejecting plausible noisy results.
    retention_percent: float = Field(ge=0, le=150)
