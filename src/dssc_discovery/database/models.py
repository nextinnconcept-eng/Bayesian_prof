"""SQLAlchemy 2.0 declarative schema mirroring the domain models from
Chunks 1.1-1.3 (PLAN.txt Section E).

This module is a pure persistence mapping: no business logic, no
validation beyond what the column types/FKs enforce structurally. Domain
objects (dssc_discovery.domain.*) are what the rest of the system works
with; the repository layer (Chunk 1.5) converts between the two.

Engine-agnostic by construction: enum columns use native_enum=False (plain
VARCHAR + CHECK constraint on every backend, not a Postgres-only native
ENUM type), and UUIDs use a portable GUID TypeDecorator (CHAR(36)) rather
than a Postgres-only UUID column type, since SQLite has neither.

Other entities reference Species by its natural key (species_code), not a
surrogate id -- matching how the domain layer already models it
(PlantSample.species_code, ExperimentalCondition.species_code are
SpeciesCode values, not UUIDs). So species.code, not species.id, is what
plant_samples.species_code and experimental_conditions.species_code
foreign-key against.
"""

from __future__ import annotations

import uuid as uuid_module
from datetime import UTC, date, datetime
from enum import StrEnum

import sqlalchemy as sa
from sqlalchemy import CheckConstraint, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import CHAR, TypeDecorator

from dssc_discovery.domain.enums import (
    ClosedLoopState,
    CounterElectrodeType,
    DesignSource,
    ElectrolyteType,
    EquivalentCircuitModel,
    IlluminationProtocol,
    MeasurementType,
    MissingValueReason,
    PigmentClass,
    PlantPart,
    SolventType,
    SpeciesCode,
    SurfaceTreatment,
)


class GUID(TypeDecorator):
    """Platform-independent UUID: stored as CHAR(36) on every backend,
    since SQLite has no native UUID type and Postgres's is backend-only.
    """

    impl = CHAR(36)
    cache_ok = True

    def process_bind_param(self, value: object, dialect: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, uuid_module.UUID):
            return str(value)
        return str(uuid_module.UUID(str(value)))

    def process_result_value(self, value: object, dialect: object) -> uuid_module.UUID | None:
        if value is None:
            return None
        return uuid_module.UUID(str(value))


def _str_enum(enum_cls: type[StrEnum], length: int = 50) -> sa.Enum:
    """A native_enum=False Enum column that stores member.value (e.g.
    "Telfairia occidentalis"), not member.name (e.g.
    "TELFAIRIA_OCCIDENTALIS") -- verified empirically that SQLAlchemy's
    default behavior stores .name, which would silently diverge from the
    domain layer's actual string values.
    """
    return sa.Enum(
        enum_cls,
        native_enum=False,
        length=length,
        values_callable=lambda cls: [member.value for member in cls],
    )


class Base(DeclarativeBase):
    pass


class ProvenanceMixin:
    """Mirrors domain.base.ProvenancedModel's fields exactly."""

    id: Mapped[uuid_module.UUID] = mapped_column(
        GUID(), primary_key=True, default=uuid_module.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))
    operator: Mapped[str] = mapped_column(sa.String(200))
    is_synthetic: Mapped[bool] = mapped_column(default=False)
    experiment_version: Mapped[str] = mapped_column(sa.String(50), default="v1")
    notes: Mapped[str] = mapped_column(sa.Text, default="")


class MeasurementProvenanceMixin(ProvenanceMixin):
    """Mirrors domain.base.MeasurementProvenancedModel's added fields."""

    instrument: Mapped[str] = mapped_column(sa.String(200))
    raw_data_reference: Mapped[str | None] = mapped_column(sa.String(500), nullable=True)
    measurement_uncertainty: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    missing_value_reason: Mapped[str | None] = mapped_column(
        _str_enum(MissingValueReason), nullable=True
    )


class SpeciesORM(ProvenanceMixin, Base):
    __tablename__ = "species"

    code: Mapped[SpeciesCode] = mapped_column(_str_enum(SpeciesCode), unique=True, index=True)
    common_name: Mapped[str] = mapped_column(sa.String(200), default="")
    family: Mapped[str] = mapped_column(sa.String(200), default="")


class PlantSampleORM(ProvenanceMixin, Base):
    __tablename__ = "plant_samples"

    species_code: Mapped[SpeciesCode] = mapped_column(
        _str_enum(SpeciesCode), ForeignKey("species.code"), index=True
    )
    plant_part: Mapped[PlantPart] = mapped_column(_str_enum(PlantPart))
    pigment_class: Mapped[PigmentClass] = mapped_column(_str_enum(PigmentClass))
    batch_code: Mapped[str] = mapped_column(sa.String(100), unique=True)
    collection_date: Mapped[date] = mapped_column(sa.Date)
    collection_location: Mapped[str] = mapped_column(sa.String(300))
    authentication_method: Mapped[str] = mapped_column(sa.String(300), default="")
    voucher_id: Mapped[str] = mapped_column(sa.String(100), default="")


class PigmentExtractBatchORM(ProvenanceMixin, Base):
    __tablename__ = "pigment_extract_batches"

    plant_sample_id: Mapped[uuid_module.UUID] = mapped_column(
        GUID(), ForeignKey("plant_samples.id"), index=True
    )
    batch_code: Mapped[str] = mapped_column(sa.String(100), unique=True)
    solvent: Mapped[SolventType] = mapped_column(_str_enum(SolventType))
    solvent_composition: Mapped[str] = mapped_column(sa.String(300), default="")
    solvent_ratio: Mapped[str] = mapped_column(sa.String(100), default="")
    ph: Mapped[float] = mapped_column(sa.Float)
    extraction_temperature_c: Mapped[float] = mapped_column(sa.Float)
    extraction_time_min: Mapped[float] = mapped_column(sa.Float)
    pigment_concentration_mm: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    extraction_yield_percent: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    protocol_reference: Mapped[str] = mapped_column(sa.String(300), default="")

    __table_args__ = (CheckConstraint("ph >= 0 AND ph <= 14", name="ck_extract_ph_range"),)


class ExperimentalConditionORM(ProvenanceMixin, Base):
    __tablename__ = "experimental_conditions"

    species_code: Mapped[SpeciesCode] = mapped_column(
        _str_enum(SpeciesCode), ForeignKey("species.code"), index=True
    )
    condition_code: Mapped[str] = mapped_column(sa.String(100), unique=True)
    design_source: Mapped[DesignSource] = mapped_column(_str_enum(DesignSource))
    state: Mapped[ClosedLoopState] = mapped_column(_str_enum(ClosedLoopState))

    # Flattened ExperimentalConditionVector (PLAN.txt Section G) -- one
    # row per condition, individually queryable/filterable, rather than a
    # JSON blob (Chunk 3.x's feature/dataset assembly reads these columns
    # directly).
    solvent: Mapped[SolventType] = mapped_column(_str_enum(SolventType))
    solvent_ratio: Mapped[float] = mapped_column(sa.Float)
    ph: Mapped[float] = mapped_column(sa.Float)
    extraction_temperature_c: Mapped[float] = mapped_column(sa.Float)
    extraction_time_min: Mapped[float] = mapped_column(sa.Float)
    pigment_concentration_mm: Mapped[float] = mapped_column(sa.Float)
    adsorption_time_min: Mapped[float] = mapped_column(sa.Float)
    sensitization_concentration_mm: Mapped[float] = mapped_column(sa.Float)
    sensitization_temperature_c: Mapped[float] = mapped_column(sa.Float)
    tio2_thickness_um: Mapped[float] = mapped_column(sa.Float)
    sintering_temperature_c: Mapped[float] = mapped_column(sa.Float)
    sintering_time_min: Mapped[float] = mapped_column(sa.Float)
    surface_treatment: Mapped[SurfaceTreatment] = mapped_column(_str_enum(SurfaceTreatment))

    __table_args__ = (CheckConstraint("ph >= 0 AND ph <= 14", name="ck_condition_ph_range"),)


class PhotoanodeFabricationBatchORM(ProvenanceMixin, Base):
    __tablename__ = "photoanode_fabrication_batches"

    batch_code: Mapped[str] = mapped_column(sa.String(100), unique=True)
    tio2_thickness_um: Mapped[float] = mapped_column(sa.Float)
    tio2_preparation_method: Mapped[str] = mapped_column(sa.String(200), default="")
    sintering_temperature_c: Mapped[float] = mapped_column(sa.Float)
    sintering_time_min: Mapped[float] = mapped_column(sa.Float)
    surface_treatment: Mapped[SurfaceTreatment] = mapped_column(_str_enum(SurfaceTreatment))
    protocol_reference: Mapped[str] = mapped_column(sa.String(300), default="")


class DeviceORM(ProvenanceMixin, Base):
    __tablename__ = "devices"

    experimental_condition_id: Mapped[uuid_module.UUID] = mapped_column(
        GUID(), ForeignKey("experimental_conditions.id"), index=True
    )
    photoanode_fabrication_batch_id: Mapped[uuid_module.UUID] = mapped_column(
        GUID(), ForeignKey("photoanode_fabrication_batches.id"), index=True
    )
    device_code: Mapped[str] = mapped_column(sa.String(100), unique=True)
    replicate_number: Mapped[int] = mapped_column(sa.Integer)
    active_area_cm2: Mapped[float] = mapped_column(sa.Float)
    electrolyte_type: Mapped[ElectrolyteType] = mapped_column(_str_enum(ElectrolyteType))
    counter_electrode_type: Mapped[CounterElectrodeType] = mapped_column(
        _str_enum(CounterElectrodeType)
    )
    device_batch_code: Mapped[str] = mapped_column(sa.String(100), default="")

    __table_args__ = (CheckConstraint("replicate_number > 0", name="ck_device_replicate_positive"),)


class SensitizationRunORM(ProvenanceMixin, Base):
    __tablename__ = "sensitization_runs"

    device_id: Mapped[uuid_module.UUID] = mapped_column(
        GUID(), ForeignKey("devices.id"), index=True
    )
    pigment_extract_batch_id: Mapped[uuid_module.UUID] = mapped_column(
        GUID(), ForeignKey("pigment_extract_batches.id"), index=True
    )
    adsorption_time_min: Mapped[float] = mapped_column(sa.Float)
    sensitization_concentration_mm: Mapped[float] = mapped_column(sa.Float)
    sensitization_temperature_c: Mapped[float] = mapped_column(sa.Float)
    treatment_notes: Mapped[str] = mapped_column(sa.Text, default="")


class RawSpectrumORM(MeasurementProvenanceMixin, Base):
    __tablename__ = "raw_spectra"

    measurement_type: Mapped[MeasurementType] = mapped_column(_str_enum(MeasurementType))
    pigment_extract_batch_id: Mapped[uuid_module.UUID | None] = mapped_column(
        GUID(), ForeignKey("pigment_extract_batches.id"), nullable=True, index=True
    )
    device_id: Mapped[uuid_module.UUID | None] = mapped_column(
        GUID(), ForeignKey("devices.id"), nullable=True, index=True
    )
    # Required here (unlike the mixin's nullable default), matching the
    # domain model's override -- enforced at the DB level too as
    # defense-in-depth against a direct write bypassing the domain layer.
    raw_data_reference: Mapped[str] = mapped_column(sa.String(500), nullable=False)
    acquisition_date: Mapped[date] = mapped_column(sa.Date)


class DerivedDescriptorORM(ProvenanceMixin, Base):
    __tablename__ = "derived_descriptors"

    source_spectrum_id: Mapped[uuid_module.UUID] = mapped_column(
        GUID(), ForeignKey("raw_spectra.id"), index=True
    )
    descriptor_name: Mapped[str] = mapped_column(sa.String(100))
    value: Mapped[float] = mapped_column(sa.Float)
    unit: Mapped[str] = mapped_column(sa.String(50), default="")
    processing_version: Mapped[str] = mapped_column(sa.String(50))
    algorithm: Mapped[str] = mapped_column(sa.String(200), default="")


class ElectrochemicalMeasurementORM(MeasurementProvenanceMixin, Base):
    __tablename__ = "electrochemical_measurements"

    device_id: Mapped[uuid_module.UUID] = mapped_column(
        GUID(), ForeignKey("devices.id"), index=True
    )
    raw_spectrum_id: Mapped[uuid_module.UUID] = mapped_column(
        GUID(), ForeignKey("raw_spectra.id"), index=True
    )
    equivalent_circuit_model: Mapped[EquivalentCircuitModel] = mapped_column(
        _str_enum(EquivalentCircuitModel)
    )
    series_resistance_ohm: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    charge_transfer_resistance_ohm: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    recombination_resistance_ohm: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    electron_lifetime_ms: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    fit_r_squared: Mapped[float | None] = mapped_column(sa.Float, nullable=True)


class PhotovoltaicMeasurementORM(MeasurementProvenanceMixin, Base):
    __tablename__ = "photovoltaic_measurements"

    device_id: Mapped[uuid_module.UUID] = mapped_column(
        GUID(), ForeignKey("devices.id"), index=True
    )
    raw_curve_id: Mapped[uuid_module.UUID | None] = mapped_column(
        GUID(), ForeignKey("raw_spectra.id"), nullable=True, index=True
    )
    illumination_protocol: Mapped[IlluminationProtocol] = mapped_column(
        _str_enum(IlluminationProtocol)
    )
    measurement_date: Mapped[date] = mapped_column(sa.Date)
    measurement_replicate_number: Mapped[int] = mapped_column(sa.Integer, default=1)
    open_circuit_voltage_v: Mapped[float] = mapped_column(sa.Float)
    short_circuit_current_density_ma_cm2: Mapped[float] = mapped_column(sa.Float)
    fill_factor: Mapped[float] = mapped_column(sa.Float)
    power_conversion_efficiency_percent: Mapped[float] = mapped_column(sa.Float)

    __table_args__ = (
        CheckConstraint("fill_factor >= 0 AND fill_factor <= 1", name="ck_pv_fill_factor_range"),
        CheckConstraint("power_conversion_efficiency_percent >= 0", name="ck_pv_pce_nonneg"),
    )


class StabilityMeasurementORM(MeasurementProvenanceMixin, Base):
    __tablename__ = "stability_measurements"

    device_id: Mapped[uuid_module.UUID] = mapped_column(
        GUID(), ForeignKey("devices.id"), index=True
    )
    baseline_measurement_id: Mapped[uuid_module.UUID] = mapped_column(
        GUID(), ForeignKey("photovoltaic_measurements.id")
    )
    timepoint_measurement_id: Mapped[uuid_module.UUID] = mapped_column(
        GUID(), ForeignKey("photovoltaic_measurements.id")
    )
    ageing_duration_h: Mapped[float] = mapped_column(sa.Float)
    ageing_conditions: Mapped[str] = mapped_column(sa.String(300), default="")
    retention_percent: Mapped[float] = mapped_column(sa.Float)
