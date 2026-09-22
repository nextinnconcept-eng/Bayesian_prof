"""Shared builders for valid, FK-chained domain objects, persisted via
their repositories. Not a test module itself (no test_ prefix) -- used by
the repository test files to avoid re-deriving the full
species -> plant_sample -> pigment_extract_batch -> experimental_condition
-> photoanode_fabrication_batch -> device -> ... chain in every test.

Every function persists via its repository (not a raw ORM insert), so
using these also continuously exercises the repositories they're built on.
"""

from __future__ import annotations

from datetime import date
from uuid import UUID

from sqlalchemy.orm import Session

from dssc_discovery.database.repositories import (
    DeviceRepository,
    ExperimentalConditionRepository,
    PhotoanodeFabricationBatchRepository,
    PigmentExtractBatchRepository,
    PlantSampleRepository,
    RawSpectrumRepository,
    SensitizationRunRepository,
    SpeciesRepository,
)
from dssc_discovery.domain.device import Device, PhotoanodeFabricationBatch, SensitizationRun
from dssc_discovery.domain.enums import (
    CounterElectrodeType,
    DesignSource,
    ElectrolyteType,
    MeasurementType,
    PigmentClass,
    PlantPart,
    SolventType,
    SpeciesCode,
    SurfaceTreatment,
)
from dssc_discovery.domain.experimental_condition import (
    ExperimentalCondition,
    ExperimentalConditionVector,
)
from dssc_discovery.domain.extraction import PigmentExtractBatch, PlantSample
from dssc_discovery.domain.measurements import RawSpectrum
from dssc_discovery.domain.species import Species


def seed_species(session: Session, code: SpeciesCode = SpeciesCode.BETA_VULGARIS) -> Species:
    return SpeciesRepository(session).add(Species(code=code, operator="J. Doe"))


def seed_plant_sample(
    session: Session, species_code: SpeciesCode = SpeciesCode.BETA_VULGARIS, **overrides
) -> PlantSample:
    kwargs = dict(
        species_code=species_code,
        plant_part=PlantPart.ROOT,
        pigment_class=PigmentClass.BETALAIN,
        batch_code=f"{species_code.name}-SAMPLE-{overrides.get('_suffix', '001')}",
        collection_date=date(2026, 1, 1),
        collection_location="Test Farm",
        operator="J. Doe",
    )
    kwargs.pop("_suffix", None)
    kwargs.update({k: v for k, v in overrides.items() if k != "_suffix"})
    return PlantSampleRepository(session).add(PlantSample(**kwargs))


def seed_pigment_extract_batch(
    session: Session, plant_sample_id: UUID, **overrides
) -> PigmentExtractBatch:
    kwargs = dict(
        plant_sample_id=plant_sample_id,
        batch_code=overrides.pop("batch_code", "EXT-001"),
        solvent=SolventType.ETHANOL,
        ph=5.5,
        extraction_temperature_c=25.0,
        extraction_time_min=60.0,
        operator="J. Doe",
    )
    kwargs.update(overrides)
    return PigmentExtractBatchRepository(session).add(PigmentExtractBatch(**kwargs))


def _default_vector(**overrides) -> ExperimentalConditionVector:
    kwargs = dict(
        solvent=SolventType.ETHANOL,
        solvent_ratio=0.7,
        ph=5.5,
        extraction_temperature_c=25.0,
        extraction_time_min=60.0,
        pigment_concentration_mm=10.0,
        adsorption_time_min=120.0,
        sensitization_concentration_mm=0.5,
        sensitization_temperature_c=25.0,
        tio2_thickness_um=12.0,
        sintering_temperature_c=450.0,
        sintering_time_min=30.0,
        surface_treatment=SurfaceTreatment.NONE,
    )
    kwargs.update(overrides)
    return ExperimentalConditionVector(**kwargs)


def seed_experimental_condition(
    session: Session, species_code: SpeciesCode = SpeciesCode.BETA_VULGARIS, **overrides
) -> ExperimentalCondition:
    condition_code = overrides.pop("condition_code", "BV-COND-001")
    kwargs = dict(
        species_code=species_code,
        condition_code=condition_code,
        design_source=DesignSource.INITIAL_DOE,
        target_parameters=_default_vector(),
        operator="J. Doe",
    )
    kwargs.update(overrides)
    return ExperimentalConditionRepository(session).add(ExperimentalCondition(**kwargs))


def seed_photoanode_fabrication_batch(session: Session, **overrides) -> PhotoanodeFabricationBatch:
    kwargs = dict(
        batch_code=overrides.pop("batch_code", "PHOTOANODE-001"),
        tio2_thickness_um=12.0,
        sintering_temperature_c=450.0,
        sintering_time_min=30.0,
        surface_treatment=SurfaceTreatment.NONE,
        operator="J. Doe",
    )
    kwargs.update(overrides)
    return PhotoanodeFabricationBatchRepository(session).add(PhotoanodeFabricationBatch(**kwargs))


def seed_device(
    session: Session,
    experimental_condition_id: UUID,
    photoanode_fabrication_batch_id: UUID,
    **overrides,
) -> Device:
    kwargs = dict(
        experimental_condition_id=experimental_condition_id,
        photoanode_fabrication_batch_id=photoanode_fabrication_batch_id,
        device_code=overrides.pop("device_code", "BV-COND-001-R1"),
        replicate_number=1,
        active_area_cm2=0.25,
        electrolyte_type=ElectrolyteType.IODIDE_TRIIODIDE,
        counter_electrode_type=CounterElectrodeType.PLATINUM,
        operator="J. Doe",
    )
    kwargs.update(overrides)
    return DeviceRepository(session).add(Device(**kwargs))


def seed_sensitization_run(
    session: Session, device_id: UUID, pigment_extract_batch_id: UUID, **overrides
) -> SensitizationRun:
    kwargs = dict(
        device_id=device_id,
        pigment_extract_batch_id=pigment_extract_batch_id,
        adsorption_time_min=120.0,
        sensitization_concentration_mm=0.5,
        sensitization_temperature_c=25.0,
        operator="J. Doe",
    )
    kwargs.update(overrides)
    return SensitizationRunRepository(session).add(SensitizationRun(**kwargs))


def seed_raw_spectrum_for_device(session: Session, device_id: UUID, **overrides) -> RawSpectrum:
    kwargs = dict(
        measurement_type=MeasurementType.EIS,
        device_id=device_id,
        raw_data_reference="s3://raw/eis/001.csv",
        acquisition_date=date(2026, 1, 20),
        operator="J. Doe",
        instrument="Autolab PGSTAT",
    )
    kwargs.update(overrides)
    return RawSpectrumRepository(session).add(RawSpectrum(**kwargs))


def seed_full_device_chain(session: Session, **overrides) -> Device:
    """species -> experimental_condition + photoanode_fabrication_batch -> device."""
    species_code = overrides.pop("species_code", SpeciesCode.BETA_VULGARIS)
    seed_species(session, species_code)
    condition = seed_experimental_condition(session, species_code)
    fab_batch = seed_photoanode_fabrication_batch(session)
    return seed_device(
        session,
        experimental_condition_id=condition.id,
        photoanode_fabrication_batch_id=fab_batch.id,
        **overrides,
    )
