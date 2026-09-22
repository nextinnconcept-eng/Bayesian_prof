"""Tests for audit logging and append-only enforcement (Chunk 1.8).

The hooks under test are registered globally at import time (see
database/audit.py's module docstring) -- these tests don't register
anything themselves, they verify the hooks that are already active by
the time any dssc_discovery.database module is imported.
"""

from __future__ import annotations

import json

import pytest

from dssc_discovery.database.audit import AppendOnlyViolationError, AuditLogORM
from dssc_discovery.database.models import (
    DerivedDescriptorORM,
    ElectrochemicalMeasurementORM,
    PhotovoltaicMeasurementORM,
    RawSpectrumORM,
    SpeciesORM,
    StabilityMeasurementORM,
)
from dssc_discovery.domain.enums import SpeciesCode
from tests.seed_helpers import seed_full_device_chain, seed_raw_spectrum_for_device, seed_species


def _audit_rows_for(session, table_name: str) -> list[AuditLogORM]:
    return session.query(AuditLogORM).filter_by(table_name=table_name).all()


# --- audit logging on insert/update/delete ----------------------------------


def test_insert_is_logged(session):
    seed_species(session, SpeciesCode.BETA_VULGARIS)
    session.commit()

    rows = _audit_rows_for(session, "species")
    assert len(rows) == 1
    assert rows[0].operation == "insert"


def test_update_on_a_non_append_only_table_is_logged(session):
    seed_species(session, SpeciesCode.BETA_VULGARIS)
    session.commit()

    species_row = session.query(SpeciesORM).one()
    species_row.common_name = "beetroot"
    session.commit()

    update_rows = [r for r in _audit_rows_for(session, "species") if r.operation == "update"]
    assert len(update_rows) == 1
    changes = json.loads(update_rows[0].changed_fields)
    assert changes["common_name"] == [None, "beetroot"] or changes["common_name"][1] == "beetroot"


def test_delete_is_logged(session):
    seed_species(session, SpeciesCode.BETA_VULGARIS)
    session.commit()

    species_row = session.query(SpeciesORM).one()
    session.delete(species_row)
    session.commit()

    delete_rows = [r for r in _audit_rows_for(session, "species") if r.operation == "delete"]
    assert len(delete_rows) == 1


def test_audit_log_entries_are_not_themselves_audited(session):
    seed_species(session, SpeciesCode.BETA_VULGARIS)
    session.commit()
    # exactly one entry (the species insert) -- not two (which would
    # happen if inserting the audit row triggered auditing itself)
    assert session.query(AuditLogORM).count() == 1


# --- append-only enforcement --------------------------------------------------

_APPEND_ONLY_ORM_CLASSES = [
    RawSpectrumORM,
    DerivedDescriptorORM,
    ElectrochemicalMeasurementORM,
    PhotovoltaicMeasurementORM,
    StabilityMeasurementORM,
]


def test_raw_spectrum_update_is_rejected(session):
    device = seed_full_device_chain(session, species_code=SpeciesCode.BETA_VULGARIS)
    session.commit()
    spectrum = seed_raw_spectrum_for_device(session, device_id=device.id)
    session.commit()

    orm_row = session.get(RawSpectrumORM, spectrum.id)
    orm_row.raw_data_reference = "tampered"
    with pytest.raises(AppendOnlyViolationError):
        session.commit()
    session.rollback()


def test_raw_spectrum_insert_is_not_blocked(session):
    # the guard is on before_update specifically, not before_insert --
    # inserting new raw data must still work.
    device = seed_full_device_chain(session, species_code=SpeciesCode.BETA_VULGARIS)
    session.commit()
    spectrum = seed_raw_spectrum_for_device(session, device_id=device.id)
    session.commit()
    assert spectrum.id is not None


def _build_append_only_rows(session):
    """One valid, persisted row per append-only ORM class, keyed by
    class, covering the full FK chain each needs."""
    from datetime import date

    from dssc_discovery.database.repositories import (
        DerivedDescriptorRepository,
        ElectrochemicalMeasurementRepository,
        PhotovoltaicMeasurementRepository,
        StabilityMeasurementRepository,
    )
    from dssc_discovery.domain.enums import EquivalentCircuitModel, IlluminationProtocol
    from dssc_discovery.domain.measurements import (
        DerivedDescriptor,
        ElectrochemicalMeasurement,
        PhotovoltaicMeasurement,
        StabilityMeasurement,
    )

    device = seed_full_device_chain(session, species_code=SpeciesCode.BETA_VULGARIS)
    spectrum = seed_raw_spectrum_for_device(session, device_id=device.id)

    descriptor = DerivedDescriptorRepository(session).add(
        DerivedDescriptor(
            source_spectrum_id=spectrum.id,
            descriptor_name="x",
            value=1.0,
            processing_version="v1",
            operator="J. Doe",
        )
    )
    eis = ElectrochemicalMeasurementRepository(session).add(
        ElectrochemicalMeasurement(
            device_id=device.id,
            raw_spectrum_id=spectrum.id,
            equivalent_circuit_model=EquivalentCircuitModel.RANDLES,
            operator="J. Doe",
            instrument="Autolab PGSTAT",
        )
    )
    pv = PhotovoltaicMeasurementRepository(session).add(
        PhotovoltaicMeasurement(
            device_id=device.id,
            illumination_protocol=IlluminationProtocol.AM1_5G_1_SUN,
            measurement_date=date(2026, 1, 20),
            open_circuit_voltage_v=0.65,
            short_circuit_current_density_ma_cm2=12.0,
            fill_factor=0.65,
            power_conversion_efficiency_percent=5.07,
            operator="J. Doe",
            instrument="Keithley 2400",
        )
    )
    pv2 = PhotovoltaicMeasurementRepository(session).add(
        PhotovoltaicMeasurement(
            device_id=device.id,
            illumination_protocol=IlluminationProtocol.AM1_5G_1_SUN,
            measurement_date=date(2026, 1, 27),
            open_circuit_voltage_v=0.60,
            short_circuit_current_density_ma_cm2=11.0,
            fill_factor=0.60,
            power_conversion_efficiency_percent=4.6,
            operator="J. Doe",
            instrument="Keithley 2400",
        )
    )
    stability = StabilityMeasurementRepository(session).add(
        StabilityMeasurement(
            device_id=device.id,
            baseline_measurement_id=pv.id,
            timepoint_measurement_id=pv2.id,
            ageing_duration_h=168.0,
            retention_percent=90.8,
            operator="J. Doe",
            instrument="Keithley 2400",
        )
    )
    session.commit()

    return {
        RawSpectrumORM: spectrum.id,
        DerivedDescriptorORM: descriptor.id,
        ElectrochemicalMeasurementORM: eis.id,
        PhotovoltaicMeasurementORM: pv.id,
        StabilityMeasurementORM: stability.id,
    }


@pytest.mark.parametrize("orm_cls", _APPEND_ONLY_ORM_CLASSES)
def test_every_append_only_table_rejects_update(session, orm_cls):
    ids_by_class = _build_append_only_rows(session)
    orm_row = session.get(orm_cls, ids_by_class[orm_cls])

    orm_row.notes = "edited after the fact"  # every ORM class has .notes (ProvenanceMixin)

    with pytest.raises(AppendOnlyViolationError):
        session.commit()
    session.rollback()


def test_species_table_is_not_append_only(session):
    # scoping check: append-only enforcement must NOT apply to
    # procedural/reference tables like species -- only to raw/measured data.
    seed_species(session, SpeciesCode.BETA_VULGARIS)
    session.commit()

    species_row = session.query(SpeciesORM).one()
    species_row.common_name = "beetroot"
    session.commit()  # must NOT raise AppendOnlyViolationError

    refetched = session.get(SpeciesORM, species_row.id)
    assert refetched.common_name == "beetroot"


def test_append_only_orm_classes_cover_all_raw_and_measurement_tables():
    covered = {cls.__tablename__ for cls in _APPEND_ONLY_ORM_CLASSES}
    assert covered == {
        "raw_spectra",
        "derived_descriptors",
        "electrochemical_measurements",
        "photovoltaic_measurements",
        "stability_measurements",
    }
