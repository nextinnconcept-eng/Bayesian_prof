"""Unit tests for behavior verified manually while building Chunk 1.4:
SQLAlchemy's Enum type defaults to storing member.name (not .value), and
SQLite has foreign key enforcement OFF by default. Both would silently
weaken the schema if unguarded -- these pin the fixes as regression tests.
"""

from __future__ import annotations

import datetime

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from dssc_discovery.database.models import (
    PigmentExtractBatchORM,
    PlantSampleORM,
    SpeciesORM,
)
from dssc_discovery.domain.enums import PigmentClass, PlantPart, SolventType, SpeciesCode

# engine fixture comes from tests/conftest.py (shared across all database tests)


def _make_species(session: Session, code: SpeciesCode = SpeciesCode.BETA_VULGARIS) -> SpeciesORM:
    species = SpeciesORM(code=code, operator="J. Doe")
    session.add(species)
    session.commit()
    return species


def _make_plant_sample(session: Session, species_code: SpeciesCode) -> PlantSampleORM:
    sample = PlantSampleORM(
        species_code=species_code,
        plant_part=PlantPart.ROOT,
        pigment_class=PigmentClass.BETALAIN,
        batch_code="BV-001",
        collection_date=datetime.date(2026, 1, 1),
        collection_location="Farm",
        operator="J. Doe",
    )
    session.add(sample)
    session.commit()
    return sample


def test_enum_columns_store_value_not_name(engine):
    with Session(engine) as session:
        _make_species(session)
        raw = session.execute(sa.text("SELECT code FROM species")).scalar()
        assert raw == "Beta vulgaris"  # the .value; NOT "BETA_VULGARIS" (.name)


def test_foreign_key_violation_is_rejected(engine):
    with Session(engine) as session:
        _make_species(session, SpeciesCode.BETA_VULGARIS)

        # references a species that was never inserted
        bad_sample = PlantSampleORM(
            species_code=SpeciesCode.CHROMOLAENA_ODORATA,
            plant_part=PlantPart.LEAF,
            pigment_class=PigmentClass.FLAVONOID,
            batch_code="CO-999",
            collection_date=datetime.date(2026, 1, 1),
            collection_location="Farm",
            operator="J. Doe",
        )
        session.add(bad_sample)
        with pytest.raises(sa.exc.IntegrityError):
            session.commit()


def test_valid_foreign_key_reference_succeeds(engine):
    with Session(engine) as session:
        _make_species(session, SpeciesCode.BETA_VULGARIS)
        sample = _make_plant_sample(session, SpeciesCode.BETA_VULGARIS)
        assert sample.id is not None


def test_check_constraint_rejects_ph_out_of_range(engine):
    with Session(engine) as session:
        _make_species(session, SpeciesCode.BETA_VULGARIS)
        sample = _make_plant_sample(session, SpeciesCode.BETA_VULGARIS)

        bad_batch = PigmentExtractBatchORM(
            plant_sample_id=sample.id,
            batch_code="BV-EXT-001",
            solvent=SolventType.ETHANOL,
            ph=20.0,  # out of [0, 14]
            extraction_temperature_c=25.0,
            extraction_time_min=60.0,
            operator="J. Doe",
        )
        session.add(bad_batch)
        with pytest.raises(sa.exc.IntegrityError):
            session.commit()


def test_valid_pigment_extract_batch_round_trips(engine):
    with Session(engine) as session:
        _make_species(session, SpeciesCode.BETA_VULGARIS)
        sample = _make_plant_sample(session, SpeciesCode.BETA_VULGARIS)

        batch = PigmentExtractBatchORM(
            plant_sample_id=sample.id,
            batch_code="BV-EXT-001",
            solvent=SolventType.ETHANOL,
            ph=5.5,
            extraction_temperature_c=25.0,
            extraction_time_min=60.0,
            operator="J. Doe",
        )
        session.add(batch)
        session.commit()

        fetched = session.get(PigmentExtractBatchORM, batch.id)
        assert fetched is not None
        assert fetched.solvent is SolventType.ETHANOL
        assert fetched.ph == 5.5
