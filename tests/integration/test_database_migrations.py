"""Integration tests for Alembic migrations against a real (temporary)
SQLite database -- Chunk 1.4's acceptance criteria: a migration applies
cleanly to an empty file, all tables/FKs are created, and rollback works.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic.config import Config

from alembic import command
from dssc_discovery.config import REPO_ROOT

EXPECTED_TABLES = {
    "species",
    "plant_samples",
    "pigment_extract_batches",
    "experimental_conditions",
    "photoanode_fabrication_batches",
    "devices",
    "sensitization_runs",
    "raw_spectra",
    "derived_descriptors",
    "electrochemical_measurements",
    "photovoltaic_measurements",
    "stability_measurements",
}


def _alembic_config(db_path) -> Config:
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{db_path.as_posix()}")
    return config


def test_upgrade_head_creates_all_tables_on_empty_db(tmp_path):
    db_path = tmp_path / "test_migration.db"
    config = _alembic_config(db_path)

    command.upgrade(config, "head")

    engine = sa.create_engine(f"sqlite:///{db_path.as_posix()}")
    tables = set(sa.inspect(engine).get_table_names())
    assert EXPECTED_TABLES <= tables
    assert "alembic_version" in tables
    engine.dispose()


def test_upgrade_creates_expected_foreign_keys(tmp_path):
    db_path = tmp_path / "test_migration.db"
    config = _alembic_config(db_path)
    command.upgrade(config, "head")

    engine = sa.create_engine(f"sqlite:///{db_path.as_posix()}")
    insp = sa.inspect(engine)

    device_fks = {fk["referred_table"] for fk in insp.get_foreign_keys("devices")}
    assert device_fks == {"experimental_conditions", "photoanode_fabrication_batches"}

    plant_sample_fks = {fk["referred_table"] for fk in insp.get_foreign_keys("plant_samples")}
    assert plant_sample_fks == {"species"}

    stability_fks = {fk["referred_table"] for fk in insp.get_foreign_keys("stability_measurements")}
    assert stability_fks == {"devices", "photovoltaic_measurements"}
    engine.dispose()


def test_downgrade_base_drops_all_domain_tables(tmp_path):
    db_path = tmp_path / "test_migration.db"
    config = _alembic_config(db_path)

    command.upgrade(config, "head")
    command.downgrade(config, "base")

    engine = sa.create_engine(f"sqlite:///{db_path.as_posix()}")
    tables = set(sa.inspect(engine).get_table_names())
    assert tables & EXPECTED_TABLES == set()
    engine.dispose()


def test_reapplying_upgrade_head_is_a_noop(tmp_path):
    db_path = tmp_path / "test_migration.db"
    config = _alembic_config(db_path)

    command.upgrade(config, "head")
    command.upgrade(config, "head")  # must not error

    engine = sa.create_engine(f"sqlite:///{db_path.as_posix()}")
    tables = set(sa.inspect(engine).get_table_names())
    assert EXPECTED_TABLES <= tables
    engine.dispose()
