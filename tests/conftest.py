"""Shared pytest fixtures for database-backed tests."""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from dssc_discovery.database.models import Base


@pytest.fixture
def engine():
    eng = sa.create_engine("sqlite://")

    @sa.event.listens_for(eng, "connect")
    def _enable_fk(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def session(engine):
    with Session(engine) as s:
        yield s
