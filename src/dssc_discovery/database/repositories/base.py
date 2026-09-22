"""Generic repository: converts between domain Pydantic objects and
SQLAlchemy ORM rows, and provides get/add/list/exists.

Domain objects are what the rest of the system works with (Section D);
nothing outside dssc_discovery.database should need to import sqlalchemy
directly. Most entities map 1:1 between domain fields and ORM columns, so
the default _to_orm/_to_domain here (via model_dump() and column
introspection) work unmodified -- only ExperimentalConditionRepository
overrides them, to flatten/unflatten its nested target_parameters.
"""

from __future__ import annotations

import builtins
from typing import Any, Generic, TypeVar
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy.orm import Session

DomainT = TypeVar("DomainT", bound=BaseModel)
ORMT = TypeVar("ORMT")


class Repository(Generic[DomainT, ORMT]):
    domain_cls: type[DomainT]
    orm_cls: type[ORMT]

    def __init__(self, session: Session) -> None:
        self._session = session

    def _to_orm(self, domain_obj: DomainT) -> ORMT:
        return self.orm_cls(**domain_obj.model_dump())

    def _to_domain(self, orm_obj: ORMT) -> DomainT:
        columns = orm_obj.__table__.columns  # type: ignore[attr-defined]
        data = {column.name: getattr(orm_obj, column.name) for column in columns}
        return self.domain_cls.model_validate(data)

    def add(self, domain_obj: DomainT) -> DomainT:
        """Stage and flush (not commit) domain_obj. Flush is enough for
        this row to be visible to later reads on the same session -- the
        caller owns the transaction boundary and decides when to commit,
        so multi-entity operations (e.g. CSV ingestion, Chunk 1.7) can
        stage several adds and commit them atomically, or roll all of
        them back together on failure.
        """
        orm_obj = self._to_orm(domain_obj)
        self._session.add(orm_obj)
        self._session.flush()
        return self._to_domain(orm_obj)

    def get(self, id: UUID) -> DomainT | None:
        orm_obj = self._session.get(self.orm_cls, id)
        return self._to_domain(orm_obj) if orm_obj is not None else None

    def get_one(self, **filters: Any) -> DomainT | None:
        """Look up a single row by arbitrary column filters (e.g. a
        natural key like condition_code) rather than by primary key.
        """
        orm_obj = self._session.query(self.orm_cls).filter_by(**filters).one_or_none()
        return self._to_domain(orm_obj) if orm_obj is not None else None

    def list(self) -> list[DomainT]:
        rows = self._session.query(self.orm_cls).all()
        return [self._to_domain(row) for row in rows]

    def filter(self, **filters: Any) -> builtins.list[DomainT]:
        """Look up every row matching arbitrary column filters (e.g. all
        PhotovoltaicMeasurements for a device_id) -- unlike get_one(),
        never raises on more than one match, since several entities
        (repeat J-V scans, multiple EIS attempts) legitimately can have
        more than one row per foreign key.

        Return type is builtins.list[DomainT], fully qualified, on
        purpose: this class already defines a method named `list`, and
        mypy resolves a bare `list[...]` annotation appearing after that
        definition to the METHOD, not the builtin generic -- a real,
        confirmed mypy quirk, not a runtime issue (this module uses
        `from __future__ import annotations`, so annotations are never
        evaluated at runtime), but it breaks static type-checking for
        every caller of this method.
        """
        rows = self._session.query(self.orm_cls).filter_by(**filters).all()
        return [self._to_domain(row) for row in rows]

    def exists(self, **filters: Any) -> bool:
        query = self._session.query(self.orm_cls).filter_by(**filters)
        return bool(self._session.query(query.exists()).scalar())
