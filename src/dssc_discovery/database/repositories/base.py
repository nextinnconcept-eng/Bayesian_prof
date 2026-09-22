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
        orm_obj = self._to_orm(domain_obj)
        self._session.add(orm_obj)
        self._session.commit()
        return self._to_domain(orm_obj)

    def get(self, id: UUID) -> DomainT | None:
        orm_obj = self._session.get(self.orm_cls, id)
        return self._to_domain(orm_obj) if orm_obj is not None else None

    def list(self) -> list[DomainT]:
        rows = self._session.query(self.orm_cls).all()
        return [self._to_domain(row) for row in rows]

    def exists(self, **filters: Any) -> bool:
        query = self._session.query(self.orm_cls).filter_by(**filters)
        return bool(self._session.query(query.exists()).scalar())
