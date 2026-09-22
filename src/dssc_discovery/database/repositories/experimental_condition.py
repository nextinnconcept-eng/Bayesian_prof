"""ExperimentalConditionRepository -- the one repository that isn't a flat
1:1 mapping. ExperimentalCondition.target_parameters is a nested
ExperimentalConditionVector in the domain model, but flattened onto
individual columns on experimental_conditions in the DB (Chunk 1.4), so
conversion needs to flatten on the way in and reassemble on the way out.
"""

from __future__ import annotations

from typing import Any

from dssc_discovery.database.models import ExperimentalConditionORM
from dssc_discovery.database.repositories.base import Repository
from dssc_discovery.domain.experimental_condition import (
    ExperimentalCondition,
    ExperimentalConditionVector,
)

_VECTOR_FIELDS = frozenset(ExperimentalConditionVector.model_fields)


class ExperimentalConditionRepository(Repository[ExperimentalCondition, ExperimentalConditionORM]):
    domain_cls = ExperimentalCondition
    orm_cls = ExperimentalConditionORM

    def _to_orm(self, domain_obj: ExperimentalCondition) -> ExperimentalConditionORM:
        data = domain_obj.model_dump(exclude={"target_parameters"})
        data.update(domain_obj.target_parameters.model_dump())
        return ExperimentalConditionORM(**data)

    def _to_domain(self, orm_obj: ExperimentalConditionORM) -> ExperimentalCondition:
        columns = orm_obj.__table__.columns
        all_data: dict[str, Any] = {c.name: getattr(orm_obj, c.name) for c in columns}
        vector_data = {k: v for k, v in all_data.items() if k in _VECTOR_FIELDS}
        condition_data = {k: v for k, v in all_data.items() if k not in _VECTOR_FIELDS}
        condition_data["target_parameters"] = ExperimentalConditionVector.model_validate(
            vector_data
        )
        return ExperimentalCondition.model_validate(condition_data)
