from __future__ import annotations

from uuid import UUID

from dssc_discovery.database.models import (
    BOIterationORM,
    ClosedLoopStateLogORM,
    HypothesisLogORM,
    RecommendationORM,
)
from dssc_discovery.database.repositories.base import Repository
from dssc_discovery.domain.enums import HypothesisStatus
from dssc_discovery.explainability.records import HypothesisRecord
from dssc_discovery.optimization.records import BOIteration, ClosedLoopStateLog, Recommendation


class HypothesisLogRepository(Repository[HypothesisRecord, HypothesisLogORM]):
    domain_cls = HypothesisRecord
    orm_cls = HypothesisLogORM

    def set_status(self, id: UUID, status: HypothesisStatus, note: str) -> HypothesisRecord:
        orm_obj = self._session.get(HypothesisLogORM, id)
        if orm_obj is None:
            raise KeyError(f"no hypothesis {id}")
        orm_obj.status = status
        orm_obj.status_note = note
        self._session.flush()
        return self._to_domain(orm_obj)


class BOIterationRepository(Repository[BOIteration, BOIterationORM]):
    domain_cls = BOIteration
    orm_cls = BOIterationORM

    def set_convergence(self, id: UUID, convergence_json: str) -> None:
        orm_obj = self._session.get(BOIterationORM, id)
        if orm_obj is None:
            raise KeyError(f"no BO iteration {id}")
        orm_obj.convergence_json = convergence_json
        self._session.flush()


class RecommendationRepository(Repository[Recommendation, RecommendationORM]):
    domain_cls = Recommendation
    orm_cls = RecommendationORM


class ClosedLoopStateLogRepository(Repository[ClosedLoopStateLog, ClosedLoopStateLogORM]):
    domain_cls = ClosedLoopStateLog
    orm_cls = ClosedLoopStateLogORM
