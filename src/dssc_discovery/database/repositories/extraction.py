from __future__ import annotations

from dssc_discovery.database.models import PigmentExtractBatchORM, PlantSampleORM
from dssc_discovery.database.repositories.base import Repository
from dssc_discovery.domain.extraction import PigmentExtractBatch, PlantSample


class PlantSampleRepository(Repository[PlantSample, PlantSampleORM]):
    domain_cls = PlantSample
    orm_cls = PlantSampleORM


class PigmentExtractBatchRepository(Repository[PigmentExtractBatch, PigmentExtractBatchORM]):
    domain_cls = PigmentExtractBatch
    orm_cls = PigmentExtractBatchORM
