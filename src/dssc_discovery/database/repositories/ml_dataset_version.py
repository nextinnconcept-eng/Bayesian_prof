from __future__ import annotations

from dssc_discovery.database.models import MLDatasetVersionORM
from dssc_discovery.database.repositories.base import Repository
from dssc_discovery.features.ml_dataset_version import MLDatasetVersion


class MLDatasetVersionRepository(Repository[MLDatasetVersion, MLDatasetVersionORM]):
    domain_cls = MLDatasetVersion
    orm_cls = MLDatasetVersionORM
