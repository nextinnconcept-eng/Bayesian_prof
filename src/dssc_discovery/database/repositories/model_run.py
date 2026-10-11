from __future__ import annotations

from dssc_discovery.database.models import ModelRunORM
from dssc_discovery.database.repositories.base import Repository
from dssc_discovery.models.model_run import ModelRun


class ModelRunRepository(Repository[ModelRun, ModelRunORM]):
    domain_cls = ModelRun
    orm_cls = ModelRunORM
