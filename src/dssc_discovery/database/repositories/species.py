from __future__ import annotations

from dssc_discovery.database.models import SpeciesORM
from dssc_discovery.database.repositories.base import Repository
from dssc_discovery.domain.species import Species


class SpeciesRepository(Repository[Species, SpeciesORM]):
    domain_cls = Species
    orm_cls = SpeciesORM
