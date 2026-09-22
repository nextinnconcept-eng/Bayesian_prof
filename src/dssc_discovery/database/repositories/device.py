from __future__ import annotations

from dssc_discovery.database.models import (
    DeviceORM,
    PhotoanodeFabricationBatchORM,
    SensitizationRunORM,
)
from dssc_discovery.database.repositories.base import Repository
from dssc_discovery.domain.device import Device, PhotoanodeFabricationBatch, SensitizationRun


class PhotoanodeFabricationBatchRepository(
    Repository[PhotoanodeFabricationBatch, PhotoanodeFabricationBatchORM]
):
    domain_cls = PhotoanodeFabricationBatch
    orm_cls = PhotoanodeFabricationBatchORM


class DeviceRepository(Repository[Device, DeviceORM]):
    domain_cls = Device
    orm_cls = DeviceORM


class SensitizationRunRepository(Repository[SensitizationRun, SensitizationRunORM]):
    domain_cls = SensitizationRun
    orm_cls = SensitizationRunORM
