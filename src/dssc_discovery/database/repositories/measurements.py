from __future__ import annotations

from dssc_discovery.database.models import (
    DerivedDescriptorORM,
    ElectrochemicalMeasurementORM,
    PhotovoltaicMeasurementORM,
    RawSpectrumORM,
    StabilityMeasurementORM,
)
from dssc_discovery.database.repositories.base import Repository
from dssc_discovery.domain.measurements import (
    DerivedDescriptor,
    ElectrochemicalMeasurement,
    PhotovoltaicMeasurement,
    RawSpectrum,
    StabilityMeasurement,
)


class RawSpectrumRepository(Repository[RawSpectrum, RawSpectrumORM]):
    domain_cls = RawSpectrum
    orm_cls = RawSpectrumORM


class DerivedDescriptorRepository(Repository[DerivedDescriptor, DerivedDescriptorORM]):
    domain_cls = DerivedDescriptor
    orm_cls = DerivedDescriptorORM


class ElectrochemicalMeasurementRepository(
    Repository[ElectrochemicalMeasurement, ElectrochemicalMeasurementORM]
):
    domain_cls = ElectrochemicalMeasurement
    orm_cls = ElectrochemicalMeasurementORM


class PhotovoltaicMeasurementRepository(
    Repository[PhotovoltaicMeasurement, PhotovoltaicMeasurementORM]
):
    domain_cls = PhotovoltaicMeasurement
    orm_cls = PhotovoltaicMeasurementORM


class StabilityMeasurementRepository(Repository[StabilityMeasurement, StabilityMeasurementORM]):
    domain_cls = StabilityMeasurement
    orm_cls = StabilityMeasurementORM
