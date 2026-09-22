"""Domain models for photoanode fabrication, sensitization, and the
physical device replicate (PLAN.txt Section E: PhotoanodeFabricationBatch,
SensitizationRun, Device).

Device references its ExperimentalCondition and PhotoanodeFabricationBatch.
SensitizationRun references the Device it was applied to -- not the
reverse -- so a device can be fabricated (and exist) before it's
sensitized, without a circular FK at construction time.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import Field

from dssc_discovery.domain.base import ProvenancedModel
from dssc_discovery.domain.enums import CounterElectrodeType, ElectrolyteType, SurfaceTreatment


class PhotoanodeFabricationBatch(ProvenancedModel):
    batch_code: str = Field(min_length=1)
    tio2_thickness_um: float = Field(gt=0, le=100)
    tio2_preparation_method: str = ""
    sintering_temperature_c: float = Field(ge=0, le=600)
    sintering_time_min: float = Field(gt=0, le=1440)
    surface_treatment: SurfaceTreatment
    protocol_reference: str = ""


class SensitizationRun(ProvenancedModel):
    device_id: UUID
    pigment_extract_batch_id: UUID
    adsorption_time_min: float = Field(gt=0, le=10080)
    sensitization_concentration_mm: float = Field(ge=0)
    sensitization_temperature_c: float = Field(ge=-20, le=200)
    treatment_notes: str = ""


class Device(ProvenancedModel):
    experimental_condition_id: UUID
    photoanode_fabrication_batch_id: UUID
    device_code: str = Field(min_length=1)
    replicate_number: int = Field(gt=0)
    active_area_cm2: float = Field(gt=0)
    electrolyte_type: ElectrolyteType
    counter_electrode_type: CounterElectrodeType
    device_batch_code: str = ""
