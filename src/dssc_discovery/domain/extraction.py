"""Domain models for plant sample collection and pigment extraction
(PLAN.txt Section E: PlantSample, PigmentExtractBatch).

Numeric field names match the keys registered in
dssc_discovery.domain.units_registry.CANONICAL_UNITS, so ingestion can
call to_canonical(value, unit, field=<this field's name>) directly.
"""

from __future__ import annotations

from datetime import date
from uuid import UUID

from pydantic import Field, field_validator

from dssc_discovery.domain.base import ProvenancedModel
from dssc_discovery.domain.enums import PigmentClass, PlantPart, SolventType, SpeciesCode


class PlantSample(ProvenancedModel):
    species_code: SpeciesCode
    plant_part: PlantPart
    pigment_class: PigmentClass
    batch_code: str = Field(min_length=1)
    collection_date: date
    collection_location: str = Field(min_length=1)
    authentication_method: str = ""
    voucher_id: str = ""

    @field_validator("collection_date")
    @classmethod
    def _collection_date_not_in_future(cls, value: date) -> date:
        if value > date.today():
            raise ValueError(f"collection_date {value} is in the future")
        return value


class PigmentExtractBatch(ProvenancedModel):
    plant_sample_id: UUID
    batch_code: str = Field(min_length=1)
    solvent: SolventType
    solvent_composition: str = ""  # free text, e.g. "ethanol/water/HCl"
    solvent_ratio: str = ""  # free text, e.g. "70:30:1"
    ph: float = Field(ge=0, le=14)
    extraction_temperature_c: float = Field(ge=-20, le=200)
    extraction_time_min: float = Field(gt=0)
    pigment_concentration_mm: float | None = Field(default=None, ge=0)
    extraction_yield_percent: float | None = Field(default=None, ge=0, le=100)
    protocol_reference: str = ""
