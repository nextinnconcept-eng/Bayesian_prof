"""Domain models for the DoE/BO "recipe" unit: the ExperimentalCondition
BO optimizes over, and its parameter vector (PLAN.txt Section G/E).

ExperimentalConditionVector is a pure value object (no identity/provenance
of its own) -- it's what the BO surrogate reads as a training point and
writes as a recommendation candidate, before any ExperimentalCondition
entity necessarily exists.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from dssc_discovery.domain.base import ProvenancedModel
from dssc_discovery.domain.enums import (
    ClosedLoopState,
    DesignSource,
    SolventType,
    SpeciesCode,
    SurfaceTreatment,
)


class ExperimentalConditionVector(BaseModel):
    """The BO-facing parameter vector.

    Field names match dssc_discovery.domain.units_registry.CANONICAL_UNITS
    keys wherever a canonical unit applies, so BO/ingestion code never
    needs a translation layer between "the vector" and the units registry.
    `ph` and `solvent_ratio` are the two exceptions: pH isn't a Pint unit
    (Chunk 0.5), and solvent_ratio is a dimensionless fraction.

    Bounds here are broad physical-plausibility checks (same spirit as
    PigmentExtractBatch's), not the tighter per-species operational search
    space -- that's SearchSpace (Chunk 6.1).
    """

    model_config = ConfigDict(extra="forbid")

    # Extraction
    solvent: SolventType
    solvent_ratio: float = Field(ge=0, le=1, description="Primary solvent volume fraction")
    ph: float = Field(ge=0, le=14)
    extraction_temperature_c: float = Field(ge=-20, le=200)
    extraction_time_min: float = Field(gt=0, le=1440)
    pigment_concentration_mm: float = Field(ge=0)

    # Sensitization
    adsorption_time_min: float = Field(gt=0, le=10080)
    sensitization_concentration_mm: float = Field(ge=0)
    sensitization_temperature_c: float = Field(ge=-20, le=200)

    # Photoanode
    tio2_thickness_um: float = Field(gt=0, le=100)
    sintering_temperature_c: float = Field(ge=0, le=600)
    sintering_time_min: float = Field(gt=0, le=1440)
    surface_treatment: SurfaceTreatment


class ExperimentalCondition(ProvenancedModel):
    species_code: SpeciesCode
    condition_code: str = Field(min_length=1)
    design_source: DesignSource
    state: ClosedLoopState = ClosedLoopState.PROPOSED
    target_parameters: ExperimentalConditionVector
