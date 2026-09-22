"""Shared base for domain models: identity + provenance fields every
entity carries, regardless of what it represents.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from dssc_discovery.domain.enums import MissingValueReason


class ProvenancedModel(BaseModel):
    """Base for every domain entity.

    `id` is a client-generated UUID, not a DB-assigned sequence, so domain
    objects are self-identifying before they're ever persisted -- required
    for synthetic data generation, ingestion staging, and tests that never
    touch a database.
    """

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    operator: str = Field(min_length=1)
    is_synthetic: bool = False
    experiment_version: str = "v1"
    notes: str = ""


class MeasurementProvenancedModel(ProvenancedModel):
    """Adds fields that only make sense for an actual instrument
    measurement, not a procedural/preparation record (PlantSample,
    PigmentExtractBatch, ...). Numeric result fields on subclasses should
    stay Optional with missing_value_reason explaining absence, rather
    than silently defaulting to 0 -- e.g. a non-convergent EIS fit.
    """

    instrument: str = Field(min_length=1)
    raw_data_reference: str | None = None
    measurement_uncertainty: float | None = Field(default=None, ge=0)
    missing_value_reason: MissingValueReason | None = None
