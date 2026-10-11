"""HypothesisRecord domain model (Chunk 5.4), separate from
hypothesis_log.py so the repository can import it without a cycle."""

from __future__ import annotations

from uuid import UUID

from pydantic import Field

from dssc_discovery.domain.base import ProvenancedModel
from dssc_discovery.domain.enums import HypothesisStatus
from dssc_discovery.explainability.base import INTERPRETATION_CAVEAT


class HypothesisRecord(ProvenancedModel):
    """observation -> candidate_mechanism -> confound_check ->
    proposed_confirmatory_experiment -> status (Section I)."""

    observation: str = Field(min_length=1)
    candidate_mechanism: str = Field(min_length=1)
    confound_check: str = Field(min_length=1)
    proposed_confirmatory_experiment: str = Field(min_length=1)
    status: HypothesisStatus = HypothesisStatus.OPEN
    status_note: str = ""
    shap_features: str = ""  # comma-separated features the finding concerns
    model_run_id: UUID | None = None
    interpretation_caveat: str = INTERPRETATION_CAVEAT
