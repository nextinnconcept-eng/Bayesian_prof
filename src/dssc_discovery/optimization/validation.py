"""Hard rejection of infeasible recommendations (Chunk 6.6). Every
candidate is re-validated against its SearchSpace before it may become a
PROPOSED condition; violations are rejected with logged reasons, never
coerced into bounds.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from dssc_discovery.domain.experimental_condition import ExperimentalConditionVector
from dssc_discovery.logging_setup import get_logger
from dssc_discovery.optimization.search_space import SearchSpace

logger = get_logger(__name__)


class InfeasibleRecommendationError(ValueError):
    def __init__(self, values: Mapping[str, object], reasons: list[str]) -> None:
        self.values = dict(values)
        self.reasons = reasons
        super().__init__("infeasible recommendation rejected: " + "; ".join(reasons))


@dataclass(frozen=True)
class RejectedCandidate:
    values: dict[str, object]
    reasons: list[str]

    def as_dict(self) -> dict[str, object]:
        return {"values": self.values, "reasons": self.reasons}


def validate_candidate(
    space: SearchSpace, values: Mapping[str, object]
) -> ExperimentalConditionVector:
    """Return the validated vector, or raise InfeasibleRecommendationError
    (logged) listing every violation. Never modifies values."""
    reasons = space.violations(values)
    if not reasons:
        try:
            return space.to_vector(values)
        except ValueError as exc:  # ExperimentalConditionVector's own physical bounds
            reasons = [f"vector validation: {exc}"]
    logger.warning(
        "rejected infeasible candidate for %s: %s", space.species.value, "; ".join(reasons)
    )
    raise InfeasibleRecommendationError(values, reasons)
