"""Species reference entity -- one per taxon in this research programme's
scope (PLAN.txt Section 1).
"""

from __future__ import annotations

from dssc_discovery.domain.base import ProvenancedModel
from dssc_discovery.domain.enums import SpeciesCode


class Species(ProvenancedModel):
    code: SpeciesCode
    common_name: str = ""
    family: str = ""
