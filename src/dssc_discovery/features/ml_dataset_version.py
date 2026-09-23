"""MLDatasetVersion's domain model, split out from versioning.py to keep
the dependency direction one-way: database/repositories/ml_dataset_version.py
imports this module for the domain_cls it needs, and versioning.py imports
both this module AND the repository for its create/hash logic. Defining
the model inside versioning.py itself would make the repository and
versioning.py import each other -- a cycle.
"""

from __future__ import annotations

from dssc_discovery.domain.base import ProvenancedModel


class MLDatasetVersion(ProvenancedModel):
    """An immutable snapshot record of one build_ml_dataset() call
    (Chunk 3.2) -- not the data itself, only enough metadata (row IDs +
    a content hash + the code version that assembled it) to detect
    drift and recompute the same dataset reproducibly (PLAN.txt Section M).
    """

    dataset_hash: str
    query_definition: str
    code_version: str
    is_dirty: bool
    row_ids_json: str
    row_count: int
    feature_columns_json: str
