"""Dataset version snapshotting: freezes one build_ml_dataset() (Chunk
3.2) call into an immutable MLDatasetVersion record (PLAN.txt Section M),
so any future ModelRun (Chunk 4.7) can cite exactly which data produced it.

Snapshots don't store the full data blob -- PLAN.txt's own roadmap is
explicit about this ("not necessarily the full data blob (recomputable
from IDs)") -- only enough metadata to detect drift and recompute the
same dataset: the row IDs actually included, the query/assembly logic's
name, and the code version (git commit) that produced them. The content
hash is a deterministic function of exactly those three things, so an
unchanged DB state at an unchanged commit always produces an identical
hash, and any change to either produces a different one.
"""

from __future__ import annotations

import hashlib
import json

from sqlalchemy.orm import Session

from dssc_discovery.config import get_git_commit_hash, is_git_working_tree_dirty
from dssc_discovery.database.repositories import MLDatasetVersionRepository
from dssc_discovery.features.dataset_builder import build_ml_dataset
from dssc_discovery.features.ml_dataset_version import MLDatasetVersion

QUERY_DEFINITION = "build_ml_dataset"


def compute_dataset_hash(row_ids: list[str], query_definition: str, code_version: str) -> str:
    """SHA-256 of the sorted row IDs + query definition + code version.
    Row IDs are sorted before hashing so the hash reflects WHICH rows
    are included, never the order build_ml_dataset happened to return
    them in.
    """
    payload = json.dumps(
        {
            "row_ids": sorted(row_ids),
            "query_definition": query_definition,
            "code_version": code_version,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def create_dataset_version(session: Session, operator: str) -> MLDatasetVersion:
    """Assemble the current ML dataset and freeze it as an immutable
    MLDatasetVersion. is_synthetic is set True only if every row in the
    snapshot is synthetic -- a snapshot mixing real and synthetic rows,
    or containing only real rows, is not flagged synthetic.
    """
    dataframe = build_ml_dataset(session)
    row_ids = sorted(str(value) for value in dataframe["device_id"])
    code_version = get_git_commit_hash()
    is_dirty = is_git_working_tree_dirty()
    dataset_hash = compute_dataset_hash(row_ids, QUERY_DEFINITION, code_version)

    is_synthetic = bool(dataframe["is_synthetic"].all()) if not dataframe.empty else False

    version = MLDatasetVersion(
        dataset_hash=dataset_hash,
        query_definition=QUERY_DEFINITION,
        code_version=code_version,
        is_dirty=is_dirty,
        row_ids_json=json.dumps(row_ids),
        row_count=len(row_ids),
        feature_columns_json=json.dumps(sorted(dataframe.columns.tolist())),
        is_synthetic=is_synthetic,
        operator=operator,
    )
    return MLDatasetVersionRepository(session).add(version)
