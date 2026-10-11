"""Closed-loop audit trail export (Chunk 8.6): everything needed to
reconstruct a closed-loop run for a methods section / supplementary
material -- conditions and their states, the full state-transition log,
BO iterations, recommendations with rationales, model runs and dataset
versions -- written as CSV tables plus a JSON manifest. Synthetic and
real runs are exported separately (real_only / synthetic_only) so they
can never be mixed in one supplementary file.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from sqlalchemy.orm import Session

from dssc_discovery.config import get_git_commit_hash
from dssc_discovery.database.repositories import (
    BOIterationRepository,
    ClosedLoopStateLogRepository,
    ExperimentalConditionRepository,
    MLDatasetVersionRepository,
    ModelRunRepository,
    RecommendationRepository,
)

Scope = Literal["real_only", "synthetic_only"]


def _frame(rows: list, flatten_vector: bool = False) -> pd.DataFrame:
    records = []
    for row in rows:
        record = row.model_dump(mode="json")
        if flatten_vector:
            record.update(record.pop("target_parameters"))
        records.append(record)
    return pd.DataFrame(records)


def export_audit_trail(session: Session, out_dir: Path, scope: Scope = "real_only") -> Path:
    want_synthetic = scope == "synthetic_only"
    out_dir.mkdir(parents=True, exist_ok=True)

    def keep(rows: list) -> list:
        return [r for r in rows if r.is_synthetic == want_synthetic]

    tables = {
        "experimental_conditions": _frame(
            keep(ExperimentalConditionRepository(session).list()), flatten_vector=True
        ),
        "closed_loop_state_log": _frame(keep(ClosedLoopStateLogRepository(session).list())),
        "bo_iterations": _frame(keep(BOIterationRepository(session).list())),
        "recommendations": _frame(keep(RecommendationRepository(session).list())),
        "model_runs": _frame(keep(ModelRunRepository(session).list())),
        "ml_dataset_versions": _frame(keep(MLDatasetVersionRepository(session).list())),
    }
    for name, frame in tables.items():
        if "created_at" in frame.columns:
            frame = frame.sort_values("created_at")
        frame.to_csv(out_dir / f"{name}.csv", index=False)

    manifest = {
        "scope": scope,
        "exported_at": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit_hash(),
        "row_counts": {name: len(frame) for name, frame in tables.items()},
        "join_path": "recommendations.bo_iteration_id -> bo_iterations.model_run_id -> "
        "model_runs.dataset_version_id -> ml_dataset_versions.id",
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path
