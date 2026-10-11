"""ModelRun's domain model (Chunk 4.7), kept separate from tracking.py
for the same reason MLDatasetVersion lives apart from versioning.py: the
repository imports this module for its domain_cls, and tracking.py
imports both -- defining the model in tracking.py would create a cycle.
"""

from __future__ import annotations

from uuid import UUID

from dssc_discovery.domain.base import ProvenancedModel


class ModelRun(ProvenancedModel):
    """One model's evaluation within one harness run: everything PLAN.txt
    Section M requires to reconstruct it -- dataset version, feature set,
    code version, seed, hyperparameters, CV configuration, metrics -- plus
    the MLflow run that holds its artifacts (predictions, fold log)."""

    dataset_version_id: UUID
    feature_set_version_id: str
    feature_names_json: str
    model_name: str
    model_family: str
    model_class: str
    strategy: str
    git_commit_hash: str
    is_dirty: bool
    random_seed: int
    hyperparameters_json: str
    cv_config_json: str
    metrics_json: str
    mlflow_run_id: str
    mlflow_experiment_name: str
