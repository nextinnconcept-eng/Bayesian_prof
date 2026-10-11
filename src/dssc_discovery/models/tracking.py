"""MLflow tracking + ModelRun persistence (Chunk 4.7, PLAN.txt Section M).

Every model evaluated by a harness run (Chunks 4.4/4.5) gets:
- one MLflow run in the local store (config mlflow_tracking_uri; a plain
  directory becomes an SQLite store + artifacts folder inside it): params
  (hyperparameters, CV configuration), metrics (point estimates, CIs,
  repeat std), tags (dataset version, feature set, git commit, strategy)
  and artifacts (out-of-fold predictions, fold log with the inner search's
  chosen hyperparameters, per-repeat metrics);
- one ModelRun DB row citing dataset_version_id, feature_set_version_id,
  git_commit_hash, seed and the MLflow run_id.

So "what produced this prediction" is one join (model_runs ->
ml_dataset_versions), and the MLflow run_id on the row leads to the
artifacts.

feature_set_version_id: there is no FeatureSetVersion table yet, so it is
a content hash of the model's feature definitions (name, dtype, stage,
source) -- deterministic, order-independent, and it changes whenever the
feature set does. A future table can adopt these hashes as its keys.

Reproducibility (Section L): rerun_model_run() re-executes the recorded
strategy/configuration for one model and returns its metrics. Every
source of randomness in the harness derives from the recorded seed and
each model's metrics are computed independently of the others, so a
re-run on the same platform and library versions reproduces the recorded
metrics to within REPRODUCIBILITY_TOLERANCE (floating-point noise only).
Different BLAS builds or library versions can move GP/boosting fits
slightly further; compare with a looser tolerance there and say so.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from sqlalchemy.orm import Session

from dssc_discovery.config import (
    get_config,
    get_git_commit_hash,
    is_git_working_tree_dirty,
    resolve_path,
)
from dssc_discovery.database.repositories import ModelRunRepository
from dssc_discovery.domain.enums import DesignSource
from dssc_discovery.evaluation.cross_species import run_cross_species_cv
from dssc_discovery.evaluation.cv_harness import CVResult, ModelSpec, run_grouped_nested_cv
from dssc_discovery.evaluation.temporal_split import run_forward_extrapolation_cv
from dssc_discovery.features.ml_dataset_version import MLDatasetVersion
from dssc_discovery.features.registry import FEATURE_REGISTRY
from dssc_discovery.models.baselines import DOMAIN_FEATURE_COLUMNS, DomainLinearBaseline
from dssc_discovery.models.model_run import ModelRun

DEFAULT_EXPERIMENT_NAME = "dssc-predictive-ml"
REPRODUCIBILITY_TOLERANCE = 1e-9

_REGISTRY_BY_NAME = {spec.name: spec for spec in FEATURE_REGISTRY}


# --- identity helpers --------------------------------------------------------


def model_feature_names(spec: ModelSpec) -> list[str]:
    """The columns a model actually reads. Wrapped models declare
    feature_specs; DomainLinearBaseline reads its two fixed columns;
    MeanBaseline reads none."""
    if isinstance(spec.estimator, DomainLinearBaseline):
        return list(DOMAIN_FEATURE_COLUMNS)
    feature_specs = spec.estimator.get_params().get("feature_specs") or []
    return [f.name for f in feature_specs]


def compute_feature_set_version_id(feature_names: Sequence[str]) -> str:
    definitions = []
    for name in sorted(set(feature_names)):
        registered = _REGISTRY_BY_NAME.get(name)
        definitions.append(
            [
                name,
                registered.dtype,
                registered.stage,
                registered.source_table,
                registered.source_field,
            ]
            if registered
            else [name]
        )
    payload = json.dumps(definitions, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def model_hyperparameters(spec: ModelSpec) -> dict[str, Any]:
    params = {k: v for k, v in spec.estimator.get_params().items() if k != "feature_specs"}
    return {
        "base_params": params,
        "param_grid": {k: list(v) for k, v in spec.param_grid.items()},
    }


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _finite_metrics(row: pd.Series) -> dict[str, float]:
    metrics = {}
    for key, value in row.items():
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int | float) and math.isfinite(float(value)):
            metrics[str(key)] = float(value)
    return metrics


def _tracking_location(tracking_uri: str | None) -> tuple[str, str | None]:
    """(MLflow tracking URI, artifact location for new experiments).

    A full URI (sqlite:///..., http://...) is used as given. A plain path
    -- config's default "./mlruns" -- is treated as a local tracking
    DIRECTORY holding an SQLite store (mlflow.db) and an artifacts/
    folder: MLflow 3.x refuses the old plain-filesystem backend by
    default, and SQLite is its recommended local replacement.
    """
    uri = tracking_uri or get_config().mlflow_tracking_uri
    if "://" in uri:
        return uri, None
    root = resolve_path(Path(uri)).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(root / 'mlflow.db').as_posix()}", (root / "artifacts").as_uri()


# --- logging -----------------------------------------------------------------


def log_cv_result(
    session: Session,
    result: CVResult,
    models: Sequence[ModelSpec],
    dataset_version: MLDatasetVersion,
    operator: str,
    *,
    tracking_uri: str | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> list[ModelRun]:
    """Log one MLflow run and write one ModelRun row per model in result.

    models must be the ModelSpecs the result was produced with (their
    hyperparameters are recorded). dataset_version must be the snapshot
    the evaluated data came from; its row count is checked against the
    result as a guard against citing the wrong version. Caller must
    session.commit() (repositories only flush).
    """
    specs_by_name = {spec.name: spec for spec in models}
    missing = [name for name in result.config["models"] if name not in specs_by_name]
    if missing:
        raise ValueError(f"no ModelSpec supplied for model(s) {missing}")
    rows_used = result.config["n_rows"] + result.config.get("n_rows_dropped", 0)
    if rows_used != dataset_version.row_count:
        raise ValueError(
            f"result covers {rows_used} rows but dataset version {dataset_version.id} has "
            f"{dataset_version.row_count}; was it computed on a different dataset?"
        )

    uri, artifact_location = _tracking_location(tracking_uri)
    mlflow.set_tracking_uri(uri)
    if artifact_location and mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)

    git_commit = get_git_commit_hash()
    is_dirty = is_git_working_tree_dirty()
    summary = result.summary.set_index("model")
    repository = ModelRunRepository(session)
    runs = []

    for name in result.config["models"]:
        spec = specs_by_name[name]
        feature_names = model_feature_names(spec)
        feature_set_version_id = compute_feature_set_version_id(feature_names)
        hyperparameters = model_hyperparameters(spec)
        metrics = _finite_metrics(summary.loc[name])
        metrics_record = {k: (None if pd.isna(v) else v) for k, v in summary.loc[name].items()}
        tags = {
            "dataset_version_id": str(dataset_version.id),
            "dataset_hash": dataset_version.dataset_hash,
            "feature_set_version_id": feature_set_version_id,
            "git_commit_hash": git_commit,
            "git_dirty": str(is_dirty),
            "strategy": result.config["strategy"],
            "model_name": name,
            "model_family": spec.family,
            "is_synthetic": str(dataset_version.is_synthetic),
        }

        with mlflow.start_run(run_name=f"{result.config['strategy']}:{name}") as run:
            mlflow.set_tags(tags)
            mlflow.log_params(
                {
                    "model_class": type(spec.estimator).__name__,
                    "features": _json(feature_names),
                    "hyperparameters": _json(hyperparameters),
                    **{f"cv.{k}": _json(v) for k, v in result.config.items()},
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.log_text(
                result.predictions[result.predictions["model"] == name].to_csv(index=False),
                "predictions.csv",
            )
            mlflow.log_text(
                result.fold_log[result.fold_log["model"] == name].to_csv(index=False),
                "fold_log.csv",
            )
            mlflow.log_text(
                result.repeat_metrics[result.repeat_metrics["model"] == name].to_csv(index=False),
                "repeat_metrics.csv",
            )
            run_id = run.info.run_id

        runs.append(
            repository.add(
                ModelRun(
                    dataset_version_id=dataset_version.id,
                    feature_set_version_id=feature_set_version_id,
                    feature_names_json=_json(feature_names),
                    model_name=name,
                    model_family=spec.family,
                    model_class=type(spec.estimator).__name__,
                    strategy=result.config["strategy"],
                    git_commit_hash=git_commit,
                    is_dirty=is_dirty,
                    random_seed=int(result.config["seed"]),
                    hyperparameters_json=_json(hyperparameters),
                    cv_config_json=_json(result.config),
                    metrics_json=_json(metrics_record),
                    mlflow_run_id=run_id,
                    mlflow_experiment_name=experiment_name,
                    operator=operator,
                    is_synthetic=dataset_version.is_synthetic,
                )
            )
        )
    return runs


# --- reproduction ------------------------------------------------------------

_COMMON_KEYS = (
    "target",
    "group_column",
    "n_repeats",
    "n_inner_splits",
    "seed",
    "n_bootstrap",
    "confidence_level",
)


def rerun_model_run(model_run: ModelRun, data: pd.DataFrame, spec: ModelSpec) -> dict[str, Any]:
    """Re-execute model_run's recorded strategy and CV configuration for
    one model on data (rebuilt from the cited dataset version) and return
    its summary metrics. spec must match the recorded model: same name,
    class, hyperparameters and feature set -- otherwise this wouldn't be
    a reproduction, so it raises."""
    if spec.name != model_run.model_name:
        raise ValueError(f"spec {spec.name!r} does not match recorded {model_run.model_name!r}")
    if type(spec.estimator).__name__ != model_run.model_class:
        raise ValueError("spec's estimator class differs from the recorded run")
    if _json(model_hyperparameters(spec)) != model_run.hyperparameters_json:
        raise ValueError("spec's hyperparameters differ from the recorded run")
    if (
        compute_feature_set_version_id(model_feature_names(spec))
        != model_run.feature_set_version_id
    ):
        raise ValueError("spec's feature set differs from the recorded run")

    config = json.loads(model_run.cv_config_json)
    kwargs = {key: config[key] for key in _COMMON_KEYS}
    strategy = config["strategy"]
    if strategy.startswith("A_"):
        result = run_grouped_nested_cv(
            data,
            [spec],
            n_outer_splits=config["n_outer_splits"],
            grouped=config["grouped"],
            **kwargs,
        )
    elif strategy == "B_forward_extrapolation":
        result = run_forward_extrapolation_cv(
            data,
            [spec],
            test_stage=DesignSource(config["test_stage"]),
            manual_policy=config["manual_policy"],
            **kwargs,
        )
    elif strategy == "C_leave_one_species_out":
        result = run_cross_species_cv(
            data, [spec], species_column=config["species_column"], **kwargs
        )
    else:
        raise ValueError(f"unknown recorded strategy {strategy!r}")

    row = result.summary.set_index("model").loc[spec.name]
    return {k: (None if pd.isna(v) else v) for k, v in row.items()}


def metrics_match(
    recorded: dict[str, Any],
    reproduced: dict[str, Any],
    tolerance: float = REPRODUCIBILITY_TOLERANCE,
) -> bool:
    """True if every numeric metric recorded agrees within tolerance
    (relative, with an absolute floor of tolerance itself).

    Non-numeric summary fields are not compared: "family" is a label, and
    "beats_all_baselines" compares the model against the OTHER models of
    the original run, so a single-model re-run can't reproduce it -- it
    isn't a property of this model's own evaluation."""
    for key, value in recorded.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        other = reproduced.get(key)
        if other is None:
            if not math.isnan(float(value)):
                return False
            continue
        if math.isnan(float(value)) and math.isnan(float(other)):
            continue
        if not math.isclose(float(value), float(other), rel_tol=tolerance, abs_tol=tolerance):
            return False
    return True
