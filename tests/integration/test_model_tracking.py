"""Integration tests for MLflow tracking + ModelRun persistence (Chunk
4.7): a harness run produces both an MLflow entry and a DB row with
matching run_id; the row joins to the dataset version it cites; and
re-running with the recorded configuration reproduces the recorded
metrics within REPRODUCIBILITY_TOLERANCE."""

from __future__ import annotations

import json

import mlflow
import pytest

from dssc_discovery.database.repositories import MLDatasetVersionRepository, ModelRunRepository
from dssc_discovery.evaluation.cv_harness import (
    ModelSpec,
    default_feature_specs,
    run_grouped_nested_cv,
)
from dssc_discovery.features.dataset_builder import build_ml_dataset
from dssc_discovery.features.synthetic import generate_synthetic_dataset
from dssc_discovery.features.versioning import create_dataset_version
from dssc_discovery.models.baselines import MeanBaseline
from dssc_discovery.models.linear_models import RidgeModel
from dssc_discovery.models.tracking import (
    compute_feature_set_version_id,
    log_cv_result,
    metrics_match,
    rerun_model_run,
)

FEATURES = ["ph", "pigment_concentration_mm", "adsorption_time_min", "tio2_thickness_um"]


def _models():
    return [
        ModelSpec("mean_baseline", MeanBaseline(), "baseline"),
        ModelSpec(
            "ridge",
            RidgeModel(feature_specs=default_feature_specs(FEATURES)),
            "linear_or_gp",
            {"alpha": [0.1, 1.0]},
        ),
    ]


@pytest.fixture
def logged_run(session, tmp_path):
    generate_synthetic_dataset(session, n=20, seed=1)
    session.commit()
    version = create_dataset_version(session, operator="tester")
    session.commit()
    data = build_ml_dataset(session)
    models = _models()
    result = run_grouped_nested_cv(
        data, models, n_repeats=2, n_outer_splits=4, n_inner_splits=3, n_bootstrap=50, seed=7
    )
    tracking_uri = str(tmp_path / "mlruns")
    runs = log_cv_result(session, result, models, version, "tester", tracking_uri=tracking_uri)
    session.commit()
    return session, version, data, models, result, runs, tracking_uri


def test_each_model_gets_an_mlflow_run_and_a_matching_db_row(logged_run):
    session, version, _, _, result, runs, _ = logged_run

    assert [run.model_name for run in runs] == ["mean_baseline", "ridge"]
    # log_cv_result leaves MLflow pointed at the tracking store it wrote to
    for run in runs:
        stored = ModelRunRepository(session).get(run.id)
        assert stored is not None
        assert stored.mlflow_run_id == run.mlflow_run_id

        mlflow_run = mlflow.get_run(run.mlflow_run_id)
        assert mlflow_run.data.tags["dataset_version_id"] == str(version.id)
        assert mlflow_run.data.tags["feature_set_version_id"] == run.feature_set_version_id
        assert mlflow_run.data.tags["git_commit_hash"] == run.git_commit_hash
        assert mlflow_run.data.metrics["mae"] == pytest.approx(
            float(result.summary.set_index("model").loc[run.model_name, "mae"])
        )
        artifacts = {a.path for a in mlflow.MlflowClient().list_artifacts(run.mlflow_run_id)}
        assert {"predictions.csv", "fold_log.csv", "repeat_metrics.csv"} <= artifacts


def test_model_run_joins_to_its_dataset_version(logged_run):
    session, version, _, _, _, runs, _ = logged_run

    run = ModelRunRepository(session).get(runs[1].id)
    cited = MLDatasetVersionRepository(session).get(run.dataset_version_id)

    assert cited is not None
    assert cited.dataset_hash == version.dataset_hash
    assert run.random_seed == 7
    assert run.is_synthetic is True
    assert json.loads(run.feature_names_json) == FEATURES
    assert run.feature_set_version_id == compute_feature_set_version_id(FEATURES)


def test_rerunning_the_recorded_config_reproduces_the_metrics(logged_run):
    _, _, data, models, _, runs, _ = logged_run

    for run, spec in zip(runs, models, strict=True):
        reproduced = rerun_model_run(run, data, spec)
        assert metrics_match(json.loads(run.metrics_json), reproduced)


def test_rerun_refuses_a_spec_that_differs_from_the_recorded_one(logged_run):
    _, _, data, _, _, runs, _ = logged_run
    different = ModelSpec(
        "ridge",
        RidgeModel(feature_specs=default_feature_specs(FEATURES), alpha=5.0),
        "linear_or_gp",
        {"alpha": [0.1, 1.0]},
    )
    with pytest.raises(ValueError, match="hyperparameters"):
        rerun_model_run(runs[1], data, different)


def test_refuses_to_cite_a_dataset_version_of_a_different_size(session, tmp_path):
    generate_synthetic_dataset(session, n=12, seed=2)
    session.commit()
    version = create_dataset_version(session, operator="tester")
    data = build_ml_dataset(session).iloc[:-2]
    models = _models()
    result = run_grouped_nested_cv(data, models, n_repeats=1, n_outer_splits=3, n_bootstrap=10)

    with pytest.raises(ValueError, match="different dataset"):
        log_cv_result(session, result, models, version, "tester", tracking_uri=str(tmp_path))


def test_metrics_match_detects_a_real_difference_but_ignores_relative_flags():
    recorded = {"mae": 1.0, "rmse": 1.2, "family": "tree", "beats_all_baselines": True}

    assert metrics_match(recorded, {"mae": 1.0, "rmse": 1.2, "beats_all_baselines": None})
    assert not metrics_match(recorded, {"mae": 1.0 + 1e-6, "rmse": 1.2})
    assert not metrics_match(recorded, {"rmse": 1.2})
