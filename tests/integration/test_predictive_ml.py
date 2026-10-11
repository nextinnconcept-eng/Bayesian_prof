"""M4 gate (Chunk 4.8): the full predictive-ML pipeline on the synthetic
dataset (3.5) -- DB -> build_ml_dataset -> dataset version -> every M4
model + both baselines under strategies A, B and C -> comparison table
-> MLflow/ModelRun logging -- asserting that real models beat both
baselines on the known ground-truth function and that the leakage canary
behaves as expected.

On synthetic data this only proves the pipeline works. Scientifically
interpretable results need a real dataset of at least the initial DoE
size (~40 conditions); see PLAN.txt 4.8 DoD.
"""

from __future__ import annotations

import json

import pytest

from dssc_discovery.database.repositories import ModelRunRepository
from dssc_discovery.domain.enums import DesignSource, SpeciesCode
from dssc_discovery.evaluation.cross_species import (
    metrics_by_held_out_species,
    run_cross_species_cv,
)
from dssc_discovery.evaluation.cv_harness import (
    ModelSpec,
    default_feature_specs,
    default_model_specs,
    run_grouped_nested_cv,
    strategy_comparison_table,
)
from dssc_discovery.evaluation.diagnostics import diagnostic_report
from dssc_discovery.evaluation.temporal_split import run_forward_extrapolation_cv
from dssc_discovery.features.dataset_builder import build_ml_dataset
from dssc_discovery.features.synthetic import generate_synthetic_dataset
from dssc_discovery.features.versioning import create_dataset_version
from dssc_discovery.models.tracking import log_cv_result
from dssc_discovery.models.tree_models import RandomForestModel

SPECIES = (SpeciesCode.BETA_VULGARIS, SpeciesCode.THUNBERGIA_ERECTA)
CONDITIONS_PER_STAGE = {
    DesignSource.INITIAL_DOE: 12,
    DesignSource.BO_CYCLE_1: 4,
    DesignSource.BO_CYCLE_2: 4,
    DesignSource.BO_CYCLE_3: 6,
}
N_REPLICATES = 2
# The four inputs synthetic_pce() depends on, plus species.
FEATURES = [
    "species_code",
    "ph",
    "pigment_concentration_mm",
    "adsorption_time_min",
    "tio2_thickness_um",
]
CV_KWARGS = dict(n_repeats=1, n_inner_splits=3, n_bootstrap=200, seed=42)
NONLINEAR_MODELS = ("random_forest", "gradient_boosting", "xgboost", "gpr")


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    import sqlalchemy as sa
    from sqlalchemy.orm import Session

    from dssc_discovery.database.models import Base

    engine = sa.create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)

    seed = 100
    for species in SPECIES:
        for stage, n_conditions in CONDITIONS_PER_STAGE.items():
            generate_synthetic_dataset(
                session,
                n=n_conditions,
                seed=seed,
                species_code=species,
                condition_code_prefix=f"{species.name}-{stage.name}",
                design_source=stage,
                n_replicates=N_REPLICATES,
                condition_effect_std_percent=0.6,
            )
            seed += 1
    session.commit()
    version = create_dataset_version(session, operator="m4-gate")
    session.commit()
    data = build_ml_dataset(session)

    specs = default_feature_specs(FEATURES)
    models = default_model_specs(specs, specs)
    results = {
        "A": run_grouped_nested_cv(data, models, n_outer_splits=5, **CV_KWARGS),
        "B": run_forward_extrapolation_cv(data, models, **CV_KWARGS),
        "C": run_cross_species_cv(data, models, **CV_KWARGS),
    }
    yield session, version, data, models, results, tmp_path_factory.mktemp("mlruns")
    session.close()
    engine.dispose()


def test_dataset_has_the_intended_structure(pipeline):
    _, version, data, _, _, _ = pipeline
    n_conditions = len(SPECIES) * sum(CONDITIONS_PER_STAGE.values())

    assert len(data) == version.row_count == n_conditions * N_REPLICATES
    assert data["group_id"].nunique() == n_conditions
    assert set(data["species_code"]) == {s.value for s in SPECIES}
    assert set(data["design_source"]) == {s.value for s in CONDITIONS_PER_STAGE}
    assert version.is_synthetic


@pytest.mark.parametrize("strategy", ["A", "B", "C"])
def test_every_model_and_baseline_is_evaluated_under_every_strategy(pipeline, strategy):
    _, _, _, models, results, _ = pipeline
    summary = results[strategy].summary

    assert list(summary["model"]) == [m.name for m in models]
    assert summary[["mae", "rmse", "mae_ci_low", "mae_ci_high"]].notna().all().all()


def test_real_models_beat_both_baselines_on_the_ground_truth(pipeline):
    _, _, _, _, results, _ = pipeline
    summary = results["A"].summary.set_index("model")

    baseline_mae = summary.loc[["mean_baseline", "domain_linear_baseline"], "mae"].min()
    beaten_by = [m for m in NONLINEAR_MODELS if summary.loc[m, "mae"] < baseline_mae]
    # synthetic_pce is a sum of Gaussian bumps: every flexible model should
    # recover it well enough to clear a mean predictor; demand most do.
    assert len(beaten_by) >= 3, summary[["mae", "mae_ci_low", "mae_ci_high"]]
    assert summary.loc["gpr", "beats_all_baselines"] is True


def test_leakage_canary_on_the_synthetic_pipeline(pipeline):
    _, _, data, _, _, _ = pipeline
    rf = [ModelSpec("rf", RandomForestModel(feature_specs=default_feature_specs(FEATURES)), "tree")]
    kwargs = dict(n_outer_splits=5, n_repeats=1, n_bootstrap=50, seed=3)

    grouped = run_grouped_nested_cv(data, rf, grouped=True, **kwargs)
    ungrouped = run_grouped_nested_cv(data, rf, grouped=False, **kwargs)

    grouped_mae = float(grouped.summary["mae"].iloc[0])
    ungrouped_mae = float(ungrouped.summary["mae"].iloc[0])
    assert ungrouped_mae < 0.85 * grouped_mae


def test_comparison_table_reports_strategies_side_by_side(pipeline, tmp_path):
    _, _, data, models, results, _ = pipeline
    table = strategy_comparison_table([results["A"], results["B"], results["C"]])

    assert list(table.columns[:4]) == ["strategy", "model", "family", "n_rows_evaluated"]
    assert set(table["strategy"]) == {
        "A_grouped_kfold",
        "B_forward_extrapolation",
        "C_leave_one_species_out",
    }
    assert len(table) == 3 * len(models)
    b_rows = table.loc[table["strategy"] == "B_forward_extrapolation", "n_rows_evaluated"]
    assert (b_rows == len(SPECIES) * CONDITIONS_PER_STAGE[DesignSource.BO_CYCLE_3] * 2).all()
    table.to_csv(tmp_path / "m4_model_comparison.csv", index=False)

    per_species = metrics_by_held_out_species(results["C"], data)
    assert set(per_species["species"]) == {s.value for s in SPECIES}


def test_diagnostics_and_tracking_run_for_the_gate_results(pipeline):
    session, version, data, models, results, mlruns = pipeline
    gpr = next(m for m in models if m.name == "gpr")

    report = diagnostic_report(results["A"], data, gpr, include_learning_curve=False)
    assert report.overfitting.cv_mae > 0
    for fig in report.figures.values():
        import matplotlib.pyplot as plt

        plt.close(fig)

    runs = log_cv_result(
        session, results["A"], models, version, "m4-gate", tracking_uri=str(mlruns)
    )
    session.commit()
    assert len(runs) == len(models)
    stored = ModelRunRepository(session).get(runs[-1].id)
    assert stored is not None
    assert json.loads(stored.cv_config_json)["strategy"] == "A_grouped_kfold"
