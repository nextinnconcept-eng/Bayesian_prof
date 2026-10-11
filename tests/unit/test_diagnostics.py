"""Tests for residual analysis and learning curves (Chunk 4.6): every
diagnostic runs without error on the synthetic ground truth (3.5), and
the flag test -- an injected heteroscedastic residual pattern is
detected (and plotted), while homoscedastic noise is not flagged."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from dssc_discovery.evaluation.cv_harness import (
    ModelSpec,
    default_feature_specs,
    run_grouped_nested_cv,
)
from dssc_discovery.evaluation.diagnostics import (
    diagnostic_report,
    heteroscedasticity_test,
    learning_curve_frame,
    overfitting_gap,
    plot_residuals_vs_feature,
    plot_residuals_vs_predicted,
    residual_frame,
)
from dssc_discovery.features.synthetic import _sample_vector, synthetic_pce
from dssc_discovery.models.linear_models import RidgeModel
from dssc_discovery.models.tree_models import RandomForestModel

TARGET = "power_conversion_efficiency_percent"
FEATURES = ["ph", "pigment_concentration_mm", "adsorption_time_min", "tio2_thickness_um"]
FEATURE_SPECS = default_feature_specs(FEATURES)


def _synthetic_frame(n=40, seed=3):
    rng = np.random.default_rng(seed)
    vectors = [_sample_vector(rng) for _ in range(n)]
    frame = pd.DataFrame({name: [getattr(v, name) for v in vectors] for name in FEATURES})
    frame["surface_treatment"] = [v.surface_treatment.value for v in vectors]
    frame["group_id"] = [f"cond-{i}" for i in range(n)]
    frame[TARGET] = [synthetic_pce(v) + rng.normal(0, 0.3) for v in vectors]
    return frame


@pytest.fixture(scope="module")
def cv_run():
    data = _synthetic_frame()
    spec = ModelSpec("rf", RandomForestModel(feature_specs=FEATURE_SPECS, n_estimators=50), "tree")
    result = run_grouped_nested_cv(data, [spec], n_repeats=2, n_outer_splits=5, n_bootstrap=50)
    return data, spec, result


def _residuals_with(noise_scale, seed=0, n=200):
    rng = np.random.default_rng(seed)
    y_pred = rng.uniform(1.0, 9.0, n)
    residual = rng.normal(0, 1, n) * noise_scale(y_pred)
    return pd.DataFrame({"y_pred": y_pred, "residual": residual, "y_true": y_pred - residual})


def test_residual_frame_has_one_row_per_evaluated_row(cv_run):
    data, spec, result = cv_run
    residuals = residual_frame(result, data, "rf")

    assert len(residuals) == len(data)
    np.testing.assert_allclose(residuals["residual"], residuals["y_pred"] - residuals["y_true"])
    assert set(FEATURES) <= set(residuals.columns)


def test_residual_frame_rejects_unknown_model(cv_run):
    data, _, result = cv_run
    with pytest.raises(ValueError, match="not found"):
        residual_frame(result, data, "nope")


def test_injected_heteroscedastic_pattern_is_detected_and_plotted():
    residuals = _residuals_with(lambda y_pred: 0.1 * y_pred**1.5)

    flagged = heteroscedasticity_test(residuals)
    assert flagged.detected
    assert flagged.spearman_rho > 0.3

    fig = plot_residuals_vs_predicted(residuals)
    xs, ys = fig.axes[0].collections[0].get_offsets().T
    np.testing.assert_allclose(xs, residuals["y_pred"])
    np.testing.assert_allclose(ys, residuals["residual"])
    plt.close(fig)


def test_homoscedastic_noise_is_not_flagged():
    residuals = _residuals_with(lambda y_pred: np.full_like(y_pred, 0.5))
    assert not heteroscedasticity_test(residuals).detected


def test_overfitting_gap_shows_in_sample_error_below_cv_error(cv_run):
    data, spec, result = cv_run
    gap = overfitting_gap(result, data, spec)

    assert gap.train_mae < gap.cv_mae
    assert gap.gap == pytest.approx(gap.cv_mae - gap.train_mae)


def test_learning_curve_grows_training_conditions(cv_run):
    data, spec, _ = cv_run
    ridge = ModelSpec("ridge", RidgeModel(feature_specs=FEATURE_SPECS), "linear_or_gp")

    curve = learning_curve_frame(data, ridge, train_fractions=(0.4, 1.0), n_splits=4)

    assert list(curve["train_fraction"]) == [0.4, 1.0]
    assert curve["n_train_groups"].is_monotonic_increasing
    assert curve[["train_mae", "validation_mae"]].notna().all().all()


def test_categorical_feature_residual_plot_runs(cv_run):
    data, _, result = cv_run
    fig = plot_residuals_vs_feature(residual_frame(result, data, "rf"), "surface_treatment")
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_diagnostic_report_bundles_everything(cv_run):
    data, spec, result = cv_run
    report = diagnostic_report(result, data, spec)

    assert report.model == "rf"
    assert {"residuals_vs_predicted", "learning_curve"} <= set(report.figures)
    assert {f"residuals_vs_{f}" for f in FEATURES} <= set(report.figures)
    assert report.learning_curve is not None
    assert report.overfitting.cv_mae > 0
    for fig in report.figures.values():
        plt.close(fig)
