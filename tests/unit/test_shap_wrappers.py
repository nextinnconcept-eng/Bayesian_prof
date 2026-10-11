"""SHAP wrappers and plots (Chunks 5.1-5.3): additivity identity
(base value + SHAP row sum == model prediction), caveat on every output,
and plot helpers returning Figures that carry the caveat."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from dssc_discovery.evaluation.cv_harness import default_feature_specs, model_features
from dssc_discovery.explainability.base import INTERACTION_CAVEAT, INTERPRETATION_CAVEAT
from dssc_discovery.explainability.plots import (
    plot_beeswarm,
    plot_dependence,
    plot_global_bar,
    plot_waterfall,
)
from dssc_discovery.explainability.shap_kernel import explain, explain_with_kernel
from dssc_discovery.explainability.shap_tree import explain_tree_model
from dssc_discovery.models.baselines import MeanBaseline
from dssc_discovery.models.gpr import GPRModel
from dssc_discovery.models.linear_models import RidgeModel
from dssc_discovery.models.tree_models import GradientBoostingModel, RandomForestModel, XGBoostModel
from tests.bo_helpers import GROUND_TRUTH_FEATURES, TARGET, ground_truth_frame

SPECS = default_feature_specs([*GROUND_TRUTH_FEATURES, "surface_treatment"])


@pytest.fixture(scope="module")
def data():
    frame = ground_truth_frame(40)
    return model_features(frame), frame[TARGET]


@pytest.mark.parametrize("model_cls", [RandomForestModel, GradientBoostingModel, XGBoostModel])
def test_tree_shap_values_sum_to_prediction_minus_base(data, model_cls):
    X, y = data
    model = model_cls(feature_specs=SPECS, n_estimators=30, random_state=0).fit(X, y)
    explanation = explain_tree_model(model, X)

    np.testing.assert_allclose(explanation.predictions(), model.predict(X), atol=1e-3)
    assert explanation.values.shape == (len(X), len(explanation.feature_names))
    assert explanation.interpretation_caveat == INTERPRETATION_CAVEAT


def test_tree_interaction_values_are_flagged_low_confidence(data):
    X, y = data
    model = RandomForestModel(feature_specs=SPECS, n_estimators=10, random_state=0).fit(X, y)
    explanation = explain_tree_model(model, X.iloc[:5], with_interactions=True)

    n_features = len(explanation.feature_names)
    assert explanation.interaction_values.shape == (5, n_features, n_features)
    assert explanation.interaction_caveat == INTERACTION_CAVEAT


@pytest.mark.parametrize("model_cls", [GPRModel, RidgeModel])
def test_kernel_shap_values_sum_to_prediction_minus_base(data, model_cls):
    X, y = data
    model = model_cls(feature_specs=SPECS, random_state=0).fit(X, y)
    explanation = explain_with_kernel(model, X.iloc[:8], background=X, background_size=20, seed=1)

    # Kernel SHAP enforces efficiency, so additivity holds up to solver error.
    np.testing.assert_allclose(explanation.predictions(), model.predict(X.iloc[:8]), atol=1e-3)
    assert explanation.explainer == "kernel"


def test_kernel_shap_is_reproducible_given_seed(data):
    X, y = data
    model = GPRModel(feature_specs=SPECS, random_state=0).fit(X, y)
    first = explain_with_kernel(model, X.iloc[:4], background_size=10, nsamples=60, seed=5)
    second = explain_with_kernel(model, X.iloc[:4], background_size=10, nsamples=60, seed=5)
    np.testing.assert_allclose(first.values, second.values)


def test_explain_dispatches_by_family_and_rejects_others(data):
    X, y = data
    tree = RandomForestModel(feature_specs=SPECS, n_estimators=5, random_state=0).fit(X, y)
    ridge = RidgeModel(feature_specs=SPECS).fit(X, y)
    assert explain(tree, X.iloc[:3], seed=0).explainer == "tree"
    assert explain(ridge, X.iloc[:3], background_size=10, seed=0).explainer == "kernel"
    with pytest.raises(TypeError):
        explain(MeanBaseline().fit(X, y), X)


def test_every_record_and_importance_row_carries_the_caveat(data):
    X, y = data
    model = RandomForestModel(feature_specs=SPECS, n_estimators=5, random_state=0).fit(X, y)
    explanation = explain_tree_model(model, X.iloc[:3])

    records = explanation.to_records()
    assert len(records) == 3 * len(explanation.feature_names)
    assert all(r.interpretation_caveat == INTERPRETATION_CAVEAT for r in records)
    assert (explanation.global_importance()["interpretation_caveat"] == INTERPRETATION_CAVEAT).all()


def test_plot_helpers_return_figures_stamped_with_the_caveat(data):
    X, y = data
    model = RandomForestModel(feature_specs=SPECS, n_estimators=10, random_state=0).fit(X, y)
    explanation = explain_tree_model(model, X)
    figures = [
        plot_global_bar(explanation),
        plot_beeswarm(explanation),
        plot_waterfall(explanation, row=0),
        plot_dependence(
            explanation, explanation.feature_names[0], color_by=explanation.feature_names[1]
        ),
    ]
    for fig in figures:
        assert isinstance(fig, Figure)
        assert any("not causal" in t.get_text() for t in fig.texts)
        plt.close(fig)
