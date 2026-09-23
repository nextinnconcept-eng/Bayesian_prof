"""Tests for the regularized linear model wrappers (Chunk 4.2):
RidgeModel, LassoModel. Same sklearn-API-conformance, determinism, and
hyperparameter-pass-through contract as the tree models, plus an exact
(not just "learnable") recovery check: with alpha small enough,
Ridge/Lasso fit to a noise-free linear target should reproduce it
almost exactly, the same standard this project already held
DomainLinearBaseline to (Chunk 4.1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from dssc_discovery.features.registry import get_feature
from dssc_discovery.models.linear_models import LassoModel, RidgeModel

MODEL_CLASSES = [RidgeModel, LassoModel]
FEATURE_SPECS = [get_feature("ph"), get_feature("pigment_concentration_mm"), get_feature("solvent")]


def _toy_frame(n=60, seed=1, noise_std=0.05):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "ph": rng.uniform(2, 9, n),
            "pigment_concentration_mm": rng.uniform(1, 30, n),
            "solvent": rng.choice(["ethanol", "methanol", "water"], n),
            "unrelated_noise_column": rng.uniform(-1000, 1000, n),
        }
    )
    y = 3.0 * df["ph"] - 0.5 * df["pigment_concentration_mm"] + rng.normal(0, noise_std, n)
    return df, y


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_raises_not_fitted_before_fit(model_cls):
    with pytest.raises(NotFittedError):
        model_cls(feature_specs=FEATURE_SPECS).predict(_toy_frame()[0])


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_fit_returns_self_and_predict_gives_right_length(model_cls):
    X, y = _toy_frame()
    model = model_cls(feature_specs=FEATURE_SPECS, alpha=0.01, random_state=0)
    fitted = model.fit(X, y)
    assert fitted is model

    predictions = model.predict(X)
    assert len(predictions) == len(X)
    assert not np.isnan(predictions).any()


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_is_deterministic_given_the_same_seed(model_cls):
    X, y = _toy_frame()
    first = model_cls(feature_specs=FEATURE_SPECS, alpha=0.01, random_state=42).fit(X, y).predict(X)
    second = (
        model_cls(feature_specs=FEATURE_SPECS, alpha=0.01, random_state=42).fit(X, y).predict(X)
    )
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_conforms_to_the_sklearn_api(model_cls):
    model = model_cls(feature_specs=FEATURE_SPECS, alpha=2.5, random_state=7)
    assert model.get_params()["alpha"] == 2.5
    assert model.get_params()["random_state"] == 7

    model.set_params(alpha=0.5)
    assert model.alpha == 0.5

    X, y = _toy_frame()
    fitted = model.fit(X, y)
    assert fitted is model

    cloned = clone(model)
    assert cloned.get_params() == model.get_params()
    assert not hasattr(cloned, "pipeline_")


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_hyperparameter_pass_through_reaches_the_fitted_estimator(model_cls):
    X, y = _toy_frame()
    model = model_cls(feature_specs=FEATURE_SPECS, alpha=3.3, random_state=0).fit(X, y)

    fitted_estimator = model.pipeline_.named_steps["model"]
    assert fitted_estimator.alpha == pytest.approx(3.3)


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_ignores_columns_outside_its_feature_specs(model_cls):
    X, y = _toy_frame()
    model_a = model_cls(feature_specs=FEATURE_SPECS, alpha=0.01, random_state=0).fit(X, y)

    X_shuffled_noise = X.copy()
    X_shuffled_noise["unrelated_noise_column"] = X["unrelated_noise_column"].to_numpy()[::-1]
    model_b = model_cls(feature_specs=FEATURE_SPECS, alpha=0.01, random_state=0).fit(
        X_shuffled_noise, y
    )

    np.testing.assert_allclose(model_a.predict(X), model_b.predict(X_shuffled_noise))


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_recovers_a_noise_free_linear_relationship_almost_exactly(model_cls):
    X, y = _toy_frame(n=80, noise_std=0.0)
    model = model_cls(feature_specs=FEATURE_SPECS, alpha=1e-4, random_state=0).fit(X, y)
    predictions = model.predict(X)

    np.testing.assert_allclose(predictions, y.to_numpy(), atol=0.15)
