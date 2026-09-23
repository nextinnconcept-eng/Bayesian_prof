"""Tests for the tree-ensemble model wrappers (Chunk 4.2):
RandomForestModel, GradientBoostingModel, XGBoostModel. Same
sklearn-API-conformance and determinism contract as the baselines
(4.1), plus: hyperparameter pass-through (a changed constructor
hyperparameter must actually reach the fitted underlying estimator,
not be silently ignored) and a genuine, non-tautological
recoverability check (fit on a known, near-noise-free relationship in
two of the selected columns and confirm the model actually learns it,
while a third, unrelated column in the same frame has no effect).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from dssc_discovery.features.registry import get_feature
from dssc_discovery.models.tree_models import GradientBoostingModel, RandomForestModel, XGBoostModel

MODEL_CLASSES = [RandomForestModel, GradientBoostingModel, XGBoostModel]
FEATURE_SPECS = [get_feature("ph"), get_feature("pigment_concentration_mm"), get_feature("solvent")]


def _toy_frame(n=60, seed=1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "ph": rng.uniform(2, 9, n),
            "pigment_concentration_mm": rng.uniform(1, 30, n),
            "solvent": rng.choice(["ethanol", "methanol", "water"], n),
            "unrelated_noise_column": rng.uniform(-1000, 1000, n),
        }
    )
    y = 3.0 * df["ph"] - 0.5 * df["pigment_concentration_mm"] + rng.normal(0, 0.05, n)
    return df, y


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_raises_not_fitted_before_fit(model_cls):
    with pytest.raises(NotFittedError):
        model_cls(feature_specs=FEATURE_SPECS).predict(_toy_frame()[0])


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_fit_returns_self_and_predict_gives_right_length(model_cls):
    X, y = _toy_frame()
    model = model_cls(feature_specs=FEATURE_SPECS, random_state=0)
    fitted = model.fit(X, y)
    assert fitted is model

    predictions = model.predict(X)
    assert len(predictions) == len(X)
    assert not np.isnan(predictions).any()


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_is_deterministic_given_the_same_seed(model_cls):
    X, y = _toy_frame()
    first = model_cls(feature_specs=FEATURE_SPECS, random_state=42).fit(X, y).predict(X)
    second = model_cls(feature_specs=FEATURE_SPECS, random_state=42).fit(X, y).predict(X)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_conforms_to_the_sklearn_api(model_cls):
    model = model_cls(feature_specs=FEATURE_SPECS, n_estimators=17, random_state=7)
    assert model.get_params()["n_estimators"] == 17
    assert model.get_params()["random_state"] == 7

    model.set_params(n_estimators=5)
    assert model.n_estimators == 5

    X, y = _toy_frame()
    fitted = model.fit(X, y)
    assert fitted is model

    cloned = clone(model)
    assert cloned.get_params() == model.get_params()
    assert not hasattr(cloned, "pipeline_")


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_hyperparameter_pass_through_reaches_the_fitted_estimator(model_cls):
    X, y = _toy_frame()
    model = model_cls(feature_specs=FEATURE_SPECS, n_estimators=13, random_state=0).fit(X, y)

    fitted_estimator = model.pipeline_.named_steps["model"]
    assert fitted_estimator.n_estimators == 13


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_ignores_columns_outside_its_feature_specs(model_cls):
    X, y = _toy_frame()
    model_a = model_cls(feature_specs=FEATURE_SPECS, random_state=0).fit(X, y)

    X_shuffled_noise = X.copy()
    X_shuffled_noise["unrelated_noise_column"] = X["unrelated_noise_column"].to_numpy()[::-1]
    model_b = model_cls(feature_specs=FEATURE_SPECS, random_state=0).fit(X_shuffled_noise, y)

    np.testing.assert_allclose(model_a.predict(X), model_b.predict(X_shuffled_noise))


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_recovers_a_learnable_relationship_on_training_data(model_cls):
    X, y = _toy_frame(n=80)
    model = model_cls(feature_specs=FEATURE_SPECS, n_estimators=150, random_state=0).fit(X, y)
    predictions = model.predict(X)

    ss_res = float(np.sum((y - predictions) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1 - ss_res / ss_tot
    assert r_squared > 0.8  # tree ensembles should fit training data closely at this N
