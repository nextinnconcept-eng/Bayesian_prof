"""Tests for the mandatory baseline predictors (Chunk 4.1): sklearn API
conformance (fit returns self, predict raises before fit, get_params/
set_params/clone round-trip -- all required for the evaluation harness,
4.4, to instantiate and clone these identically to "real" models),
determinism, and -- for DomainLinearBaseline -- that it genuinely
recovers a known linear relationship from exactly its two named
columns and ignores everything else in the frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from dssc_discovery.models.baselines import (
    DOMAIN_FEATURE_COLUMNS,
    DomainLinearBaseline,
    MeanBaseline,
)

# --- MeanBaseline --------------------------------------------------------------


def test_mean_baseline_predicts_the_training_mean_for_every_row():
    X = pd.DataFrame({"anything": [1, 2, 3]})
    y = pd.Series([2.0, 4.0, 6.0])

    model = MeanBaseline()
    model.fit(X, y)
    predictions = model.predict(pd.DataFrame({"anything": [10, 20]}))

    assert predictions.tolist() == [4.0, 4.0]


def test_mean_baseline_ignores_feature_values_entirely():
    y = pd.Series([2.0, 4.0, 6.0])
    model_a = MeanBaseline().fit(pd.DataFrame({"a": [1, 2, 3]}), y)
    model_b = MeanBaseline().fit(pd.DataFrame({"b": [-99, 0, 99]}), y)

    assert (
        model_a.predict(pd.DataFrame({"a": [1]})).tolist()
        == model_b.predict(pd.DataFrame({"b": [1]})).tolist()
    )


def test_mean_baseline_raises_not_fitted_before_fit():
    with pytest.raises(NotFittedError):
        MeanBaseline().predict(pd.DataFrame({"a": [1, 2]}))


def test_mean_baseline_is_deterministic_given_seed():
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([1.0, 5.0, 9.0])

    first = MeanBaseline(random_state=42).fit(X, y).predict(X)
    second = MeanBaseline(random_state=42).fit(X, y).predict(X)
    assert first.tolist() == second.tolist()


def test_mean_baseline_conforms_to_the_sklearn_api():
    model = MeanBaseline(random_state=7)
    assert model.get_params() == {"random_state": 7}

    model.set_params(random_state=3)
    assert model.random_state == 3

    fitted = model.fit(pd.DataFrame({"a": [1, 2]}), pd.Series([1.0, 3.0]))
    assert fitted is model  # fit returns self

    cloned = clone(model)
    assert cloned.get_params() == model.get_params()
    assert not hasattr(cloned, "mean_")  # clone() must not carry fitted state


# --- DomainLinearBaseline --------------------------------------------------------


def _domain_frame(n=20, seed=1):
    rng = np.random.default_rng(seed)
    lambda_max = rng.uniform(450, 600, n)
    dye_loading = rng.uniform(0.1, 2.0, n)
    noise_column = rng.uniform(-1000, 1000, n)  # must be ignored by the model
    return pd.DataFrame(
        {
            "lambda_max_nm": lambda_max,
            "sensitization_concentration_mm": dye_loading,
            "unrelated_noise_column": noise_column,
        }
    )


def test_domain_linear_baseline_raises_not_fitted_before_fit():
    with pytest.raises(NotFittedError):
        DomainLinearBaseline().predict(_domain_frame())


def test_domain_linear_baseline_rejects_non_dataframe_input():
    model = DomainLinearBaseline()
    with pytest.raises(ValueError, match="pandas DataFrame"):
        model.fit(np.zeros((5, 2)), np.zeros(5))


def test_domain_linear_baseline_rejects_missing_required_columns():
    model = DomainLinearBaseline()
    with pytest.raises(ValueError, match="missing required column"):
        model.fit(pd.DataFrame({"lambda_max_nm": [500.0, 510.0]}), pd.Series([1.0, 2.0]))


def test_domain_linear_baseline_ignores_columns_outside_its_two_named_features():
    X = _domain_frame()
    y = pd.Series(2.0 * X["lambda_max_nm"] + 5.0 * X["sensitization_concentration_mm"])

    model_a = DomainLinearBaseline().fit(X, y)
    # a second frame with the SAME two relevant columns but a wildly
    # different noise column must give identical predictions
    X_shuffled_noise = X.copy()
    X_shuffled_noise["unrelated_noise_column"] = X["unrelated_noise_column"].to_numpy()[::-1]
    model_b = DomainLinearBaseline().fit(X_shuffled_noise, y)

    np.testing.assert_allclose(model_a.predict(X), model_b.predict(X_shuffled_noise))


def test_domain_linear_baseline_recovers_a_known_noise_free_linear_relationship():
    X = _domain_frame(n=50)
    y = 2.0 * X["lambda_max_nm"] + 5.0 * X["sensitization_concentration_mm"] + 10.0

    model = DomainLinearBaseline().fit(X, y)
    predictions = model.predict(X)

    np.testing.assert_allclose(predictions, y.to_numpy(), rtol=1e-6)


def test_domain_linear_baseline_handles_missing_values_via_imputation():
    X = _domain_frame(n=30)
    X.loc[X.index[0], "lambda_max_nm"] = np.nan
    X.loc[X.index[1], "sensitization_concentration_mm"] = np.nan
    y = 2.0 * X["lambda_max_nm"].fillna(X["lambda_max_nm"].median()) + 1.0

    model = DomainLinearBaseline().fit(X, y)
    predictions = model.predict(X)

    assert not np.isnan(predictions).any()
    assert len(predictions) == len(X)


def test_domain_linear_baseline_is_deterministic_given_seed():
    X = _domain_frame()
    y = pd.Series(2.0 * X["lambda_max_nm"] + 5.0 * X["sensitization_concentration_mm"])

    first = DomainLinearBaseline(random_state=42).fit(X, y).predict(X)
    second = DomainLinearBaseline(random_state=42).fit(X, y).predict(X)
    np.testing.assert_array_equal(first, second)


def test_domain_linear_baseline_conforms_to_the_sklearn_api():
    model = DomainLinearBaseline(random_state=7)
    assert model.get_params() == {"random_state": 7}

    model.set_params(random_state=3)
    assert model.random_state == 3

    X = _domain_frame()
    y = pd.Series(2.0 * X["lambda_max_nm"])
    fitted = model.fit(X, y)
    assert fitted is model

    cloned = clone(model)
    assert cloned.get_params() == model.get_params()
    assert not hasattr(cloned, "pipeline_")


def test_domain_feature_columns_matches_the_two_columns_actually_used():
    # guards against the constant and the implementation silently drifting apart
    assert DOMAIN_FEATURE_COLUMNS == ("lambda_max_nm", "sensitization_concentration_mm")
