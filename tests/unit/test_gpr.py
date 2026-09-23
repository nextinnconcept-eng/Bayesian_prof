"""Tests for the Gaussian Process regression wrapper (Chunk 4.3):
sklearn-API conformance and determinism (same contract as every other
model wrapper), that it actually uses an explicit Matern-5/2 kernel
(BoTorch's SingleTaskGP default is RBF, confirmed empirically before
writing the module -- worth checking here too, not just trusting the
constructor call), and the two PLAN.txt-mandated checks: fitting on
the synthetic ground truth (3.5) recovers its shape on held-out points
within tolerance, and predictive std is lower near training points
than far outside the training distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from dssc_discovery.features.registry import get_feature
from dssc_discovery.features.synthetic import _sample_vector, synthetic_pce
from dssc_discovery.models.gpr import GPRModel

FEATURE_SPECS = [get_feature("ph"), get_feature("pigment_concentration_mm")]
GP_GROUND_TRUTH_FEATURE_NAMES = (
    "ph",
    "pigment_concentration_mm",
    "adsorption_time_min",
    "tio2_thickness_um",
)
GP_GROUND_TRUTH_FEATURE_SPECS = [get_feature(name) for name in GP_GROUND_TRUTH_FEATURE_NAMES]


def _toy_frame(n=40, seed=1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "ph": rng.uniform(2, 9, n),
            "pigment_concentration_mm": rng.uniform(1, 30, n),
            "unrelated_noise_column": rng.uniform(-1000, 1000, n),
        }
    )
    y = 3.0 * df["ph"] - 0.5 * df["pigment_concentration_mm"] + rng.normal(0, 0.1, n)
    return df, y


def _ground_truth_frame(n, seed):
    rng = np.random.default_rng(seed)
    vectors = [_sample_vector(rng) for _ in range(n)]
    df = pd.DataFrame(
        {name: [getattr(v, name) for v in vectors] for name in GP_GROUND_TRUTH_FEATURE_NAMES}
    )
    true_pce = np.array([synthetic_pce(v) for v in vectors])
    return df, true_pce


def test_raises_not_fitted_before_fit():
    with pytest.raises(NotFittedError):
        GPRModel(feature_specs=FEATURE_SPECS).predict(_toy_frame()[0])


def test_fit_returns_self_and_predict_gives_right_length():
    X, y = _toy_frame()
    model = GPRModel(feature_specs=FEATURE_SPECS, random_state=0)
    fitted = model.fit(X, y)
    assert fitted is model

    predictions = model.predict(X)
    assert len(predictions) == len(X)
    assert not np.isnan(predictions).any()


def test_is_deterministic_given_the_same_seed():
    X, y = _toy_frame()
    first = GPRModel(feature_specs=FEATURE_SPECS, random_state=42).fit(X, y).predict(X)
    second = GPRModel(feature_specs=FEATURE_SPECS, random_state=42).fit(X, y).predict(X)
    np.testing.assert_allclose(first, second)


def test_conforms_to_the_sklearn_api():
    model = GPRModel(feature_specs=FEATURE_SPECS, random_state=7)
    assert model.get_params()["random_state"] == 7

    model.set_params(random_state=3)
    assert model.random_state == 3

    X, y = _toy_frame()
    fitted = model.fit(X, y)
    assert fitted is model

    cloned = clone(model)
    assert cloned.get_params() == model.get_params()
    assert not hasattr(cloned, "pipeline_")


def test_ignores_columns_outside_its_feature_specs():
    X, y = _toy_frame()
    model_a = GPRModel(feature_specs=FEATURE_SPECS, random_state=0).fit(X, y)

    X_shuffled_noise = X.copy()
    X_shuffled_noise["unrelated_noise_column"] = X["unrelated_noise_column"].to_numpy()[::-1]
    model_b = GPRModel(feature_specs=FEATURE_SPECS, random_state=0).fit(X_shuffled_noise, y)

    np.testing.assert_allclose(model_a.predict(X), model_b.predict(X_shuffled_noise))


def test_return_std_gives_a_mean_matching_the_default_call_plus_positive_std():
    X, y = _toy_frame()
    model = GPRModel(feature_specs=FEATURE_SPECS, random_state=0).fit(X, y)

    mean_only = model.predict(X)
    mean_with_std, std = model.predict(X, return_std=True)

    np.testing.assert_allclose(mean_only, mean_with_std)
    assert std.shape == mean_with_std.shape
    assert (std > 0).all()


def test_uses_an_explicit_matern_5_2_kernel_not_botorchs_default_rbf():
    X, y = _toy_frame(n=25)
    model = GPRModel(feature_specs=FEATURE_SPECS, random_state=0).fit(X, y)

    fitted_gp = model.pipeline_.named_steps["model"].model_
    base_kernel = fitted_gp.covar_module.base_kernel
    assert type(base_kernel).__name__ == "MaternKernel"
    assert base_kernel.nu == 2.5


def test_recovers_the_synthetic_pce_ground_truth_shape_on_held_out_points():
    rng = np.random.default_rng(2024)
    train_df, train_true_pce = _ground_truth_frame(80, seed=2024)
    train_y_noisy = train_true_pce + rng.normal(0, 0.3, len(train_true_pce))
    test_df, test_true_pce = _ground_truth_frame(60, seed=99)

    model = GPRModel(feature_specs=GP_GROUND_TRUTH_FEATURE_SPECS, random_state=0)
    model.fit(train_df, train_y_noisy)
    predicted = model.predict(test_df)

    mae = float(np.mean(np.abs(predicted - test_true_pce)))
    # empirically ~0.28 at this N against a target with a ~1-8.5% range;
    # a loose bound so this doesn't flake, while still failing on a
    # genuinely broken fit (e.g. an untrained/unconverged model, whose
    # MAE would be far larger against a function with this much shape).
    assert mae < 1.5


def test_predictive_std_is_lower_near_training_points_than_far_outside_the_range():
    rng = np.random.default_rng(1)
    train_df, train_true_pce = _ground_truth_frame(60, seed=1)
    train_y_noisy = train_true_pce + rng.normal(0, 0.3, len(train_true_pce))

    model = GPRModel(feature_specs=GP_GROUND_TRUTH_FEATURE_SPECS, random_state=0)
    model.fit(train_df, train_y_noisy)

    near_point = train_df.iloc[[0]]
    far_point = pd.DataFrame(
        {
            "ph": [500.0],
            "pigment_concentration_mm": [5000.0],
            "adsorption_time_min": [50000.0],
            "tio2_thickness_um": [5000.0],
        }
    )

    _, std_near = model.predict(near_point, return_std=True)
    _, std_far = model.predict(far_point, return_std=True)

    assert std_far[0] > std_near[0] * 3
