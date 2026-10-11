"""Tests for the grouped nested CV harness (Chunk 4.4). The two
PLAN.txt-mandated checks:

- LEAKAGE CANARY: on replicate-structured data (several devices per
  condition sharing a condition-level PCE offset no feature explains),
  deliberately ungrouped CV must score clearly BETTER than grouped CV --
  proving the grouping is doing something, and that the ungrouped case
  really is the leakage failure mode the harness guards against.
- Repeated-run determinism given a seed.

Plus structural checks that no fit -- outer or inner -- ever sees the
held-out condition, that every row gets exactly one out-of-fold
prediction per repeat, that the feature budget is enforced, and that the
full default model suite (every M4 model + both baselines) runs end to
end and produces the comparative table with CIs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from dssc_discovery.evaluation.cv_harness import (
    OUTCOME_COLUMNS,
    ModelSpec,
    default_feature_specs,
    default_model_specs,
    run_grouped_nested_cv,
)
from dssc_discovery.models.baselines import MeanBaseline
from dssc_discovery.models.linear_models import RidgeModel
from dssc_discovery.models.tree_models import RandomForestModel

TARGET = "power_conversion_efficiency_percent"
FEATURES = ["ph", "pigment_concentration_mm", "adsorption_time_min"]
FEATURE_SPECS = default_feature_specs(FEATURES)


def _replicate_frame(n_conditions=30, n_replicates=3, condition_effect_std=1.5, seed=0):
    """Replicates of a condition share identical features and one
    condition-level offset; the learnable part is a smooth function of
    the features."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_conditions):
        ph = rng.uniform(2, 9)
        conc = rng.uniform(1, 30)
        ads = rng.uniform(30, 300)
        offset = rng.normal(0, condition_effect_std)
        signal = 4.0 + 0.6 * (ph - 5.5) - 0.05 * (conc - 15) + 0.004 * (ads - 150)
        for r in range(n_replicates):
            rows.append(
                {
                    "group_id": f"cond-{c:03d}",
                    "ph": ph,
                    "pigment_concentration_mm": conc,
                    "adsorption_time_min": ads,
                    "lambda_max_nm": 520 + 10 * rng.normal(),
                    "sensitization_concentration_mm": rng.uniform(0.1, 2.0),
                    TARGET: signal + offset + rng.normal(0, 0.05),
                }
            )
    return pd.DataFrame(rows)


class _GroupSpy(BaseEstimator, RegressorMixin):
    """Records which groups and columns every fit() call saw. Class-level
    log because GridSearchCV fits clones, not this instance."""

    fits: list[set[str]] = []
    columns_seen: list[set[str]] = []

    def __init__(self, c=1.0, random_state=None):
        self.c = c
        self.random_state = random_state

    def fit(self, X, y):
        type(self).fits.append(set(X["group_id"]))
        type(self).columns_seen.append(set(X.columns))
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def _mae(result, model):
    return float(result.summary.set_index("model").loc[model, "mae"])


def test_leakage_canary_ungrouped_cv_scores_better_than_grouped_cv():
    data = _replicate_frame()
    models = [ModelSpec("rf", RandomForestModel(feature_specs=FEATURE_SPECS), "tree")]
    kwargs = dict(n_repeats=2, n_outer_splits=5, n_bootstrap=50, seed=11)

    grouped = run_grouped_nested_cv(data, models, grouped=True, **kwargs)
    ungrouped = run_grouped_nested_cv(data, models, grouped=False, **kwargs)

    # Ungrouped CV lets the RF memorize a held-out replicate's sibling
    # (identical features, same condition offset); grouped CV can't.
    assert _mae(ungrouped, "rf") < 0.5 * _mae(grouped, "rf")


def test_repeated_runs_are_deterministic_given_the_seed():
    data = _replicate_frame(n_conditions=15, n_replicates=2)
    models = [
        ModelSpec(
            "rf",
            RandomForestModel(feature_specs=FEATURE_SPECS, n_estimators=20),
            "tree",
            {"max_depth": [2, None]},
        )
    ]
    kwargs = dict(n_repeats=2, n_outer_splits=3, n_inner_splits=3, n_bootstrap=50)

    first = run_grouped_nested_cv(data, models, seed=5, **kwargs)
    second = run_grouped_nested_cv(data, models, seed=5, **kwargs)
    other = run_grouped_nested_cv(data, models, seed=6, **kwargs)

    pd.testing.assert_frame_equal(first.summary, second.summary)
    pd.testing.assert_frame_equal(first.predictions, second.predictions)
    assert not np.allclose(first.predictions["y_pred"], other.predictions["y_pred"])


def test_no_fit_outer_or_inner_ever_sees_the_held_out_condition():
    _GroupSpy.fits = []
    _GroupSpy.columns_seen = []
    data = _replicate_frame(n_conditions=8, n_replicates=2)
    models = [ModelSpec("spy", _GroupSpy(), "baseline", {"c": [1.0, 2.0]})]

    result = run_grouped_nested_cv(data, models, n_repeats=1, n_inner_splits=3, n_bootstrap=10)

    all_groups = set(data["group_id"])
    n_folds = len(all_groups)  # LOCO
    # per outer fold: 3 inner splits x 2 grid points + 1 refit
    assert len(_GroupSpy.fits) == n_folds * (3 * 2 + 1)
    # LOCO: every fit lacks at least the held-out condition; a global fit
    # on all conditions would be leakage.
    assert all(len(seen) <= len(all_groups) - 1 for seen in _GroupSpy.fits)
    # Inner fits lack at least two conditions (outer + inner held out).
    assert sum(len(seen) == len(all_groups) - 1 for seen in _GroupSpy.fits) == n_folds
    assert result.config["strategy"] == "A_loco"


def test_outcome_columns_never_reach_a_model():
    _GroupSpy.fits = []
    _GroupSpy.columns_seen = []
    data = _replicate_frame(n_conditions=6, n_replicates=2)
    data["open_circuit_voltage_v"] = 0.6
    data["short_circuit_current_density_ma_cm2"] = 10.0
    data["fill_factor"] = 0.65

    run_grouped_nested_cv(data, [ModelSpec("spy", _GroupSpy(), "baseline")], n_repeats=1)

    assert _GroupSpy.columns_seen
    for columns in _GroupSpy.columns_seen:
        assert not columns & set(OUTCOME_COLUMNS)


def test_every_row_gets_exactly_one_prediction_per_model_and_repeat():
    data = _replicate_frame(n_conditions=10, n_replicates=3)
    models = [
        ModelSpec("mean", MeanBaseline(), "baseline"),
        ModelSpec("ridge", RidgeModel(feature_specs=FEATURE_SPECS), "linear_or_gp"),
    ]

    result = run_grouped_nested_cv(data, models, n_repeats=2, n_outer_splits=4, n_bootstrap=20)

    counts = result.predictions.groupby(["model", "repeat"])["row"].agg(["count", "nunique"])
    assert (counts["count"] == len(data)).all()
    assert (counts["nunique"] == len(data)).all()
    assert result.predictions["y_pred"].notna().all()
    # grouped k-fold: a condition's replicates always land in the same test fold
    assert (result.fold_log["n_test"] == result.fold_log["n_test_groups"] * 3).all()


def test_summary_reports_cis_and_flags_models_that_beat_the_baselines():
    data = _replicate_frame(n_conditions=40, n_replicates=2, condition_effect_std=0.2)
    models = [
        ModelSpec("mean", MeanBaseline(), "baseline"),
        ModelSpec("ridge", RidgeModel(feature_specs=FEATURE_SPECS), "linear_or_gp"),
    ]

    result = run_grouped_nested_cv(data, models, n_repeats=2, n_outer_splits=5, n_bootstrap=200)
    summary = result.comparison_table().set_index("model")

    for metric in ("mae", "rmse", "r2"):
        assert (summary[f"{metric}_ci_low"] <= summary[metric]).all()
        assert (summary[metric] <= summary[f"{metric}_ci_high"]).all()
    assert summary.loc["ridge", "beats_all_baselines"] is True
    assert summary.loc["mean", "beats_all_baselines"] is None
    assert summary.loc["ridge", "r2"] > 0.8
    assert len(result.repeat_metrics) == 2 * 2


def test_feature_budget_is_enforced():
    too_many = default_feature_specs(
        [
            "ph",
            "pigment_concentration_mm",
            "adsorption_time_min",
            "solvent_ratio",
            "extraction_temperature_c",
            "extraction_time_min",
            "sensitization_concentration_mm",
            "sensitization_temperature_c",
            "tio2_thickness_um",
        ]
    )
    models = [ModelSpec("rf", RandomForestModel(feature_specs=too_many), "tree")]

    with pytest.raises(ValueError, match="budget"):
        run_grouped_nested_cv(_replicate_frame(n_conditions=6), models, n_repeats=1)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda df: df.drop(columns=[TARGET]), "missing required column"),
        (lambda df: df.assign(**{TARGET: np.nan}), "contains NaN"),
        (lambda df: df.assign(group_id="only-one"), "at least 3 distinct groups"),
    ],
)
def test_rejects_invalid_data(mutate, message):
    data = mutate(_replicate_frame(n_conditions=6))
    models = [ModelSpec("mean", MeanBaseline(), "baseline")]
    with pytest.raises(ValueError, match=message):
        run_grouped_nested_cv(data, models, n_repeats=1)


def test_rejects_duplicate_model_names():
    models = [
        ModelSpec("mean", MeanBaseline(), "baseline"),
        ModelSpec("mean", MeanBaseline(), "baseline"),
    ]
    with pytest.raises(ValueError, match="unique"):
        run_grouped_nested_cv(_replicate_frame(n_conditions=6), models, n_repeats=1)


def test_default_suite_runs_every_m4_model_and_both_baselines():
    data = _replicate_frame(n_conditions=24, n_replicates=2, condition_effect_std=0.2)
    models = default_model_specs(FEATURE_SPECS, FEATURE_SPECS)

    result = run_grouped_nested_cv(
        data, models, n_repeats=1, n_outer_splits=3, n_inner_splits=3, n_bootstrap=50
    )

    assert list(result.summary["model"]) == [
        "mean_baseline",
        "domain_linear_baseline",
        "ridge",
        "lasso",
        "random_forest",
        "gradient_boosting",
        "xgboost",
        "gpr",
    ]
    assert result.summary[["mae", "rmse", "r2"]].notna().all().all()
    tuned = result.fold_log[result.fold_log["model"] == "random_forest"]["best_params"]
    assert all(set(p) == {"max_depth", "min_samples_leaf"} for p in tuned)
