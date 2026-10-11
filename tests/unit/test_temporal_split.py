"""Tests for validation strategy B, forward extrapolation (Chunk 4.5):
split boundaries are correct on fixture data -- earlier stages train,
the test stage tests, later stages and (by default) MANUAL conditions
are dropped -- and the runner evaluates only the test stage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dssc_discovery.domain.enums import DesignSource
from dssc_discovery.evaluation.cv_harness import ModelSpec, default_feature_specs
from dssc_discovery.evaluation.temporal_split import (
    forward_extrapolation_rows,
    run_forward_extrapolation_cv,
)
from dssc_discovery.models.baselines import MeanBaseline
from dssc_discovery.models.linear_models import RidgeModel

TARGET = "power_conversion_efficiency_percent"

# stage -> number of conditions; two replicates each
STAGE_SIZES = {
    "initial_doe": 8,
    "bo_cycle_1": 4,
    "bo_cycle_2": 4,
    "bo_cycle_3": 4,
    "manual": 2,
}


def _staged_frame(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for stage, n_conditions in STAGE_SIZES.items():
        for c in range(n_conditions):
            ph = rng.uniform(2, 9)
            for r in range(2):
                rows.append(
                    {
                        "group_id": f"{stage}-{c}",
                        "design_source": stage,
                        "ph": ph,
                        TARGET: 2.0 + 0.5 * ph + rng.normal(0, 0.05),
                    }
                )
    return pd.DataFrame(rows)


def test_default_split_trains_on_doe_and_cycles_1_2_and_tests_on_cycle_3():
    kept, (train_idx, test_idx) = forward_extrapolation_rows(_staged_frame())

    assert set(kept["design_source"].iloc[train_idx]) == {
        "initial_doe",
        "bo_cycle_1",
        "bo_cycle_2",
    }
    assert set(kept["design_source"].iloc[test_idx]) == {"bo_cycle_3"}
    assert "manual" not in set(kept["design_source"])
    assert len(train_idx) == (8 + 4 + 4) * 2
    assert len(test_idx) == 4 * 2
    assert not set(kept["group_id"].iloc[train_idx]) & set(kept["group_id"].iloc[test_idx])


def test_earlier_test_stage_drops_later_stages_entirely():
    kept, (train_idx, test_idx) = forward_extrapolation_rows(
        _staged_frame(), test_stage=DesignSource.BO_CYCLE_1
    )

    assert set(kept["design_source"]) == {"initial_doe", "bo_cycle_1"}
    assert set(kept["design_source"].iloc[train_idx]) == {"initial_doe"}
    assert set(kept["design_source"].iloc[test_idx]) == {"bo_cycle_1"}


def test_manual_policy_train_keeps_manual_conditions_as_training_rows_only():
    kept, (train_idx, test_idx) = forward_extrapolation_rows(_staged_frame(), manual_policy="train")

    assert "manual" in set(kept["design_source"].iloc[train_idx])
    assert "manual" not in set(kept["design_source"].iloc[test_idx])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"test_stage": DesignSource.INITIAL_DOE}, "test_stage"),
        ({"test_stage": DesignSource.MANUAL}, "test_stage"),
        ({"manual_policy": "test"}, "manual_policy"),
    ],
)
def test_rejects_invalid_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        forward_extrapolation_rows(_staged_frame(), **kwargs)


def test_rejects_an_empty_test_stage():
    data = _staged_frame()
    data = data[data["design_source"] != "bo_cycle_3"]
    with pytest.raises(ValueError, match="no test rows"):
        forward_extrapolation_rows(data)


def test_rejects_a_condition_straddling_two_stages():
    data = _staged_frame()
    data.loc[data.index[-1 - 2 * STAGE_SIZES["manual"]], "group_id"] = data["group_id"].iloc[0]
    with pytest.raises(ValueError, match="both train and test"):
        forward_extrapolation_rows(data)


def test_runner_evaluates_only_the_test_stage():
    data = _staged_frame()
    models = [
        ModelSpec("mean", MeanBaseline(), "baseline"),
        ModelSpec("ridge", RidgeModel(feature_specs=default_feature_specs(["ph"])), "linear_or_gp"),
    ]

    result = run_forward_extrapolation_cv(
        data, models, n_repeats=2, n_inner_splits=3, n_bootstrap=50
    )

    assert result.config["strategy"] == "B_forward_extrapolation"
    assert result.config["test_stage"] == "bo_cycle_3"
    assert result.config["n_rows_evaluated"] == 4 * 2
    assert result.config["n_rows_dropped"] == 2 * 2  # the manual conditions
    assert set(result.predictions["group_id"].str.split("-").str[0]) == {"bo_cycle_3"}
    summary = result.summary.set_index("model")
    assert summary.loc["ridge", "mae"] < summary.loc["mean", "mae"]
