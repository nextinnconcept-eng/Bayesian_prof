"""Validation strategy B -- forward extrapolation (Chunk 4.5, PLAN.txt
Section H): train on the initial DoE plus earlier BO cycles, test on a
later BO cycle (default: DoE + cycles 1-2 -> cycle 3). This is the
operational question -- does the model, as it will actually be used,
generalize to conditions proposed AFTER its training data existed? -- and
is reported separately from strategy A, never folded into one number.

"Time" is the design_source stage (DesignSource), not a wall-clock
timestamp: a condition's position in the DoE/BO programme is what decides
what a model could legitimately have known when it was proposed.
Conditions from stages after the test stage are dropped entirely (never
train on the future).

MANUAL conditions have no place in that ordering -- one could have been
added before the DoE or after cycle 3 -- so the caller must decide:
manual_policy="exclude" (default, the conservative choice) drops them;
"train" keeps them as training rows, appropriate only if they're known
to predate the test stage. They are never used as test rows.

Inner hyperparameter search stays grouped by condition within the
training stages (run_nested_cv); since all replicates of a condition
share one design_source, a condition can never straddle the split.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from dssc_discovery.domain.enums import DesignSource
from dssc_discovery.evaluation.cv_harness import (
    GROUP_COLUMN,
    CVResult,
    ModelSpec,
    Split,
    run_nested_cv,
)

DESIGN_SOURCE_COLUMN = "design_source"

ManualPolicy = Literal["exclude", "train"]

# Chronological order of the DoE/BO programme. MANUAL is deliberately
# absent -- see manual_policy.
STAGE_ORDER: tuple[DesignSource, ...] = (
    DesignSource.INITIAL_DOE,
    DesignSource.BO_CYCLE_1,
    DesignSource.BO_CYCLE_2,
    DesignSource.BO_CYCLE_3,
)


def _stage_values(data: pd.DataFrame, column: str) -> pd.Series:
    if column not in data.columns:
        raise ValueError(f"data is missing required column {column!r}")
    values = data[column].astype(str)
    valid = {s.value for s in DesignSource}
    unknown = sorted(set(values) - valid)
    if unknown:
        raise ValueError(f"unknown design_source value(s) {unknown}")
    return values


def forward_extrapolation_rows(
    data: pd.DataFrame,
    test_stage: DesignSource = DesignSource.BO_CYCLE_3,
    manual_policy: ManualPolicy = "exclude",
    design_source_column: str = DESIGN_SOURCE_COLUMN,
) -> tuple[pd.DataFrame, Split]:
    """Return (the rows strategy B uses, positional (train_idx, test_idx)
    into that frame). Raises if either side would be empty or if any
    condition (group) would land on both sides."""
    if test_stage not in STAGE_ORDER or test_stage == STAGE_ORDER[0]:
        raise ValueError(
            f"test_stage must be a BO cycle with earlier stages to train on, got {test_stage!r}"
        )
    if manual_policy not in ("exclude", "train"):
        raise ValueError(f"manual_policy must be 'exclude' or 'train', got {manual_policy!r}")

    stages = _stage_values(data, design_source_column)
    test_position = STAGE_ORDER.index(test_stage)
    train_stages = {s.value for s in STAGE_ORDER[:test_position]}
    if manual_policy == "train":
        train_stages.add(DesignSource.MANUAL.value)

    is_train = stages.isin(train_stages).to_numpy()
    is_test = (stages == test_stage.value).to_numpy()
    kept = data.loc[is_train | is_test].reset_index(drop=True)
    is_test_kept = is_test[is_train | is_test]

    train_idx = np.flatnonzero(~is_test_kept)
    test_idx = np.flatnonzero(is_test_kept)
    if len(train_idx) == 0:
        raise ValueError(f"no training rows from stages before {test_stage.value!r}")
    if len(test_idx) == 0:
        raise ValueError(f"no test rows for stage {test_stage.value!r}")
    if GROUP_COLUMN in kept.columns:
        straddling = set(kept[GROUP_COLUMN].iloc[train_idx]) & set(
            kept[GROUP_COLUMN].iloc[test_idx]
        )
        if straddling:
            raise ValueError(
                f"condition(s) {sorted(straddling)[:5]} appear in both train and test stages; "
                "a condition's replicates must share one design_source"
            )
    return kept, (train_idx, test_idx)


def run_forward_extrapolation_cv(
    data: pd.DataFrame,
    models: Sequence[ModelSpec],
    *,
    test_stage: DesignSource = DesignSource.BO_CYCLE_3,
    manual_policy: ManualPolicy = "exclude",
    design_source_column: str = DESIGN_SOURCE_COLUMN,
    **kwargs: Any,
) -> CVResult:
    """Strategy B. Extra keyword arguments (n_repeats, seed,
    n_inner_splits, n_bootstrap, ...) go to run_nested_cv. The outer
    split is fixed, so repeats vary only model seeds and inner shuffles;
    the CI (condition bootstrap over the test stage) carries the real
    uncertainty."""
    kept, split = forward_extrapolation_rows(data, test_stage, manual_policy, design_source_column)
    stages = _stage_values(data, design_source_column)
    test_position = STAGE_ORDER.index(test_stage)
    return run_nested_cv(
        kept,
        models,
        lambda _data, _seed: [split],
        strategy="B_forward_extrapolation",
        extra_config={
            "test_stage": test_stage.value,
            "train_stages": [s.value for s in STAGE_ORDER[:test_position]]
            + (["manual"] if manual_policy == "train" else []),
            "manual_policy": manual_policy,
            "n_rows_dropped": int(len(data) - len(kept)),
            "n_manual_rows": int((stages == DesignSource.MANUAL.value).sum()),
        },
        **kwargs,
    )
