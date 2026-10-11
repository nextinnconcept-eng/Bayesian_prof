"""Leakage-safe grouped nested cross-validation harness (Chunk 4.4):
validation strategy A (interpolation, PLAN.txt Section H) -- outer
Leave-One-Condition-Out (LOCO) CV grouped by experimental_condition_id,
with hyperparameter search in an inner, also-grouped k-fold run INSIDE
each outer training fold only.

Why grouping is the whole point: replicate devices of one
ExperimentalCondition share every design-time feature value and a
condition-level offset in PCE that no feature explains (same pigment
batch, same fabrication day, ...). If one replicate sits in the training
fold and its sibling in the test fold, a flexible model can simply
memorize the sibling's PCE and look far more accurate than it will ever
be on a genuinely new condition. tests/unit/test_cv_harness.py's leakage
canary proves this failure mode is real on replicate-structured data and
that grouping removes it -- run_grouped_nested_cv(grouped=False) exists
ONLY so that canary can demonstrate the difference; it is not a valid
evaluation mode for reporting.

Where fitting happens, explicitly:
- Every model is cloned fresh for every (repeat, outer fold) and fit only
  on that fold's training rows. Each model wrapper (models/base.py)
  builds its preprocessing pipeline (3.4) inside fit(), so imputation /
  scaling / encoding statistics are fit on training rows only, too.
- Hyperparameter search (GridSearchCV) receives ONLY the outer training
  rows and their groups; its inner splitter is grouped as well, so the
  inner validation score that picks hyperparameters isn't inflated by
  replicate leakage either. The refit on the full outer training fold is
  what predicts the held-out condition.
- No fit ever sees the outer test rows. Measured outcomes other than the
  target (Voc, Jsc, FF -- they determine PCE almost exactly) are dropped
  from X before any model sees it, so a model that selects columns
  carelessly can't cheat through them either.

Repeats and confidence intervals: LOCO's outer split is deterministic,
so each of n_repeats re-runs with a fresh seed drawn from `seed` --
threaded into every model's random_state and into the inner (and, with
n_outer_splits set, the outer) grouped k-fold shuffle. Metrics are
computed on each repeat's pooled out-of-fold predictions (a LOCO test
fold holding one condition has no meaningful per-fold R²). The point
estimate is the mean over repeats; the CI is a percentile bootstrap that
resamples CONDITIONS (whole groups, never individual replicates) with
replacement, which captures the sampling uncertainty that dominates at
N~70-85 rather than only the comparatively tiny seed-to-seed variation.
The same bootstrap resamples are used for every model, so CIs are
comparable side by side.

Feature budget (registry.FEATURE_BUDGET) is enforced here, at the one
place models are fit, so "just use all the columns" can't slip through.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, LeaveOneGroupOut

from dssc_discovery.features.registry import FEATURE_BUDGET, FeatureSpec, get_feature
from dssc_discovery.models.baselines import DomainLinearBaseline, MeanBaseline
from dssc_discovery.models.gpr import GPRModel
from dssc_discovery.models.linear_models import LassoModel, RidgeModel
from dssc_discovery.models.tree_models import (
    GradientBoostingModel,
    RandomForestModel,
    XGBoostModel,
)

ModelFamily = Literal["baseline", "tree", "linear_or_gp"]

TARGET_COLUMN = "power_conversion_efficiency_percent"
GROUP_COLUMN = "group_id"

# Measured device outcomes (dataset_builder.DATASET_COLUMNS). PCE is
# Voc * Jsc * FF / Pin, so any of the other three alongside the target
# would be a near-perfect leak -- they never reach a model as features.
OUTCOME_COLUMNS: tuple[str, ...] = (
    "open_circuit_voltage_v",
    "short_circuit_current_density_ma_cm2",
    "fill_factor",
    TARGET_COLUMN,
)

_METRICS: tuple[str, ...] = ("mae", "rmse", "r2")

Split = tuple[np.ndarray, np.ndarray]
# (positionally indexed data, repeat seed) -> list of (train_idx, test_idx).
# Every strategy (A here, B/C in temporal_split.py / cross_species.py) is
# just a different OuterSplitFn plugged into run_nested_cv().
OuterSplitFn = Callable[[pd.DataFrame, int], list[Split]]


@dataclass(frozen=True)
class ModelSpec:
    """One candidate model for the harness: an unfit prototype estimator
    (cloned fresh for every fold, never fit itself), the family that
    decides its feature budget, and an optional GridSearchCV param_grid
    searched in the inner CV. An empty param_grid means "no tuning": the
    model is fit directly on the outer training fold.
    """

    name: str
    estimator: BaseEstimator
    family: ModelFamily
    param_grid: Mapping[str, Sequence[Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class CVResult:
    """Everything one harness run produced.

    - predictions: one row per (model, repeat, dataset row) -- the
      out-of-fold prediction for that row.
    - fold_log: one row per (model, repeat, outer fold): sizes and the
      hyperparameters the inner search chose, for auditability.
    - repeat_metrics: MAE/RMSE/R² on each repeat's pooled out-of-fold
      predictions.
    - summary: the comparative table (comparison_table()).
    """

    config: dict[str, Any]
    predictions: pd.DataFrame
    fold_log: pd.DataFrame
    repeat_metrics: pd.DataFrame
    summary: pd.DataFrame

    def comparison_table(self) -> pd.DataFrame:
        return self.summary.copy()


def default_model_specs(
    tree_features: Sequence[FeatureSpec],
    linear_or_gp_features: Sequence[FeatureSpec],
) -> list[ModelSpec]:
    """All M4 models plus both mandatory baselines (Section H), with
    deliberately small grids: at N~70-85 conditions an inner fold holds
    ~50 conditions, and a wide grid would mostly fit noise in the inner
    validation score.
    """
    tree = list(tree_features)
    linear_or_gp = list(linear_or_gp_features)
    return [
        ModelSpec("mean_baseline", MeanBaseline(), "baseline"),
        ModelSpec("domain_linear_baseline", DomainLinearBaseline(), "baseline"),
        ModelSpec(
            "ridge",
            RidgeModel(feature_specs=linear_or_gp),
            "linear_or_gp",
            {"alpha": [0.1, 1.0, 10.0]},
        ),
        ModelSpec(
            "lasso",
            LassoModel(feature_specs=linear_or_gp),
            "linear_or_gp",
            {"alpha": [0.01, 0.1, 1.0]},
        ),
        ModelSpec(
            "random_forest",
            RandomForestModel(feature_specs=tree),
            "tree",
            {"max_depth": [3, None], "min_samples_leaf": [1, 3]},
        ),
        ModelSpec(
            "gradient_boosting",
            GradientBoostingModel(feature_specs=tree),
            "tree",
            {"learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
        ),
        ModelSpec(
            "xgboost",
            XGBoostModel(feature_specs=tree),
            "tree",
            {"learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
        ),
        ModelSpec("gpr", GPRModel(feature_specs=linear_or_gp), "linear_or_gp"),
    ]


def default_feature_specs(names: Sequence[str]) -> list[FeatureSpec]:
    """Convenience: registry lookups for a list of column names."""
    return [get_feature(name) for name in names]


# --- validation --------------------------------------------------------------


def _check_feature_budget(spec: ModelSpec) -> None:
    if spec.family == "baseline":
        return
    feature_specs = spec.estimator.get_params().get("feature_specs")
    if feature_specs is None:
        return
    budget = FEATURE_BUDGET[spec.family]
    if len(feature_specs) > budget:
        raise ValueError(
            f"Model {spec.name!r} uses {len(feature_specs)} features, over the "
            f"{spec.family!r} budget of {budget} (registry.FEATURE_BUDGET, PLAN.txt "
            "Section H). Select fewer features."
        )


def _validate_inputs(
    data: pd.DataFrame,
    models: Sequence[ModelSpec],
    target: str,
    group_column: str,
    n_repeats: int,
    n_inner_splits: int,
    confidence_level: float,
) -> None:
    for column in (target, group_column):
        if column not in data.columns:
            raise ValueError(f"data is missing required column {column!r}")
    if data[target].isna().any():
        raise ValueError(
            f"target column {target!r} contains NaN; rows without a measured target "
            "don't belong in an ML-ready dataset (dataset_builder excludes them)"
        )
    if data[group_column].isna().any():
        raise ValueError(f"group column {group_column!r} contains NaN")
    n_groups = data[group_column].nunique()
    if n_groups < 3:
        raise ValueError(f"need at least 3 distinct groups for nested CV, got {n_groups}")
    if not models:
        raise ValueError("models must be non-empty")
    names = [m.name for m in models]
    if len(set(names)) != len(names):
        raise ValueError(f"model names must be unique, got {names}")
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")
    if n_inner_splits < 2:
        raise ValueError("n_inner_splits must be >= 2")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    for spec in models:
        _check_feature_budget(spec)


# --- splitting ---------------------------------------------------------------


def loco_or_grouped_kfold_splits(
    group_column: str, n_outer_splits: int | None, grouped: bool = True
) -> OuterSplitFn:
    """Strategy A's outer splitter. n_outer_splits=None -> LOCO;
    an integer -> shuffled grouped k-fold (reshuffled per repeat seed).
    grouped=False is the leakage-canary mode: same fold count as the
    grouped run, so training-set sizes are comparable, but replicates of
    one condition are free to straddle train/test."""

    def split(data: pd.DataFrame, seed: int) -> list[Split]:
        groups = data[group_column].to_numpy()
        if grouped:
            splitter: Any = (
                LeaveOneGroupOut()
                if n_outer_splits is None
                else GroupKFold(n_splits=n_outer_splits, shuffle=True, random_state=seed)
            )
        else:
            n_splits = n_outer_splits or len(np.unique(groups))
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            return list(splitter.split(data))
        return list(splitter.split(data, groups=groups))

    return split


def _inner_splitter(train_groups: np.ndarray, grouped: bool, n_inner_splits: int, seed: int) -> Any:
    if grouped:
        n_splits = min(n_inner_splits, len(np.unique(train_groups)))
        if n_splits < 2:
            raise ValueError("an outer training fold has fewer than 2 groups for inner CV")
        return GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return KFold(n_splits=min(n_inner_splits, len(train_groups)), shuffle=True, random_state=seed)


# --- fitting -----------------------------------------------------------------


def _seeded_clone(estimator: BaseEstimator, seed: int) -> BaseEstimator:
    model = clone(estimator)
    if "random_state" in model.get_params():
        model.set_params(random_state=seed)
    return model


def _fit_one_fold(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    grouped: bool,
    n_inner_splits: int,
    seed: int,
) -> tuple[BaseEstimator, dict[str, Any]]:
    model = _seeded_clone(spec.estimator, seed)
    if not spec.param_grid:
        return model.fit(X_train, y_train), {}

    search = GridSearchCV(
        model,
        param_grid=dict(spec.param_grid),
        cv=_inner_splitter(groups_train, grouped, n_inner_splits, seed),
        scoring="neg_mean_absolute_error",
        refit=True,
        error_score="raise",
    )
    search.fit(X_train, y_train, **({"groups": groups_train} if grouped else {}))
    return search.best_estimator_, dict(search.best_params_)


def model_features(data: pd.DataFrame, target: str = TARGET_COLUMN) -> pd.DataFrame:
    """The frame models are allowed to see: everything except measured
    outcomes. Shared with diagnostics.py so in-sample fits there see
    exactly what the harness's fits see."""
    return data.drop(columns=[c for c in (*OUTCOME_COLUMNS, target) if c in data.columns])


# --- metrics -----------------------------------------------------------------


def _metrics_by_repeat(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized over repeats: y_pred has shape (n_repeats, n_rows)."""
    errors = y_pred - y_true
    sse = np.sum(errors**2, axis=-1)
    sst = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "mae": np.mean(np.abs(errors), axis=-1),
        "rmse": np.sqrt(sse / y_true.shape[-1]),
        "r2": 1.0 - sse / sst if sst > 0 else np.full(sse.shape, np.nan),
    }


def _bootstrap_row_indices(
    group_codes: np.ndarray, n_bootstrap: int, seed: int
) -> Iterator[np.ndarray]:
    n_groups = int(group_codes.max()) + 1
    rows_by_group = [np.flatnonzero(group_codes == g) for g in range(n_groups)]
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        sampled = rng.integers(0, n_groups, n_groups)
        yield np.concatenate([rows_by_group[g] for g in sampled])


def _summarize(
    models: Sequence[ModelSpec],
    y: np.ndarray,
    oof: dict[str, np.ndarray],
    group_codes: np.ndarray,
    n_bootstrap: int,
    confidence_level: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """y / oof / group_codes cover only the rows that were ever held out
    (all rows for A and C; only the test stage for B)."""
    repeat_rows = []
    point: dict[str, dict[str, np.ndarray]] = {}
    for spec in models:
        per_repeat = _metrics_by_repeat(y, oof[spec.name])
        point[spec.name] = per_repeat
        for r in range(oof[spec.name].shape[0]):
            repeat_rows.append(
                {"model": spec.name, "repeat": r, **{m: float(per_repeat[m][r]) for m in _METRICS}}
            )

    # Same resampled condition sets for every model -> comparable CIs.
    boot: dict[str, dict[str, list[float]]] = {
        spec.name: {m: [] for m in _METRICS} for spec in models
    }
    for idx in _bootstrap_row_indices(group_codes, n_bootstrap, seed):
        for spec in models:
            sample = _metrics_by_repeat(y[idx], oof[spec.name][:, idx])
            for m in _METRICS:
                boot[spec.name][m].append(float(np.mean(sample[m])))

    alpha = 1.0 - confidence_level
    summary_rows = []
    for spec in models:
        row: dict[str, Any] = {"model": spec.name, "family": spec.family}
        for m in _METRICS:
            values = point[spec.name][m]
            row[m] = float(np.mean(values))
            boot_values = np.asarray(boot[spec.name][m])
            if np.isnan(boot_values).all():
                low = high = float("nan")
            else:
                low, high = np.nanquantile(boot_values, [alpha / 2, 1 - alpha / 2])
            row[f"{m}_ci_low"] = float(low)
            row[f"{m}_ci_high"] = float(high)
            row[f"{m}_repeat_std"] = float(np.std(values))
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    baseline_mae = summary.loc[summary["family"] == "baseline", "mae"]
    summary["beats_all_baselines"] = [
        None if family == "baseline" or baseline_mae.empty else bool(mae < baseline_mae.min())
        for family, mae in zip(summary["family"], summary["mae"], strict=True)
    ]
    return pd.DataFrame(repeat_rows), summary


# --- entry points ------------------------------------------------------------


def run_nested_cv(
    data: pd.DataFrame,
    models: Sequence[ModelSpec],
    outer_split_fn: OuterSplitFn,
    *,
    strategy: str,
    target: str = TARGET_COLUMN,
    group_column: str = GROUP_COLUMN,
    n_repeats: int = 10,
    n_inner_splits: int = 5,
    seed: int = 0,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    grouped: bool = True,
    extra_config: Mapping[str, Any] | None = None,
) -> CVResult:
    """Strategy-agnostic nested CV core. outer_split_fn decides which
    rows each outer fold holds out; everything else -- per-fold cloning
    and seeding, inner grouped hyperparameter search on the outer
    training rows only, pooled metrics, condition-bootstrap CIs -- is
    identical for strategies A, B and C, so their numbers are directly
    comparable (though, per Section H, never merged into one figure).

    A row may be held out at most once per repeat; rows never held out
    (e.g. strategy B's training stages) are simply absent from the
    metrics and from CVResult.predictions. Which rows are held out must
    not depend on the seed.

    Deterministic for a given seed: every source of randomness (model
    random_state, inner/outer shuffles, bootstrap) derives from it.
    """
    _validate_inputs(
        data, models, target, group_column, n_repeats, n_inner_splits, confidence_level
    )

    data = data.reset_index(drop=True)
    X = model_features(data, target)
    y = data[target].to_numpy(dtype=float)
    groups = data[group_column].to_numpy()

    rng = np.random.default_rng(seed)
    repeat_seeds = [int(s) for s in rng.integers(0, 2**31 - 1, n_repeats)]
    bootstrap_seed = int(rng.integers(0, 2**31 - 1))

    oof = {spec.name: np.full((n_repeats, len(data)), np.nan) for spec in models}
    tested: np.ndarray | None = None
    fold_rows = []
    for r, repeat_seed in enumerate(repeat_seeds):
        # One set of outer splits per repeat, shared by every model, so
        # models are compared on identical held-out conditions.
        splits = outer_split_fn(data, repeat_seed)
        if not splits:
            raise ValueError(f"strategy {strategy!r} produced no outer splits")
        held_out = np.zeros(len(data), dtype=int)
        for train_idx, test_idx in splits:
            if len(train_idx) == 0 or len(test_idx) == 0:
                raise ValueError(f"strategy {strategy!r} produced an empty train or test fold")
            if np.intersect1d(train_idx, test_idx).size:
                raise ValueError(f"strategy {strategy!r} produced overlapping train/test rows")
            held_out[test_idx] += 1
        if (held_out > 1).any():
            raise ValueError(f"strategy {strategy!r} held out a row more than once per repeat")
        if tested is None:
            tested = held_out == 1
        elif not np.array_equal(tested, held_out == 1):
            raise ValueError(f"strategy {strategy!r} held out different rows across repeats")

        for spec in models:
            for fold, (train_idx, test_idx) in enumerate(splits):
                fitted, best_params = _fit_one_fold(
                    spec,
                    X.iloc[train_idx],
                    y[train_idx],
                    groups[train_idx],
                    grouped,
                    n_inner_splits,
                    repeat_seed,
                )
                oof[spec.name][r, test_idx] = fitted.predict(X.iloc[test_idx])
                fold_rows.append(
                    {
                        "model": spec.name,
                        "repeat": r,
                        "fold": fold,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        "n_test_groups": len(np.unique(groups[test_idx])),
                        "best_params": best_params,
                    }
                )

    assert tested is not None
    tested_rows = np.flatnonzero(tested)
    for spec in models:
        if np.isnan(oof[spec.name][:, tested_rows]).any():
            raise RuntimeError(f"model {spec.name!r} produced NaN out-of-fold predictions")

    tested_group_codes, _ = pd.factorize(groups[tested_rows])
    repeat_metrics, summary = _summarize(
        models,
        y[tested_rows],
        {name: values[:, tested_rows] for name, values in oof.items()},
        tested_group_codes,
        n_bootstrap,
        confidence_level,
        bootstrap_seed,
    )

    predictions = pd.concat(
        [
            pd.DataFrame(
                {
                    "model": spec.name,
                    "repeat": r,
                    "row": tested_rows,
                    "group_id": groups[tested_rows],
                    "y_true": y[tested_rows],
                    "y_pred": oof[spec.name][r, tested_rows],
                }
            )
            for spec in models
            for r in range(n_repeats)
        ],
        ignore_index=True,
    )

    config = {
        "strategy": strategy,
        "grouped": grouped,
        "target": target,
        "group_column": group_column,
        "n_rows": len(data),
        "n_rows_evaluated": int(tested.sum()),
        "n_groups": int(data[group_column].nunique()),
        "n_repeats": n_repeats,
        "n_inner_splits": n_inner_splits,
        "seed": seed,
        "n_bootstrap": n_bootstrap,
        "confidence_level": confidence_level,
        "models": [spec.name for spec in models],
        **(extra_config or {}),
    }
    return CVResult(
        config=config,
        predictions=predictions,
        fold_log=pd.DataFrame(fold_rows),
        repeat_metrics=repeat_metrics,
        summary=summary,
    )


def run_grouped_nested_cv(
    data: pd.DataFrame,
    models: Sequence[ModelSpec],
    *,
    target: str = TARGET_COLUMN,
    group_column: str = GROUP_COLUMN,
    n_repeats: int = 10,
    n_outer_splits: int | None = None,
    n_inner_splits: int = 5,
    seed: int = 0,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    grouped: bool = True,
) -> CVResult:
    """Strategy A (interpolation): evaluate every model under the same
    outer splits and report a comparative table with condition-bootstrap
    CIs.

    data: an ML-ready frame (dataset_builder.build_ml_dataset() shape) --
        one row per device, a measured target, and group_column holding
        each row's experimental_condition_id.
    n_outer_splits: None (default) = LOCO, Section H's primary strategy.
        An integer switches to shuffled grouped k-fold, which is cheaper
        and makes repeats resample the outer split as well.
    grouped: leave True. False exists only for the leakage canary test.
    """
    if group_column in data.columns:
        n_groups = data[group_column].nunique()
        if n_outer_splits is not None and not 2 <= n_outer_splits <= n_groups:
            raise ValueError(f"n_outer_splits must be in [2, {n_groups}] or None (LOCO)")
    return run_nested_cv(
        data,
        models,
        loco_or_grouped_kfold_splits(group_column, n_outer_splits, grouped),
        strategy="A_loco" if n_outer_splits is None else "A_grouped_kfold",
        target=target,
        group_column=group_column,
        n_repeats=n_repeats,
        n_inner_splits=n_inner_splits,
        seed=seed,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        grouped=grouped,
        extra_config={"n_outer_splits": n_outer_splits},
    )


def strategy_comparison_table(results: Sequence[CVResult]) -> pd.DataFrame:
    """Stack the summaries of several runs (typically strategies A, B and
    C) into one table with a leading strategy column. One row per
    (strategy, model): Section H requires the strategies to be reported
    side by side, never averaged into a single accuracy figure."""
    if not results:
        raise ValueError("results must be non-empty")
    frames = [
        result.summary.assign(
            strategy=result.config["strategy"],
            n_rows_evaluated=result.config["n_rows_evaluated"],
        )
        for result in results
    ]
    table = pd.concat(frames, ignore_index=True)
    leading = ["strategy", "model", "family", "n_rows_evaluated"]
    return table[leading + [c for c in table.columns if c not in leading]]
