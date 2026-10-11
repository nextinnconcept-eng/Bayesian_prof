"""Residual analysis, overfitting checks and learning curves (Chunk 4.6).
PLAN.txt spec section 4: R² alone is banned as evidence -- every model
evaluation must also show where and how it is wrong.

Everything here consumes a CVResult (4.4/4.5) plus the data it was run
on, so residuals are always OUT-OF-FOLD residuals; in-sample residuals
look flattering at this N and are only ever shown as the "train" side of
the train-vs-CV gap, never on their own.

- residual_frame(): long table of out-of-fold residuals (averaged over
  repeats, per row).
- heteroscedasticity_test(): does |residual| grow with the predicted
  value? Spearman correlation of |residual| vs prediction, with its
  p-value -- rank-based, so a few large PCE values can't drive it.
- overfitting_gap(): in-sample error of a model refit on all rows vs its
  cross-validated error. A large gap is the classic small-N overfitting
  signature.
- learning_curve_frame(): grouped-CV score vs training-set size (in
  conditions, not rows), the honest answer to "would more experiments
  help?" at N~70-85.
- plot_*(): matplotlib Figures, UI-agnostic (the 11.x reports and the
  dashboard decide where they're shown). The Agg backend is used, so
  nothing ever tries to open a window.
- diagnostic_report(): all of the above for one model in one call.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from dssc_discovery.evaluation.cv_harness import (  # noqa: E402
    GROUP_COLUMN,
    TARGET_COLUMN,
    CVResult,
    ModelSpec,
    _seeded_clone,
    model_features,
)


@dataclass(frozen=True)
class HeteroscedasticityResult:
    spearman_rho: float
    p_value: float
    alpha: float
    detected: bool


@dataclass(frozen=True)
class OverfittingGap:
    train_mae: float
    cv_mae: float
    gap: float
    ratio: float


@dataclass(frozen=True)
class DiagnosticReport:
    model: str
    residuals: pd.DataFrame
    heteroscedasticity: HeteroscedasticityResult
    overfitting: OverfittingGap
    learning_curve: pd.DataFrame | None
    figures: dict[str, Figure] = field(default_factory=dict)


def residual_frame(result: CVResult, data: pd.DataFrame, model: str) -> pd.DataFrame:
    """One row per evaluated dataset row: y_true, out-of-fold prediction
    averaged over repeats, residual (prediction - truth), plus that
    row's feature columns for residual-vs-feature plots."""
    predictions = result.predictions[result.predictions["model"] == model]
    if predictions.empty:
        raise ValueError(f"model {model!r} not found in this CVResult")
    per_row = (
        predictions.groupby("row")
        .agg(group_id=("group_id", "first"), y_true=("y_true", "first"), y_pred=("y_pred", "mean"))
        .reset_index()
    )
    per_row["residual"] = per_row["y_pred"] - per_row["y_true"]
    features = data.reset_index(drop=True).iloc[per_row["row"]].reset_index(drop=True)
    features = features.drop(columns=[c for c in per_row.columns if c in features.columns])
    return pd.concat([per_row, features], axis=1)


def heteroscedasticity_test(
    residuals: pd.DataFrame, alpha: float = 0.05
) -> HeteroscedasticityResult:
    """Flags residual spread that changes with the predicted value. A
    positive, significant rho means errors grow with predicted PCE --
    worth knowing before trusting the model's high-PCE predictions,
    which are exactly the ones BO cares about."""
    abs_residual = residuals["residual"].abs().to_numpy()
    predicted = residuals["y_pred"].to_numpy()
    if np.ptp(predicted) == 0 or np.ptp(abs_residual) == 0:
        return HeteroscedasticityResult(float("nan"), 1.0, alpha, False)
    rho, p_value = stats.spearmanr(predicted, abs_residual)
    return HeteroscedasticityResult(float(rho), float(p_value), alpha, bool(p_value < alpha))


def overfitting_gap(
    result: CVResult,
    data: pd.DataFrame,
    spec: ModelSpec,
    seed: int = 0,
) -> OverfittingGap:
    """Refit spec on every row (default hyperparameters, no search) and
    compare its in-sample MAE with the cross-validated MAE in result.
    Diagnostic only: this refit's predictions are never reported as a
    performance figure."""
    target = result.config["target"]
    data = data.reset_index(drop=True)
    model = _seeded_clone(spec.estimator, seed).fit(model_features(data, target), data[target])
    train_mae = float(
        np.mean(np.abs(model.predict(model_features(data, target)) - data[target].to_numpy()))
    )
    cv_mae = float(result.summary.set_index("model").loc[spec.name, "mae"])
    return OverfittingGap(
        train_mae=train_mae,
        cv_mae=cv_mae,
        gap=cv_mae - train_mae,
        ratio=cv_mae / train_mae if train_mae > 0 else float("inf"),
    )


def learning_curve_frame(
    data: pd.DataFrame,
    spec: ModelSpec,
    *,
    target: str = TARGET_COLUMN,
    group_column: str = GROUP_COLUMN,
    train_fractions: Sequence[float] = (0.3, 0.5, 0.7, 0.85, 1.0),
    n_splits: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Grouped k-fold MAE as a function of how many CONDITIONS the model
    trains on. For each outer fold, the training conditions are
    subsampled (whole conditions, so replicates stay together) to each
    fraction; train and validation MAE are averaged over folds."""
    data = data.reset_index(drop=True)
    X = model_features(data, target)
    y = data[target].to_numpy(dtype=float)
    groups = data[group_column].to_numpy()
    n_splits = min(n_splits, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)

    rows = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
        train_groups = np.unique(groups[train_idx])
        order = rng.permutation(train_groups)
        for fraction in train_fractions:
            n_groups = max(2, int(round(fraction * len(train_groups))))
            chosen = set(order[:n_groups])
            sub_idx = train_idx[np.isin(groups[train_idx], list(chosen))]
            model = _seeded_clone(spec.estimator, seed).fit(X.iloc[sub_idx], y[sub_idx])
            rows.append(
                {
                    "fold": fold,
                    "train_fraction": fraction,
                    "n_train_groups": n_groups,
                    "n_train_rows": len(sub_idx),
                    "train_mae": float(
                        np.mean(np.abs(model.predict(X.iloc[sub_idx]) - y[sub_idx]))
                    ),
                    "validation_mae": float(
                        np.mean(np.abs(model.predict(X.iloc[test_idx]) - y[test_idx]))
                    ),
                }
            )
    folds = pd.DataFrame(rows)
    return (
        folds.groupby("train_fraction")
        .agg(
            n_train_groups=("n_train_groups", "mean"),
            n_train_rows=("n_train_rows", "mean"),
            train_mae=("train_mae", "mean"),
            validation_mae=("validation_mae", "mean"),
            validation_mae_std=("validation_mae", "std"),
        )
        .reset_index()
    )


# --- plots -------------------------------------------------------------------


def plot_residuals_vs_predicted(residuals: pd.DataFrame, title: str = "") -> Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(residuals["y_pred"], residuals["residual"], s=18, alpha=0.7)
    ax.axhline(0.0, color="grey", linewidth=1, linestyle="--")
    ax.set_xlabel("Out-of-fold predicted PCE (%)")
    ax.set_ylabel("Residual (predicted - measured)")
    ax.set_title(title or "Residuals vs predicted")
    fig.tight_layout()
    return fig


def plot_residuals_vs_feature(residuals: pd.DataFrame, feature: str, title: str = "") -> Figure:
    if feature not in residuals.columns:
        raise ValueError(f"feature {feature!r} not in residual frame")
    fig, ax = plt.subplots(figsize=(5, 4))
    values = residuals[feature]
    if pd.api.types.is_numeric_dtype(values):
        ax.scatter(values, residuals["residual"], s=18, alpha=0.7)
    else:
        categories = sorted(values.dropna().astype(str).unique())
        ax.boxplot(
            [residuals.loc[values.astype(str) == c, "residual"] for c in categories],
            tick_labels=categories,
        )
        ax.tick_params(axis="x", rotation=45)
    ax.axhline(0.0, color="grey", linewidth=1, linestyle="--")
    ax.set_xlabel(feature)
    ax.set_ylabel("Residual (predicted - measured)")
    ax.set_title(title or f"Residuals vs {feature}")
    fig.tight_layout()
    return fig


def plot_learning_curve(curve: pd.DataFrame, title: str = "") -> Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(curve["n_train_groups"], curve["train_mae"], marker="o", label="train")
    ax.errorbar(
        curve["n_train_groups"],
        curve["validation_mae"],
        yerr=curve["validation_mae_std"].fillna(0.0),
        marker="o",
        capsize=3,
        label="validation (grouped CV)",
    )
    ax.set_xlabel("Training conditions")
    ax.set_ylabel("MAE (PCE %)")
    ax.set_title(title or "Learning curve")
    ax.legend()
    fig.tight_layout()
    return fig


def diagnostic_report(
    result: CVResult,
    data: pd.DataFrame,
    spec: ModelSpec,
    *,
    features: Sequence[str] = (),
    include_learning_curve: bool = True,
    seed: int = 0,
    alpha: float = 0.05,
) -> DiagnosticReport:
    """The standard per-model diagnostic bundle every ModelRun can
    produce (Chunk 4.6 DoD). features: columns to plot residuals
    against; defaults to the model's own feature_specs when it has any.
    Callers own the returned Figures (plt.close() them when done)."""
    residuals = residual_frame(result, data, spec.name)
    if not features:
        specs = spec.estimator.get_params().get("feature_specs") or []
        features = [s.name for s in specs]

    figures: dict[str, Figure] = {
        "residuals_vs_predicted": plot_residuals_vs_predicted(residuals, f"{spec.name}: residuals")
    }
    for feature in features:
        if feature in residuals.columns:
            figures[f"residuals_vs_{feature}"] = plot_residuals_vs_feature(residuals, feature)

    curve: pd.DataFrame | None = None
    if include_learning_curve:
        curve = learning_curve_frame(
            data,
            spec,
            target=result.config["target"],
            group_column=result.config["group_column"],
            seed=seed,
        )
        figures["learning_curve"] = plot_learning_curve(curve, f"{spec.name}: learning curve")

    return DiagnosticReport(
        model=spec.name,
        residuals=residuals,
        heteroscedasticity=heteroscedasticity_test(residuals, alpha),
        overfitting=overfitting_gap(result, data, spec, seed),
        learning_curve=curve,
        figures=figures,
    )
