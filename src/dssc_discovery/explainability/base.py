"""Shared SHAP output contract (PLAN.txt Section G, SHAPExplanation):
every SHAP-derived artifact carries a mandatory interpretation_caveat, so
the "associational, not causal" disclaimer travels with the numbers
instead of living only in a README (Section I: "No exceptions").

SHAP values are computed in the model's PREPROCESSED feature space
(imputed, standardized, one-hot encoded -- models/base.py), because that
is what the fitted estimator actually sees; feature_names are the
preprocessing pipeline's output names (e.g. "continuous__ph",
"categorical__solvent_ethanol"). raw_data keeps the original rows for
plotting against real units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID

import numpy as np
import pandas as pd

INTERPRETATION_CAVEAT = (
    "SHAP values describe how this fitted model's predictions depend on its "
    "inputs over this dataset. They are associational, not causal: correlated "
    "features share credit arbitrarily, and at N~70-85 conditions estimates are "
    "noisy. Treat findings as hypotheses to test experimentally (HypothesisLog), "
    "never as mechanisms."
)
INTERACTION_CAVEAT = (
    "Interaction SHAP values are LOW-CONFIDENCE at this sample size: pairwise "
    "effects need far more data than main effects to estimate reliably."
)

ExplainerKind = Literal["tree", "kernel"]


@dataclass(frozen=True)
class SHAPExplanation:
    """One (row, feature) SHAP value -- the Section G record shape."""

    row: int
    feature: str
    shap_value: float
    feature_value: float
    base_value: float
    model_run_id: UUID | None
    interpretation_caveat: str = INTERPRETATION_CAVEAT


@dataclass(frozen=True)
class SHAPExplanationSet:
    """SHAP values for a batch of rows from one fitted model."""

    values: np.ndarray  # (n_rows, n_features)
    base_value: float
    feature_names: list[str]
    data: np.ndarray  # preprocessed feature values, same shape as values
    raw_data: pd.DataFrame
    explainer: ExplainerKind
    model_name: str
    model_run_id: UUID | None = None
    interaction_values: np.ndarray | None = None  # (n_rows, n_features, n_features)
    interpretation_caveat: str = INTERPRETATION_CAVEAT
    interaction_caveat: str | None = field(default=None)

    def predictions(self) -> np.ndarray:
        """base_value + row sum: SHAP's additivity identity."""
        return self.base_value + self.values.sum(axis=1)

    def global_importance(self) -> pd.DataFrame:
        importance = np.abs(self.values).mean(axis=0)
        return (
            pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": importance})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
            .assign(interpretation_caveat=self.interpretation_caveat)
        )

    def to_records(self) -> list[SHAPExplanation]:
        return [
            SHAPExplanation(
                row=i,
                feature=name,
                shap_value=float(self.values[i, j]),
                feature_value=float(self.data[i, j]),
                base_value=self.base_value,
                model_run_id=self.model_run_id,
                interpretation_caveat=self.interpretation_caveat,
            )
            for i in range(self.values.shape[0])
            for j, name in enumerate(self.feature_names)
        ]


def preprocessed(model: object, X: pd.DataFrame) -> tuple[np.ndarray, list[str], object]:
    """(transformed X, output feature names, fitted final estimator) for a
    fitted PreprocessingWrappedRegressor (models/base.py)."""
    pipeline = getattr(model, "pipeline_", None)
    if pipeline is None:
        raise ValueError(
            f"{type(model).__name__} is not a fitted preprocessing-wrapped model; "
            "SHAP wrappers explain the 4.2/4.3 model wrappers only"
        )
    names = [spec.name for spec in model.feature_specs]  # type: ignore[attr-defined]
    preprocess = pipeline.named_steps["preprocess"]
    transformed = np.asarray(preprocess.transform(X[names]), dtype=float)
    return transformed, list(preprocess.get_feature_names_out()), pipeline.named_steps["model"]
