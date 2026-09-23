"""Mandatory baseline predictors (Chunk 4.1, PLAN.txt line 316-318):
every "real" model in this project's evaluation harness (Chunk 4.4)
must be reported alongside, and beat, both of these -- a model that
can't outperform a mean predictor or a two-feature domain-knowledge
linear fit has learned nothing worth publishing.

"Dye loading" (PLAN.txt's own architecture diagram, line 52: the
SensitizationRun stage is "(adsorption time, dye loading, T)") maps
onto this schema's sensitization_concentration_mm -- the sensitization
bath concentration used during dye adsorption. There is no directly
measured "amount of dye actually adsorbed on the TiO2" field at this
stage of the project (that would require a desorption assay), so the
process parameter is the defensible proxy; this mapping choice is
recorded here explicitly rather than assumed silently.

Design choice, and why it matters for Chunk 4.4: unlike the "real"
model wrappers (4.2), which will each wrap their own
build_preprocessing_pipeline() (3.4) internally and so can accept
whatever feature columns that pipeline was built for, both baselines
here index their input by COLUMN NAME on a raw, un-preprocessed pandas
DataFrame (the same shape build_ml_dataset() produces). Selecting
"lambda_max_nm" and "sensitization_concentration_mm" by name would be
impossible after a shared ColumnTransformer has already run -- one-hot
encoding inflates the column count and StandardScaler's output is a
bare ndarray with no column identity left. So the harness must pass
these two baselines the raw feature DataFrame for a fold, not the
fitted-pipeline output it passes to the other models.

Both estimators accept (but do not use) a random_state constructor
parameter purely so the evaluation harness (4.4) can instantiate every
model -- baseline or "real" -- through one uniform constructor
signature; neither MeanBaseline nor DomainLinearBaseline (ordinary
least squares, closed-form) has any actual source of randomness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

DOMAIN_FEATURE_COLUMNS: tuple[str, str] = ("lambda_max_nm", "sensitization_concentration_mm")


class MeanBaseline(BaseEstimator, RegressorMixin):
    """Predicts the training-set mean PCE for every row, regardless of
    features -- the floor any real model must clear."""

    def __init__(self, random_state: int | None = None) -> None:
        self.random_state = random_state

    def fit(self, X: object, y: pd.Series | np.ndarray) -> MeanBaseline:
        self.mean_ = float(np.mean(np.asarray(y, dtype=float)))
        return self

    def predict(self, X: object) -> np.ndarray:
        check_is_fitted(self, "mean_")
        return np.full(len(X), self.mean_)  # type: ignore[arg-type]


class DomainLinearBaseline(BaseEstimator, RegressorMixin):
    """Ordinary least squares on exactly DOMAIN_FEATURE_COLUMNS (median
    imputation + standardization first, so a missing value on either
    column doesn't crash the fit and the two differently-scaled
    features contribute comparably). X must be a pandas DataFrame
    containing both columns -- selecting them by name, not position, is
    the entire point of a domain-knowledge baseline.
    """

    def __init__(self, random_state: int | None = None) -> None:
        self.random_state = random_state

    def _select(self, X: object) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "DomainLinearBaseline requires a pandas DataFrame so it can select "
                f"{DOMAIN_FEATURE_COLUMNS} by name; got {type(X).__name__}"
            )
        missing = [c for c in DOMAIN_FEATURE_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(f"DomainLinearBaseline: missing required column(s) {missing}")
        return X[list(DOMAIN_FEATURE_COLUMNS)]

    def fit(self, X: object, y: pd.Series | np.ndarray) -> DomainLinearBaseline:
        pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("regress", LinearRegression()),
            ]
        )
        self.pipeline_ = pipeline.fit(self._select(X), y)
        return self

    def predict(self, X: object) -> np.ndarray:
        check_is_fitted(self, "pipeline_")
        return self.pipeline_.predict(self._select(X))
