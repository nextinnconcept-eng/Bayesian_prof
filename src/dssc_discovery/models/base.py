"""Shared fit/predict/pipeline-assembly logic for the "real" model
wrappers (Chunk 4.2): each wraps its own preprocessing pipeline (3.4)
internally, so -- like the baselines (4.1) -- it can be fit/predicted
directly on the raw feature DataFrame build_ml_dataset() (3.2)
produces. That's the uniform interface Chunk 4.4's evaluation harness
needs to treat every model, baseline or "real", identically: one
fit(X_raw_df, y) / predict(X_raw_df) contract regardless of what each
model does internally to get from raw features to a prediction.

Subclasses declare their own hyperparameters as plain __init__
attributes (required for sklearn's get_params()/clone() introspection,
which reads each class's own __init__ signature -- a base-class
__init__ here would hide subclass-specific params from it) and
implement _make_estimator() to build a fresh, unfit sklearn/xgboost
estimator from those attributes. This base class owns only fit(),
predict(), and the pipeline assembly in between -- never a subclass's
hyperparameters.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from dssc_discovery.features.registry import FeatureSpec
from dssc_discovery.preprocessing.pipelines import build_preprocessing_pipeline


class PreprocessingWrappedRegressor(BaseEstimator, RegressorMixin):
    """Not instantiated directly -- concrete subclasses (tree_models.py,
    linear_models.py) provide __init__ (setting feature_specs,
    random_state, and their own hyperparameters) and _make_estimator().
    """

    feature_specs: Sequence[FeatureSpec]
    random_state: int | None

    def _make_estimator(self) -> BaseEstimator:
        raise NotImplementedError

    def _feature_names(self) -> list[str]:
        return [spec.name for spec in self.feature_specs]

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> PreprocessingWrappedRegressor:
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessing_pipeline(list(self.feature_specs))),
                ("model", self._make_estimator()),
            ]
        )
        self.pipeline_ = pipeline.fit(X[self._feature_names()], y)
        return self

    def predict(self, X: pd.DataFrame, **predict_params: object) -> np.ndarray:
        # **predict_params forwards to the final pipeline step's own
        # predict() (e.g. GPRModel's return_std=True) -- confirmed
        # empirically that sklearn.pipeline.Pipeline.predict() forwards
        # keyword params to the last step unchanged when metadata
        # routing is disabled (this project's default, unset config).
        check_is_fitted(self, "pipeline_")
        return self.pipeline_.predict(X[self._feature_names()], **predict_params)
