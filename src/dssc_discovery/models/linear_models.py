"""Regularized linear model wrappers (Chunk 4.2): RidgeModel,
LassoModel -- consistent sklearn-API wrappers around
build_preprocessing_pipeline() (3.4) + the underlying estimator, all
seeded via random_state, all pluggable into the evaluation harness
(4.4) exactly like the baselines (4.1) and the tree models. See
models/base.py for the shared fit/predict/pipeline-assembly logic.

random_state is accepted and threaded through for interface
consistency with every other model wrapper, even though Ridge/Lasso's
default solvers ("auto"/cyclic coordinate descent) are deterministic
and ignore it -- it only takes effect for the non-default sag/saga/
random solver choices, which these thin wrappers don't otherwise
expose.
"""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.base import BaseEstimator
from sklearn.linear_model import Lasso, Ridge

from dssc_discovery.features.registry import FeatureSpec
from dssc_discovery.models.base import PreprocessingWrappedRegressor


class RidgeModel(PreprocessingWrappedRegressor):
    def __init__(
        self,
        feature_specs: Sequence[FeatureSpec],
        alpha: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        self.feature_specs = feature_specs
        self.alpha = alpha
        self.random_state = random_state

    def _make_estimator(self) -> BaseEstimator:
        return Ridge(alpha=self.alpha, random_state=self.random_state)


class LassoModel(PreprocessingWrappedRegressor):
    def __init__(
        self,
        feature_specs: Sequence[FeatureSpec],
        alpha: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        self.feature_specs = feature_specs
        self.alpha = alpha
        self.random_state = random_state

    def _make_estimator(self) -> BaseEstimator:
        return Lasso(alpha=self.alpha, random_state=self.random_state)
