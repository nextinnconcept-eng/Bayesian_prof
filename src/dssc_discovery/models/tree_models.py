"""Tree-ensemble model wrappers (Chunk 4.2): RandomForestModel,
GradientBoostingModel (sklearn), XGBoostModel -- consistent
sklearn-API wrappers around build_preprocessing_pipeline() (3.4) +
the underlying estimator, all seeded via random_state, all pluggable
into the evaluation harness (4.4) exactly like the baselines (4.1).
See models/base.py for the shared fit/predict/pipeline-assembly logic.
"""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor

from dssc_discovery.features.registry import FeatureSpec
from dssc_discovery.models.base import PreprocessingWrappedRegressor


class RandomForestModel(PreprocessingWrappedRegressor):
    def __init__(
        self,
        feature_specs: Sequence[FeatureSpec],
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        random_state: int | None = None,
    ) -> None:
        self.feature_specs = feature_specs
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def _make_estimator(self) -> BaseEstimator:
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )


class GradientBoostingModel(PreprocessingWrappedRegressor):
    def __init__(
        self,
        feature_specs: Sequence[FeatureSpec],
        n_estimators: int = 200,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: int | None = None,
    ) -> None:
        self.feature_specs = feature_specs
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    def _make_estimator(self) -> BaseEstimator:
        return GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )


class XGBoostModel(PreprocessingWrappedRegressor):
    def __init__(
        self,
        feature_specs: Sequence[FeatureSpec],
        n_estimators: int = 200,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: int | None = None,
    ) -> None:
        self.feature_specs = feature_specs
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    def _make_estimator(self) -> BaseEstimator:
        return XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
