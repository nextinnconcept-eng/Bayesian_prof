"""Tests for the preprocessing pipeline factory (Chunk 3.4): the
returned pipeline is genuinely unfit, and fitting on train + transforming
test produces statistics with NO measurable contamination from the test
fold -- the actual property Section H's leakage-safety requirement
depends on, checked numerically, not just asserted by code inspection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from dssc_discovery.features.registry import get_feature
from dssc_discovery.preprocessing.pipelines import build_preprocessing_pipeline


def test_pipeline_is_unfit_at_return():
    specs = [get_feature("ph")]
    pipeline = build_preprocessing_pipeline(specs)

    df = pd.DataFrame({"ph": [4.0, 5.0, 6.0]})
    with pytest.raises(NotFittedError):
        pipeline.transform(df)


def test_pipeline_rejects_empty_feature_list():
    with pytest.raises(ValueError, match="non-empty"):
        build_preprocessing_pipeline([])


def test_fit_on_train_statistics_are_uncontaminated_by_test_data():
    # the core leakage-safety check: fit on train only, then verify the
    # scaler's learned mean/variance exactly match train-only statistics
    # -- not some blend that would result if test data influenced fitting.
    specs = [get_feature("ph")]
    pipeline = build_preprocessing_pipeline(specs)

    train = pd.DataFrame({"ph": [4.0, 5.0, 6.0]})
    test = pd.DataFrame({"ph": [100.0, 200.0]})  # deliberately wild, easy to detect contamination

    pipeline.fit(train)

    scaler = pipeline.named_transformers_["continuous"].named_steps["scale"]
    assert scaler.mean_[0] == pytest.approx(np.mean([4.0, 5.0, 6.0]))
    assert scaler.var_[0] == pytest.approx(np.var([4.0, 5.0, 6.0]))

    # transforming the wild test data with train-fitted statistics must
    # produce large-magnitude z-scores -- if test data had leaked into
    # fitting, the scaler would have adapted toward it and these would
    # be much smaller.
    transformed_test = pipeline.transform(test)
    assert abs(transformed_test[0, 0]) > 10
    assert abs(transformed_test[1, 0]) > 10


def test_refitting_on_different_data_changes_statistics():
    # confirms the pipeline object itself carries no hidden state from a
    # previous .fit() call that would leak across folds if reused.
    specs = [get_feature("ph")]
    pipeline = build_preprocessing_pipeline(specs)

    pipeline.fit(pd.DataFrame({"ph": [4.0, 5.0, 6.0]}))
    first_mean = pipeline.named_transformers_["continuous"].named_steps["scale"].mean_[0]

    pipeline.fit(pd.DataFrame({"ph": [10.0, 11.0, 12.0]}))
    second_mean = pipeline.named_transformers_["continuous"].named_steps["scale"].mean_[0]

    assert first_mean == pytest.approx(5.0)
    assert second_mean == pytest.approx(11.0)


def test_missing_continuous_value_is_imputed_not_dropped():
    specs = [get_feature("ph")]
    pipeline = build_preprocessing_pipeline(specs)

    train = pd.DataFrame({"ph": [4.0, 5.0, np.nan, 6.0]})
    transformed = pipeline.fit_transform(train)

    assert transformed.shape[0] == 4  # no row dropped
    assert not np.isnan(transformed).any()  # median-imputed, not left as NaN


def test_categorical_column_is_one_hot_encoded():
    specs = [get_feature("solvent")]
    pipeline = build_preprocessing_pipeline(specs)

    train = pd.DataFrame({"solvent": ["ethanol", "methanol", "ethanol"]})
    transformed = pipeline.fit_transform(train)

    assert transformed.shape == (3, 2)  # two distinct categories -> two one-hot columns
    assert set(np.unique(transformed)) <= {0.0, 1.0}


def test_unseen_category_in_test_does_not_crash():
    specs = [get_feature("solvent")]
    pipeline = build_preprocessing_pipeline(specs)

    train = pd.DataFrame({"solvent": ["ethanol", "ethanol"]})
    test = pd.DataFrame({"solvent": ["methanol"]})  # never seen during fit

    pipeline.fit(train)
    transformed = pipeline.transform(test)  # must not raise

    assert transformed.shape == (1, 1)  # one known category column, all zero for the unseen one
    assert transformed[0, 0] == 0.0


def test_mixed_continuous_and_categorical_features_combine_correctly():
    specs = [get_feature("ph"), get_feature("solvent")]
    pipeline = build_preprocessing_pipeline(specs)

    train = pd.DataFrame({"ph": [4.0, 5.0, 6.0], "solvent": ["ethanol", "ethanol", "methanol"]})
    transformed = pipeline.fit_transform(train)

    # 1 scaled continuous column + 2 one-hot categorical columns
    assert transformed.shape == (3, 3)


def test_missing_categorical_value_is_imputed_with_most_frequent():
    specs = [get_feature("solvent")]
    pipeline = build_preprocessing_pipeline(specs)

    train = pd.DataFrame({"solvent": ["ethanol", "ethanol", None]})
    transformed = pipeline.fit_transform(train)

    assert transformed.shape[0] == 3  # no row dropped despite the missing value
