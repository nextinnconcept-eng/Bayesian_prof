"""Preprocessing pipeline factory: builds an UNFIT sklearn
ColumnTransformer for a given set of candidate features (Chunk 3.1),
one impute+scale sub-pipeline for continuous columns and one
impute+encode sub-pipeline for categorical columns.

The factory returns immediately without touching any data. This is the
entire point (PLAN.txt Section H): preprocessing statistics (imputation
values, scaling mean/std, one-hot categories) must be fit ONLY on a CV
fold's training data, never on the full dataset before splitting --
fitting globally leaks test-fold information into training-time
statistics and optimistically biases every downstream metric. The
evaluation harness (Chunk 4.4) is the only code in this project that
should ever call .fit()/.fit_transform() on a pipeline this factory
returns; calling .transform() on a fresh pipeline correctly raises
sklearn's own NotFittedError (verified empirically before writing this
module), which is the structural enforcement -- there's no way to use
an unfit pipeline by accident and get silently wrong numbers.

Continuous columns: median imputation (robust to outliers at this
sample size) + standard scaling. Categorical columns: most-frequent
imputation + one-hot encoding with handle_unknown="ignore" -- at
N~70-85 with grouped/LOCO CV (Chunk 4.4), a held-out fold can plausibly
contain the only occurrence of a rare category, and the pipeline must
not crash when that happens (it encodes as all-zeros for that column
instead).

One unified recipe for every model family (tree ensembles don't
strictly need scaling, but it doesn't hurt them either) -- simpler to
maintain than per-family variants, and defensible at this project's
scale.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dssc_discovery.features.registry import FeatureSpec


def build_preprocessing_pipeline(feature_specs: list[FeatureSpec]) -> ColumnTransformer:
    """Build an unfit ColumnTransformer for the given features. Raises
    ValueError if feature_specs is empty (nothing to build a transformer
    for) -- an explicit failure, not a silently useless empty transformer.
    """
    if not feature_specs:
        raise ValueError("feature_specs must be non-empty")

    continuous_names = [f.name for f in feature_specs if f.dtype == "continuous"]
    categorical_names = [f.name for f in feature_specs if f.dtype == "categorical"]

    transformers = []
    if continuous_names:
        continuous_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("continuous", continuous_pipeline, continuous_names))

    if categorical_names:
        categorical_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_names))

    return ColumnTransformer(transformers=transformers, remainder="drop")
