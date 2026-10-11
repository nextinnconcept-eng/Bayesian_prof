"""TreeExplainer wrapper for the tree-ensemble models (Chunk 5.1):
RandomForestModel, GradientBoostingModel, XGBoostModel. Exact (path-
dependent) SHAP, so base_value + sum(values) reproduces the model's
prediction up to floating-point error.
"""

from __future__ import annotations

from uuid import UUID

import numpy as np
import pandas as pd
import shap

from dssc_discovery.explainability.base import (
    INTERACTION_CAVEAT,
    SHAPExplanationSet,
    preprocessed,
)
from dssc_discovery.models.tree_models import (
    GradientBoostingModel,
    RandomForestModel,
    XGBoostModel,
)

TREE_MODEL_TYPES = (RandomForestModel, GradientBoostingModel, XGBoostModel)


def explain_tree_model(
    model: RandomForestModel | GradientBoostingModel | XGBoostModel,
    X: pd.DataFrame,
    *,
    model_name: str | None = None,
    model_run_id: UUID | None = None,
    with_interactions: bool = False,
) -> SHAPExplanationSet:
    if not isinstance(model, TREE_MODEL_TYPES):
        raise TypeError(
            f"explain_tree_model needs a tree model wrapper, got {type(model).__name__}"
        )
    data, names, estimator = preprocessed(model, X)
    explainer = shap.TreeExplainer(estimator)
    values = np.asarray(explainer.shap_values(data), dtype=float)
    interactions = (
        np.asarray(explainer.shap_interaction_values(data), dtype=float)
        if with_interactions
        else None
    )
    return SHAPExplanationSet(
        values=values,
        base_value=float(np.ravel(explainer.expected_value)[0]),
        feature_names=names,
        data=data,
        raw_data=X.reset_index(drop=True),
        explainer="tree",
        model_name=model_name or type(model).__name__,
        model_run_id=model_run_id,
        interaction_values=interactions,
        interaction_caveat=INTERACTION_CAVEAT if with_interactions else None,
    )
