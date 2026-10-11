"""Kernel SHAP wrapper for GPR and linear models (Chunk 5.2). Model-
agnostic, so it is approximate: additivity holds up to Kernel SHAP's
estimation error, which shrinks with nsamples. At N~70-85 rows the cost
is negligible (Section I), so the defaults favor accuracy.

The background set (what "feature absent" means) is a seeded sample of
the explained data's rows, capped at background_size.
"""

from __future__ import annotations

from uuid import UUID

import numpy as np
import pandas as pd
import shap

from dssc_discovery.explainability.base import SHAPExplanationSet, preprocessed
from dssc_discovery.explainability.shap_tree import TREE_MODEL_TYPES, explain_tree_model
from dssc_discovery.models.gpr import GPRModel
from dssc_discovery.models.linear_models import LassoModel, RidgeModel

KERNEL_MODEL_TYPES = (GPRModel, RidgeModel, LassoModel)


def explain_with_kernel(
    model: GPRModel | RidgeModel | LassoModel,
    X: pd.DataFrame,
    *,
    background: pd.DataFrame | None = None,
    background_size: int = 50,
    nsamples: int | str = "auto",
    seed: int = 0,
    model_name: str | None = None,
    model_run_id: UUID | None = None,
) -> SHAPExplanationSet:
    if not isinstance(model, KERNEL_MODEL_TYPES):
        raise TypeError(
            f"explain_with_kernel needs a GPR/linear wrapper, got {type(model).__name__}"
        )
    data, names, estimator = preprocessed(model, X)
    background_frame = X if background is None else background
    background_data, _, _ = preprocessed(model, background_frame)
    if len(background_data) > background_size:
        rng = np.random.default_rng(seed)
        background_data = background_data[
            rng.choice(len(background_data), background_size, replace=False)
        ]

    def predict(matrix: np.ndarray) -> np.ndarray:
        return np.asarray(estimator.predict(matrix), dtype=float)

    explainer = shap.KernelExplainer(predict, background_data, feature_names=names)
    # Kernel SHAP samples coalitions with numpy's global RNG; seed it so
    # the approximation is reproducible.
    np.random.seed(seed)
    values = np.asarray(explainer.shap_values(data, nsamples=nsamples, silent=True), dtype=float)
    return SHAPExplanationSet(
        values=values,
        base_value=float(np.ravel(explainer.expected_value)[0]),
        feature_names=names,
        data=data,
        raw_data=X.reset_index(drop=True),
        explainer="kernel",
        model_name=model_name or type(model).__name__,
        model_run_id=model_run_id,
    )


def explain(model: object, X: pd.DataFrame, **kwargs: object) -> SHAPExplanationSet:
    """Pick the explainer by model family (Section I)."""
    if isinstance(model, TREE_MODEL_TYPES):
        kwargs.pop("background", None)
        kwargs.pop("background_size", None)
        kwargs.pop("nsamples", None)
        kwargs.pop("seed", None)
        return explain_tree_model(model, X, **kwargs)  # type: ignore[arg-type]
    if isinstance(model, KERNEL_MODEL_TYPES):
        return explain_with_kernel(model, X, **kwargs)  # type: ignore[arg-type]
    raise TypeError(f"no SHAP explainer for {type(model).__name__}")
