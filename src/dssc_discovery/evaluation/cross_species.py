"""Validation strategy C -- leave-one-species-out, rotated (Chunk 4.5,
PLAN.txt Section H): each species is held out exactly once while the
model trains on all the others. Answers "has the model learned
transferable chemistry-performance relationships, or species-specific
patterns?" (Paper 7) -- a different question from A and B, reported
separately.

Inner hyperparameter search stays grouped by experimental condition
(run_nested_cv), so replicate leakage can't inflate the inner score
either. species_code may itself be a feature; the preprocessing
pipeline's OneHotEncoder(handle_unknown="ignore") encodes the unseen
held-out species as all-zeros rather than crashing -- which is exactly
the situation this strategy is meant to probe.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from dssc_discovery.evaluation.cv_harness import CVResult, ModelSpec, Split, run_nested_cv

SPECIES_COLUMN = "species_code"


def leave_one_species_out_splits(
    data: pd.DataFrame, species_column: str = SPECIES_COLUMN
) -> list[tuple[str, Split]]:
    """One (held-out species, (train_idx, test_idx)) per species, in
    sorted species order -- every species is the test fold exactly once."""
    if species_column not in data.columns:
        raise ValueError(f"data is missing required column {species_column!r}")
    if data[species_column].isna().any():
        raise ValueError(f"species column {species_column!r} contains NaN")
    species = data[species_column].astype(str).to_numpy()
    unique = sorted(set(species))
    if len(unique) < 2:
        raise ValueError(f"leave-one-species-out needs at least 2 species, got {unique}")
    return [(s, (np.flatnonzero(species != s), np.flatnonzero(species == s))) for s in unique]


def run_cross_species_cv(
    data: pd.DataFrame,
    models: Sequence[ModelSpec],
    *,
    species_column: str = SPECIES_COLUMN,
    **kwargs: Any,
) -> CVResult:
    """Strategy C. Extra keyword arguments go to run_nested_cv."""
    named_splits = leave_one_species_out_splits(data.reset_index(drop=True), species_column)
    splits = [split for _, split in named_splits]
    return run_nested_cv(
        data,
        models,
        lambda _data, _seed: splits,
        strategy="C_leave_one_species_out",
        extra_config={
            "species_column": species_column,
            "held_out_species_order": [s for s, _ in named_splits],
        },
        **kwargs,
    )


def metrics_by_held_out_species(
    result: CVResult, data: pd.DataFrame, species_column: str = SPECIES_COLUMN
) -> pd.DataFrame:
    """MAE/RMSE per (model, held-out species), averaged over repeats --
    Paper 7 needs to see WHICH species transfer poorly, not only the
    pooled figure."""
    species = data.reset_index(drop=True)[species_column].astype(str).to_numpy()
    predictions = result.predictions.assign(species=species[result.predictions["row"]])
    error = predictions["y_pred"] - predictions["y_true"]
    per_repeat = (
        predictions.assign(abs_error=error.abs(), sq_error=error**2)
        .groupby(["model", "species", "repeat"])
        .agg(mae=("abs_error", "mean"), mse=("sq_error", "mean"), n=("row", "size"))
        .reset_index()
    )
    per_repeat["rmse"] = np.sqrt(per_repeat["mse"])
    return (
        per_repeat.groupby(["model", "species"])
        .agg(mae=("mae", "mean"), rmse=("rmse", "mean"), n_rows=("n", "first"))
        .reset_index()
    )


def species_rotations(
    data: pd.DataFrame, species_column: str = SPECIES_COLUMN
) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Chunk 10.1: the "train on the others / hold one out" rotation as
    reusable (held-out species, train frame, test frame) triples, one per
    species -- for any per-rotation analysis beyond CV metrics (e.g.
    species_invariance.py's SHAP comparison)."""
    data = data.reset_index(drop=True)
    return [
        (
            species,
            data.iloc[train_idx].reset_index(drop=True),
            data.iloc[test_idx].reset_index(drop=True),
        )
        for species, (train_idx, test_idx) in leave_one_species_out_splits(data, species_column)
    ]


def rotation_summary(
    result: CVResult, data: pd.DataFrame, species_column: str = SPECIES_COLUMN
) -> pd.DataFrame:
    """Per-model aggregate across all rotations: mean, worst and spread of
    per-held-out-species MAE -- a model that transfers well has a small
    spread, not just a good pooled number."""
    per_species = metrics_by_held_out_species(result, data, species_column)
    return (
        per_species.groupby("model")
        .agg(
            mean_species_mae=("mae", "mean"),
            worst_species_mae=("mae", "max"),
            species_mae_std=("mae", "std"),
            n_rotations=("species", "nunique"),
        )
        .reset_index()
    )
