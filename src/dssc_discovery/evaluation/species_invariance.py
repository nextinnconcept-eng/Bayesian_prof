"""Species-invariant vs species-specific feature importance (Chunk 10.2).

For every leave-one-species-out rotation (10.1), fit the model on the
training species and compute global SHAP importance (5.x). A feature is
TRANSFERABLE if it ranks in the top_k in every rotation, SPECIES-SPECIFIC
otherwise. species_code one-hot columns are excluded -- by construction
they can't transfer to an unseen species. Carries the SHAP
interpretation caveat: consistent importance is still associational.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import clone

from dssc_discovery.evaluation.cross_species import SPECIES_COLUMN, species_rotations
from dssc_discovery.evaluation.cv_harness import TARGET_COLUMN, ModelSpec, model_features
from dssc_discovery.explainability.base import INTERPRETATION_CAVEAT
from dssc_discovery.explainability.shap_kernel import explain


def species_invariance_report(
    data: pd.DataFrame,
    spec: ModelSpec,
    *,
    top_k: int = 5,
    seed: int = 0,
    target: str = TARGET_COLUMN,
    species_column: str = SPECIES_COLUMN,
) -> pd.DataFrame:
    """One row per feature: importance and rank in each rotation, mean
    importance, worst rank, and classification."""
    per_rotation: dict[str, pd.Series] = {}
    for held_out, train, _ in species_rotations(data, species_column):
        model = clone(spec.estimator)
        if "random_state" in model.get_params():
            model.set_params(random_state=seed)
        model.fit(model_features(train, target), train[target])
        explanation = explain(model, model_features(train, target), seed=seed, model_name=spec.name)
        importance = explanation.global_importance().set_index("feature")["mean_abs_shap"]
        importance = importance[~importance.index.str.contains(f"{species_column}_")]
        per_rotation[held_out] = importance

    table = pd.DataFrame(per_rotation).fillna(0.0)
    ranks = table.rank(ascending=False, method="min")
    report = pd.DataFrame(
        {
            "mean_importance": table.mean(axis=1),
            "min_importance": table.min(axis=1),
            "worst_rank": ranks.max(axis=1).astype(int),
            "best_rank": ranks.min(axis=1).astype(int),
        }
    )
    for species in table.columns:
        report[f"importance_without_{species}"] = table[species]
    report["classification"] = [
        "transferable" if worst <= top_k else "species_specific" for worst in report["worst_rank"]
    ]
    report["interpretation_caveat"] = INTERPRETATION_CAVEAT
    return (
        report.sort_values("mean_importance", ascending=False).rename_axis("feature").reset_index()
    )
