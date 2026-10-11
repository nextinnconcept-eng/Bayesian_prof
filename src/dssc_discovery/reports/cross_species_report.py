"""Cross-species validation report (Chunk 10.4): strategy C metrics for
every model, per-held-out-species breakdown, rotation spread, and the
species-invariance table for one explanatory model -- written as CSVs
plus a short markdown summary. Meaningful only on real data once >= 3
species have enough real conditions; on synthetic data it only proves
the pipeline runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from dssc_discovery.evaluation.cross_species import (
    metrics_by_held_out_species,
    rotation_summary,
    run_cross_species_cv,
)
from dssc_discovery.evaluation.cv_harness import ModelSpec
from dssc_discovery.evaluation.species_invariance import species_invariance_report


def _table(frame: pd.DataFrame) -> str:
    return "```\n" + frame.to_string(index=False, float_format=lambda v: f"{v:.3f}") + "\n```"


def write_cross_species_report(
    data: pd.DataFrame,
    models: Sequence[ModelSpec],
    explanatory_model: ModelSpec,
    out_dir: Path,
    *,
    n_repeats: int = 5,
    seed: int = 0,
    top_k: int = 5,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_cross_species_cv(data, models, n_repeats=n_repeats, seed=seed)
    per_species = metrics_by_held_out_species(result, data)
    rotations = rotation_summary(result, data)
    invariance = species_invariance_report(data, explanatory_model, top_k=top_k, seed=seed)

    result.summary.to_csv(out_dir / "strategy_c_summary.csv", index=False)
    per_species.to_csv(out_dir / "per_held_out_species.csv", index=False)
    rotations.to_csv(out_dir / "rotation_spread.csv", index=False)
    invariance.to_csv(out_dir / "species_invariance.csv", index=False)

    synthetic = bool(data["is_synthetic"].any()) if "is_synthetic" in data.columns else False
    transferable = invariance.loc[invariance["classification"] == "transferable", "feature"]
    lines = [
        "# Cross-species validation (strategy C)",
        "",
        "**SYNTHETIC DATA -- pipeline check only, not a scientific result.**" if synthetic else "",
        f"Species rotated: {', '.join(result.config['held_out_species_order'])}",
        "",
        "## Pooled metrics",
        _table(result.summary[["model", "mae", "mae_ci_low", "mae_ci_high", "rmse", "r2"]]),
        "",
        "## Spread across held-out species",
        _table(rotations),
        "",
        f"## Feature importance across rotations ({explanatory_model.name}, top {top_k})",
        f"Transferable: {', '.join(transferable) or 'none'}",
        "",
        f"> {invariance['interpretation_caveat'].iloc[0]}" if not invariance.empty else "",
    ]
    path = out_dir / "cross_species_report.md"
    path.write_text("\n".join(lines))
    return path
