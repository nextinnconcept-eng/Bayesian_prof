"""Tests for validation strategy C, leave-one-species-out (Chunk 4.5):
the rotation holds out every species exactly once, never mixes the
held-out species into training, and the runner plus the per-species
breakdown cover every row."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dssc_discovery.evaluation.cross_species import (
    leave_one_species_out_splits,
    metrics_by_held_out_species,
    run_cross_species_cv,
)
from dssc_discovery.evaluation.cv_harness import ModelSpec, default_feature_specs
from dssc_discovery.models.baselines import MeanBaseline
from dssc_discovery.models.linear_models import RidgeModel

TARGET = "power_conversion_efficiency_percent"
SPECIES = ("Beta vulgaris", "Hibiscus sabdariffa", "Telfairia occidentalis")


def _multi_species_frame(conditions_per_species=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for species in SPECIES:
        for c in range(conditions_per_species):
            ph = rng.uniform(2, 9)
            for r in range(2):
                rows.append(
                    {
                        "group_id": f"{species}-{c}",
                        "species_code": species,
                        "ph": ph,
                        TARGET: 2.0 + 0.5 * ph + rng.normal(0, 0.05),
                    }
                )
    return pd.DataFrame(rows)


def test_rotates_through_every_species_exactly_once():
    data = _multi_species_frame()
    splits = leave_one_species_out_splits(data)

    assert [species for species, _ in splits] == sorted(SPECIES)
    held_out_counts = np.zeros(len(data), dtype=int)
    for species, (train_idx, test_idx) in splits:
        assert set(data["species_code"].iloc[test_idx]) == {species}
        assert species not in set(data["species_code"].iloc[train_idx])
        assert len(train_idx) + len(test_idx) == len(data)
        held_out_counts[test_idx] += 1
    assert (held_out_counts == 1).all()


def test_rejects_a_single_species():
    data = _multi_species_frame()
    data["species_code"] = SPECIES[0]
    with pytest.raises(ValueError, match="at least 2 species"):
        leave_one_species_out_splits(data)


def test_runner_and_per_species_breakdown_cover_every_row():
    data = _multi_species_frame()
    models = [
        ModelSpec("mean", MeanBaseline(), "baseline"),
        ModelSpec("ridge", RidgeModel(feature_specs=default_feature_specs(["ph"])), "linear_or_gp"),
    ]

    result = run_cross_species_cv(data, models, n_repeats=2, n_inner_splits=3, n_bootstrap=50)

    assert result.config["strategy"] == "C_leave_one_species_out"
    assert result.config["held_out_species_order"] == sorted(SPECIES)
    assert result.config["n_rows_evaluated"] == len(data)
    assert len(result.fold_log) == 2 * 2 * len(SPECIES)  # models x repeats x species

    breakdown = metrics_by_held_out_species(result, data)
    assert set(breakdown["species"]) == set(SPECIES)
    assert (breakdown.groupby("model")["n_rows"].sum() == len(data)).all()
