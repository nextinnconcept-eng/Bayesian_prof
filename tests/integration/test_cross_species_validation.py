"""M10 (Chunks 10.1-10.4): rotation utility, species-invariance SHAP
analysis, exploratory transfer study, and the cross-species report, on
synthetic multi-species data (pipeline check only)."""

from __future__ import annotations

import pandas as pd
import pytest

from dssc_discovery.domain.enums import SpeciesCode
from dssc_discovery.evaluation.cross_species import (
    rotation_summary,
    run_cross_species_cv,
    species_rotations,
)
from dssc_discovery.evaluation.cv_harness import ModelSpec, default_feature_specs
from dssc_discovery.evaluation.species_invariance import species_invariance_report
from dssc_discovery.explainability.base import INTERPRETATION_CAVEAT
from dssc_discovery.models.baselines import MeanBaseline
from dssc_discovery.models.linear_models import RidgeModel
from dssc_discovery.models.tree_models import RandomForestModel
from dssc_discovery.optimization.cross_species_prior import run_transfer_study
from dssc_discovery.reports.cross_species_report import write_cross_species_report
from tests.bo_helpers import FAST, GROUND_TRUTH_FEATURES, ground_truth_frame

SPECIES = [
    SpeciesCode.BETA_VULGARIS,
    SpeciesCode.THUNBERGIA_ERECTA,
    SpeciesCode.TELFAIRIA_OCCIDENTALIS,
]
SPECS = default_feature_specs(["species_code", *GROUND_TRUTH_FEATURES])


@pytest.fixture(scope="module")
def data():
    frames = []
    for i, species in enumerate(SPECIES):
        frame = ground_truth_frame(15, seed=10 + i)
        frame["species_code"] = species.value
        frame["group_id"] = [f"{species.name}-{j}" for j in range(15)]
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["is_synthetic"] = True
    return combined


def test_rotations_hold_out_each_species_once(data):
    rotations = species_rotations(data)
    assert [held for held, _, _ in rotations] == sorted(s.value for s in SPECIES)
    for held, train, test in rotations:
        assert set(test["species_code"]) == {held}
        assert held not in set(train["species_code"])
        assert len(train) + len(test) == len(data)


def test_rotation_summary_aggregates_all_rotations(data):
    models = [
        ModelSpec("mean", MeanBaseline(), "baseline"),
        ModelSpec("ridge", RidgeModel(feature_specs=SPECS), "linear_or_gp"),
    ]
    result = run_cross_species_cv(data, models, n_repeats=1, n_inner_splits=3, n_bootstrap=20)
    summary = rotation_summary(result, data)
    assert set(summary["model"]) == {"mean", "ridge"}
    assert (summary["n_rotations"] == 3).all()


def test_species_invariance_separates_transferable_features(data):
    spec = ModelSpec("rf", RandomForestModel(feature_specs=SPECS, n_estimators=30), "tree")
    report = species_invariance_report(data, spec, top_k=4, seed=0)

    assert not report["feature"].str.contains("species_code").any()
    assert set(report["classification"]) <= {"transferable", "species_specific"}
    # the shared ground truth depends on these four everywhere
    transferable = set(report.loc[report["classification"] == "transferable", "feature"])
    assert transferable == {f"continuous__{f}" for f in GROUND_TRUTH_FEATURES}
    assert (report["interpretation_caveat"] == INTERPRETATION_CAVEAT).all()


def test_transfer_study_reports_a_finding():
    result = run_transfer_study(
        n_init=6, n_source=15, batch_size=2, seeds=1, acquisition_options=FAST
    )
    assert result.cold_start_regret >= 0 and result.transfer_regret >= 0
    assert "exploratory" in result.summary()


def test_cross_species_report_files(data, tmp_path):
    models = [
        ModelSpec("mean", MeanBaseline(), "baseline"),
        ModelSpec("ridge", RidgeModel(feature_specs=SPECS), "linear_or_gp"),
    ]
    explanatory = ModelSpec("rf", RandomForestModel(feature_specs=SPECS, n_estimators=20), "tree")
    path = write_cross_species_report(data, models, explanatory, tmp_path, n_repeats=1)

    text = path.read_text()
    assert "SYNTHETIC DATA" in text and "Transferable:" in text
    for name in (
        "strategy_c_summary",
        "per_held_out_species",
        "rotation_spread",
        "species_invariance",
    ):
        assert (tmp_path / f"{name}.csv").exists()
