"""Tests for the feature registry (Chunk 3.1): registry entries match
real DB columns, the budget constants are importable and consistent with
their own documented rationale, and the design/characterization stage
split is exhaustive and non-overlapping -- the split that keeps the BO
surrogate (Chunk 6.x) from ever seeing a feature it couldn't actually
have at recommendation time.
"""

from __future__ import annotations

import pytest

from dssc_discovery.chemistry.uvvis import derive_uvvis_descriptors
from dssc_discovery.database.models import Base
from dssc_discovery.features.registry import (
    FEATURE_BUDGET,
    FEATURE_REGISTRY,
    FeatureSpec,
    characterization_features,
    design_features,
    feature_names,
    get_feature,
)

# Matches this project's planned final dataset scale (PLAN.txt Section 13)
PLANNED_N_CONDITIONS = 70


# --- registry entries match real DB columns ----------------------------------


def test_every_feature_source_field_is_a_real_db_column():
    for feature in FEATURE_REGISTRY:
        table = Base.metadata.tables.get(feature.source_table)
        assert table is not None, f"{feature.name}: no such table {feature.source_table!r}"
        assert feature.source_field in table.columns, (
            f"{feature.name}: {feature.source_table}.{feature.source_field} "
            f"is not a real column"
        )


def test_uvvis_feature_names_match_the_actual_descriptor_names_chunk_2_2_produces(
    tmp_path, monkeypatch
):
    # a real cross-module consistency check: if uvvis.py's descriptor
    # naming ever changes, this registry would silently stop matching
    # real data unless this test catches the drift.
    from uuid import uuid4

    import numpy as np

    from dssc_discovery.chemistry.raw_storage import register_raw_file
    from dssc_discovery.config import clear_config_cache
    from dssc_discovery.domain.enums import MeasurementType

    monkeypatch.setenv("DSSC_DATA_DIR", str(tmp_path))
    clear_config_cache()

    source = tmp_path / "spectrum.csv"
    wavelength = np.arange(400.0, 650.5, 0.5)
    absorbance = 0.85 * np.exp(-((wavelength - 520.0) ** 2) / (2 * 15.0**2))
    lines = ["wavelength_nm,absorbance"]
    lines += [f"{w},{a}" for w, a in zip(wavelength, absorbance)]
    source.write_text("\n".join(lines), encoding="utf-8")

    reference = register_raw_file(source, MeasurementType.UV_VIS, dest_filename="s.csv")
    descriptors = derive_uvvis_descriptors(
        uuid4(), reference.to_reference_string(), operator="J. Doe"
    )
    actual_names = {d.descriptor_name for d in descriptors}

    clear_config_cache()

    registered_uvvis_names = {
        f.name for f in FEATURE_REGISTRY if f.source_table == "derived_descriptors"
    }
    assert registered_uvvis_names <= actual_names


# --- FEATURE_BUDGET ------------------------------------------------------------


def test_feature_budget_importable_and_has_expected_keys():
    assert FEATURE_BUDGET["tree"] == 8
    assert FEATURE_BUDGET["linear_or_gp"] == 14


def test_feature_budget_is_consistent_with_its_own_documented_rationale():
    # tree: ~N/10, linear_or_gp: ~N/5, at the project's planned N~70-85
    assert FEATURE_BUDGET["tree"] <= PLANNED_N_CONDITIONS / 10 + 1
    assert FEATURE_BUDGET["linear_or_gp"] <= PLANNED_N_CONDITIONS / 5 + 1
    assert FEATURE_BUDGET["tree"] < FEATURE_BUDGET["linear_or_gp"]


# --- design vs characterization stage split -----------------------------------


def test_design_and_characterization_partition_the_full_registry_exhaustively():
    design = set(design_features())
    characterization = set(characterization_features())
    assert design | characterization == set(FEATURE_REGISTRY)
    assert design & characterization == set()  # no overlap


def test_bo_relevant_experimental_condition_vector_fields_are_all_design_stage():
    # every field of ExperimentalConditionVector (Chunk 1.2) must be
    # importable by the BO surrogate -- i.e. tagged "design", not
    # "characterization".
    from dssc_discovery.domain.experimental_condition import ExperimentalConditionVector

    design_names = set(feature_names(stage="design"))
    for field_name in ExperimentalConditionVector.model_fields:
        assert field_name in design_names, f"{field_name} must be a design-stage feature"


def test_characterization_features_are_not_importable_as_design_features():
    design_names = set(feature_names(stage="design"))
    for feature in characterization_features():
        assert feature.name not in design_names


# --- registry integrity ---------------------------------------------------------


def test_no_duplicate_feature_names():
    names = [f.name for f in FEATURE_REGISTRY]
    assert len(names) == len(set(names))


def test_every_feature_name_is_a_valid_identifier():
    # features become DataFrame columns eventually (Chunk 3.2) --
    # valid Python identifiers avoid a whole class of downstream bugs.
    for feature in FEATURE_REGISTRY:
        assert feature.name.isidentifier(), f"{feature.name!r} is not a valid identifier"


def test_get_feature_returns_the_matching_spec():
    feature = get_feature("ph")
    assert isinstance(feature, FeatureSpec)
    assert feature.name == "ph"
    assert feature.dtype == "continuous"
    assert feature.stage == "design"


def test_get_feature_raises_for_unknown_name():
    with pytest.raises(KeyError):
        get_feature("not_a_real_feature")


def test_feature_names_with_no_stage_returns_everything():
    assert set(feature_names()) == {f.name for f in FEATURE_REGISTRY}
