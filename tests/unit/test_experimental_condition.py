import pytest
from pydantic import ValidationError

from dssc_discovery.domain.enums import (
    ClosedLoopState,
    DesignSource,
    SolventType,
    SpeciesCode,
    SurfaceTreatment,
)
from dssc_discovery.domain.experimental_condition import (
    ExperimentalCondition,
    ExperimentalConditionVector,
)
from dssc_discovery.domain.units_registry import CANONICAL_UNITS


def _vector_kwargs(**overrides):
    kwargs = dict(
        solvent=SolventType.ETHANOL,
        solvent_ratio=0.7,
        ph=5.5,
        extraction_temperature_c=25.0,
        extraction_time_min=60.0,
        pigment_concentration_mm=10.0,
        adsorption_time_min=120.0,
        sensitization_concentration_mm=0.5,
        sensitization_temperature_c=25.0,
        tio2_thickness_um=12.0,
        sintering_temperature_c=450.0,
        sintering_time_min=30.0,
        surface_treatment=SurfaceTreatment.NONE,
    )
    kwargs.update(overrides)
    return kwargs


def test_valid_vector_construction():
    vector = ExperimentalConditionVector(**_vector_kwargs())
    assert vector.solvent is SolventType.ETHANOL


def test_vector_round_trips_through_dict():
    vector = ExperimentalConditionVector(**_vector_kwargs())
    restored = ExperimentalConditionVector.model_validate(vector.model_dump())
    assert restored == vector


def test_vector_field_names_are_exactly_the_bo_parameter_set():
    # This is the contract BO reads/writes (PLAN.txt Section G) -- pinning
    # the exact field set here means Chunk 6.x's search space and Chunk
    # 3.x's dataset assembly can rely on it without re-deriving it.
    expected = {
        "solvent",
        "solvent_ratio",
        "ph",
        "extraction_temperature_c",
        "extraction_time_min",
        "pigment_concentration_mm",
        "adsorption_time_min",
        "sensitization_concentration_mm",
        "sensitization_temperature_c",
        "tio2_thickness_um",
        "sintering_temperature_c",
        "sintering_time_min",
        "surface_treatment",
    }
    assert set(ExperimentalConditionVector.model_fields) == expected


def test_vector_unit_bearing_fields_match_units_registry():
    unit_bearing_fields = [
        "extraction_temperature_c",
        "extraction_time_min",
        "pigment_concentration_mm",
        "adsorption_time_min",
        "sensitization_concentration_mm",
        "sensitization_temperature_c",
        "tio2_thickness_um",
        "sintering_temperature_c",
        "sintering_time_min",
    ]
    for field in unit_bearing_fields:
        assert field in CANONICAL_UNITS
        assert field in ExperimentalConditionVector.model_fields


@pytest.mark.parametrize("bad_ph", [-1, 14.5])
def test_vector_rejects_ph_out_of_range(bad_ph):
    with pytest.raises(ValidationError):
        ExperimentalConditionVector(**_vector_kwargs(ph=bad_ph))


@pytest.mark.parametrize("bad_ratio", [-0.1, 1.1])
def test_vector_rejects_solvent_ratio_out_of_unit_interval(bad_ratio):
    with pytest.raises(ValidationError):
        ExperimentalConditionVector(**_vector_kwargs(solvent_ratio=bad_ratio))


def test_vector_rejects_zero_thickness():
    with pytest.raises(ValidationError):
        ExperimentalConditionVector(**_vector_kwargs(tio2_thickness_um=0))


def test_vector_rejects_unknown_categorical_value():
    with pytest.raises(ValidationError):
        ExperimentalConditionVector(**_vector_kwargs(solvent="kerosene"))


def _condition_kwargs(**overrides):
    kwargs = dict(
        species_code=SpeciesCode.BETA_VULGARIS,
        condition_code="BV-COND-001",
        design_source=DesignSource.INITIAL_DOE,
        target_parameters=ExperimentalConditionVector(**_vector_kwargs()),
        operator="J. Doe",
    )
    kwargs.update(overrides)
    return kwargs


def test_valid_experimental_condition_construction():
    condition = ExperimentalCondition(**_condition_kwargs())
    assert condition.state is ClosedLoopState.PROPOSED  # default entry state


def test_experimental_condition_round_trips_through_dict():
    condition = ExperimentalCondition(**_condition_kwargs())
    restored = ExperimentalCondition.model_validate(condition.model_dump())
    assert restored == condition


def test_experimental_condition_rejects_empty_condition_code():
    with pytest.raises(ValidationError):
        ExperimentalCondition(**_condition_kwargs(condition_code=""))


def test_experimental_condition_state_is_settable():
    condition = ExperimentalCondition(**_condition_kwargs(state=ClosedLoopState.APPROVED))
    assert condition.state is ClosedLoopState.APPROVED
