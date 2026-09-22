import pytest
from pydantic import BaseModel

from dssc_discovery.domain.enums import (
    ClosedLoopState,
    CounterElectrodeType,
    DesignSource,
    ElectrolyteType,
    EquivalentCircuitModel,
    IlluminationProtocol,
    MeasurementType,
    MissingValueReason,
    PigmentClass,
    PlantPart,
    SolventType,
    SpeciesCode,
    SurfaceTreatment,
)

ALL_ENUMS = [
    SpeciesCode,
    PlantPart,
    PigmentClass,
    SolventType,
    SurfaceTreatment,
    CounterElectrodeType,
    ElectrolyteType,
    EquivalentCircuitModel,
    IlluminationProtocol,
    MeasurementType,
    MissingValueReason,
    DesignSource,
    ClosedLoopState,
]


@pytest.mark.parametrize("enum_cls", ALL_ENUMS)
def test_enum_is_nonempty_and_values_are_str(enum_cls):
    members = list(enum_cls)
    assert len(members) > 0
    for member in members:
        assert isinstance(member.value, str)
        assert member == member.value  # StrEnum: member compares equal to its raw string


def test_species_code_covers_all_four_scoped_species():
    values = {member.value for member in SpeciesCode}
    assert values == {
        "Telfairia occidentalis",
        "Thunbergia erecta",
        "Beta vulgaris",
        "Chromolaena odorata",
    }


def test_closed_loop_state_covers_full_state_machine():
    names = {member.name for member in ClosedLoopState}
    assert names == {
        "PROPOSED",
        "APPROVED",
        "FABRICATED",
        "MEASURED",
        "VALIDATED",
        "INGESTED",
        "MODEL_UPDATED",
        "REJECTED",
        "FAILED_FABRICATION",
        "FAILED_MEASUREMENT",
        "SUPERSEDED",
    }


def test_species_code_and_closed_loop_state_have_no_other_escape_hatch():
    assert "OTHER" not in {member.name for member in SpeciesCode}
    assert "OTHER" not in {member.name for member in ClosedLoopState}
    assert "OTHER" not in {member.name for member in DesignSource}


@pytest.mark.parametrize("enum_cls", [PlantPart, PigmentClass, SolventType, SurfaceTreatment])
def test_lab_entry_enums_have_other_escape_hatch(enum_cls):
    assert "OTHER" in {member.name for member in enum_cls}


def test_pydantic_round_trip_via_json():
    class Sample(BaseModel):
        species: SpeciesCode
        state: ClosedLoopState

    original = Sample(species=SpeciesCode.BETA_VULGARIS, state=ClosedLoopState.PROPOSED)
    restored = Sample.model_validate_json(original.model_dump_json())

    assert restored.species is SpeciesCode.BETA_VULGARIS
    assert restored.state is ClosedLoopState.PROPOSED
    assert original.model_dump() == restored.model_dump()


def test_pydantic_rejects_unrecognized_enum_value():
    class Sample(BaseModel):
        species: SpeciesCode

    with pytest.raises(Exception):
        Sample(species="Not a real species")
