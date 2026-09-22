import pytest
from pydantic import ValidationError

from dssc_discovery.domain.enums import SpeciesCode
from dssc_discovery.domain.species import Species


def test_valid_species_construction():
    species = Species(code=SpeciesCode.BETA_VULGARIS, common_name="beetroot", operator="J. Doe")
    assert species.code is SpeciesCode.BETA_VULGARIS
    assert species.is_synthetic is False
    assert species.id is not None


def test_species_round_trips_through_dict():
    species = Species(code=SpeciesCode.CHROMOLAENA_ODORATA, operator="J. Doe")
    restored = Species.model_validate(species.model_dump())
    assert restored == species


def test_species_requires_operator():
    with pytest.raises(ValidationError, match="operator"):
        Species(code=SpeciesCode.BETA_VULGARIS)


def test_species_rejects_empty_operator():
    with pytest.raises(ValidationError):
        Species(code=SpeciesCode.BETA_VULGARIS, operator="")


def test_species_rejects_unknown_code():
    with pytest.raises(ValidationError):
        Species(code="Not a real species", operator="J. Doe")


def test_species_rejects_unexpected_field():
    with pytest.raises(ValidationError):
        Species(code=SpeciesCode.BETA_VULGARIS, operator="J. Doe", made_up_field=1)
