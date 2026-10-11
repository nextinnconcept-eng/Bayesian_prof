"""Per-species search space (6.1), initial design (6.4) and hard
rejection (6.6): sampled/generated points always satisfy bounds and
constraints, known-infeasible combinations are excluded, and an injected
infeasible candidate is rejected with reasons, never coerced."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from dssc_discovery.domain.enums import SpeciesCode
from dssc_discovery.optimization.initial_design import (
    latin_hypercube_design,
    min_pairwise_distance,
)
from dssc_discovery.optimization.search_space import (
    DEFAULT_SEARCH_SPACES,
    ContinuousParam,
    InfeasibleRule,
    LinearConstraint,
    SearchSpace,
    search_space_for,
)
from dssc_discovery.optimization.validation import (
    InfeasibleRecommendationError,
    validate_candidate,
)

BASE = search_space_for(SpeciesCode.BETA_VULGARIS)
RULE = InfeasibleRule(
    "TiCl4 needs >= 420 C sintering (test rule)",
    categorical={"surface_treatment": "ticl4"},
    ranges={"sintering_temperature_c": (400.0, 420.0)},
)
CONSTRAINT = LinearConstraint(
    {"extraction_temperature_c": 1.0, "sensitization_temperature_c": 1.0}, 100.0, "total heat"
)
CONSTRAINED = replace(BASE, infeasible_rules=(RULE,), linear_constraints=(CONSTRAINT,))


def _values(space, encoded):
    return [space.decode_values(row) for row in encoded]


def test_one_valid_space_per_species():
    assert set(DEFAULT_SEARCH_SPACES) == set(SpeciesCode)
    for space in DEFAULT_SEARCH_SPACES.values():
        assert len(space.categorical_combinations()) == 18


def test_sampled_points_satisfy_bounds_steps_and_constraints():
    encoded = CONSTRAINED.sample(300, np.random.default_rng(0))
    for values in _values(CONSTRAINED, encoded):
        assert CONSTRAINED.violations(values) == []
        assert values["sintering_time_min"] % 5 == 0
        assert not RULE.matches(values)
        assert values["extraction_temperature_c"] + values["sensitization_temperature_c"] <= 100


def test_encode_decode_round_trips():
    for row in BASE.sample(20, np.random.default_rng(1)):
        vector = BASE.to_vector(BASE.decode_values(row))
        np.testing.assert_allclose(BASE.encode(vector), row, atol=1e-9)


def test_known_infeasible_combination_is_reported():
    values = BASE.decode_values(BASE.sample(1, np.random.default_rng(2))[0])
    values.update(surface_treatment="ticl4", sintering_temperature_c=410.0)
    assert any("known-infeasible" in v for v in CONSTRAINED.violations(values))


def test_injected_infeasible_candidate_is_rejected_not_coerced():
    values = BASE.decode_values(BASE.sample(1, np.random.default_rng(3))[0])
    values["ph"] = 12.5
    values["sintering_time_min"] = 17.0
    before = dict(values)

    with pytest.raises(InfeasibleRecommendationError) as excinfo:
        validate_candidate(BASE, values)
    assert any(r.startswith("ph=12.5 outside") for r in excinfo.value.reasons)
    assert any("not on a 5 step" in r for r in excinfo.value.reasons)
    assert values == before


def test_space_construction_guards():
    with pytest.raises(ValueError, match="every ExperimentalConditionVector field"):
        SearchSpace(species=SpeciesCode.BETA_VULGARIS, params=BASE.params[:-1])
    with pytest.raises(ValueError, match="only non-log continuous"):
        replace(
            BASE, linear_constraints=(LinearConstraint({"pigment_concentration_mm": 1.0}, 5.0),)
        )
    with pytest.raises(ValueError):
        ContinuousParam("x", 0.0, 1.0, log_scale=True)


def test_initial_design_is_feasible_deterministic_and_space_filling():
    design = latin_hypercube_design(CONSTRAINED, 40, seed=0)
    assert len(design) == 40
    for vector in design:
        assert CONSTRAINED.violations(vector.model_dump(mode="json")) == []
    assert latin_hypercube_design(CONSTRAINED, 40, seed=0) == design

    lhs_distance = min_pairwise_distance(CONSTRAINED, design)
    random_distances = [
        min_pairwise_distance(
            CONSTRAINED,
            [
                CONSTRAINED.to_vector(v)
                for v in _values(CONSTRAINED, CONSTRAINED.sample(40, np.random.default_rng(s)))
            ],
        )
        for s in range(5)
    ]
    assert lhs_distance > np.median(random_distances)
