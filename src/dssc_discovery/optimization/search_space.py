"""Per-species constrained search spaces (Chunk 6.1, PLAN.txt Section J).

One SearchSpace per species, not a joint space: conditional feasibility
(concentration ranges differ by plant, some treatment combinations are
infeasible) is far easier to encode as independent bounded spaces than
as one space with conditional parameters.

Parameter kinds:
- ContinuousParam: explicit bounds, optional log scaling (strictly
  positive, skewed variables such as concentration), optional discrete
  step (e.g. furnace programs in fixed increments).
- CategoricalParam: a closed choice list (solvent, surface treatment).

Constraints:
- LinearConstraint: sum(coefficient * value) <= rhs in REAL units, on
  non-log parameters only (log scaling would make it non-linear).
- InfeasibleRule: a known-infeasible combination (categorical value
  and/or continuous ranges) excluded at acquisition time, not left for
  the GP to discover the hard way.

Encoding for BoTorch: continuous -> [0, 1] (log-scaled where flagged),
categorical -> integer index; column order is SearchSpace.param_names.
Decoding rounds discrete params to their step -- the only "snapping" the
system does, because a non-integral furnace step is not a real setting.
Everything else is validated, never clamped (validation.py, Chunk 6.6).

DEFAULT_SEARCH_SPACES are PROVISIONAL: they mirror the synthetic
generator's sampling ranges (features/synthetic.py) until the lab
confirms per-species operating windows; constraint/rule lists are empty
until real infeasibilities are known. Edit them here, in one place.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field

import numpy as np

from dssc_discovery.domain.enums import SolventType, SpeciesCode, SurfaceTreatment
from dssc_discovery.domain.experimental_condition import ExperimentalConditionVector


@dataclass(frozen=True)
class ContinuousParam:
    name: str
    low: float
    high: float
    log_scale: bool = False
    step: float | None = None

    def __post_init__(self) -> None:
        if not self.low < self.high:
            raise ValueError(f"{self.name}: low must be < high")
        if self.log_scale and self.low <= 0:
            raise ValueError(f"{self.name}: log_scale needs a strictly positive range")
        if self.step is not None and self.step <= 0:
            raise ValueError(f"{self.name}: step must be positive")

    def encode(self, value: float) -> float:
        if self.log_scale:
            return (math.log(value) - math.log(self.low)) / (
                math.log(self.high) - math.log(self.low)
            )
        return (value - self.low) / (self.high - self.low)

    def decode(self, unit: float) -> float:
        if self.log_scale:
            value = math.exp(math.log(self.low) + unit * (math.log(self.high) - math.log(self.low)))
        else:
            value = self.low + unit * (self.high - self.low)
        if self.step is not None:
            value = self.low + round((value - self.low) / self.step) * self.step
        return float(value)


@dataclass(frozen=True)
class CategoricalParam:
    name: str
    choices: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.choices) < 1 or len(set(self.choices)) != len(self.choices):
            raise ValueError(f"{self.name}: choices must be non-empty and unique")


@dataclass(frozen=True)
class LinearConstraint:
    """sum(coefficients[name] * value) <= rhs, real units."""

    coefficients: Mapping[str, float]
    rhs: float
    description: str = ""


@dataclass(frozen=True)
class InfeasibleRule:
    """A combination known to be infeasible: matches when every listed
    categorical equals its value AND every listed continuous param lies in
    its [low, high] range."""

    description: str
    categorical: Mapping[str, str] = field(default_factory=dict)
    ranges: Mapping[str, tuple[float, float]] = field(default_factory=dict)

    def matches(self, values: Mapping[str, object]) -> bool:
        for name, choice in self.categorical.items():
            if str(values[name]) != choice:
                return False
        for name, (low, high) in self.ranges.items():
            value = float(values[name])  # type: ignore[arg-type]
            if not low <= value <= high:
                return False
        return True


Param = ContinuousParam | CategoricalParam


@dataclass(frozen=True)
class SearchSpace:
    species: SpeciesCode
    params: tuple[Param, ...]
    linear_constraints: tuple[LinearConstraint, ...] = ()
    infeasible_rules: tuple[InfeasibleRule, ...] = ()
    version: str = "provisional-1"

    def __post_init__(self) -> None:
        names = [p.name for p in self.params]
        vector_fields = set(ExperimentalConditionVector.model_fields)
        if set(names) != vector_fields or len(names) != len(vector_fields):
            raise ValueError(
                "a SearchSpace must define every ExperimentalConditionVector field exactly once"
            )
        for constraint in self.linear_constraints:
            for name in constraint.coefficients:
                param = self.param(name)
                if not isinstance(param, ContinuousParam) or param.log_scale:
                    raise ValueError(
                        f"linear constraint on {name!r}: only non-log continuous params allowed"
                    )

    # --- structure --------------------------------------------------------

    @property
    def param_names(self) -> list[str]:
        return [p.name for p in self.params]

    def param(self, name: str) -> Param:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(name)

    @property
    def categorical_dims(self) -> list[int]:
        return [i for i, p in enumerate(self.params) if isinstance(p, CategoricalParam)]

    @property
    def continuous_dims(self) -> list[int]:
        return [i for i, p in enumerate(self.params) if isinstance(p, ContinuousParam)]

    def bounds(self) -> np.ndarray:
        """(2, d) bounds in encoded space."""
        upper = [
            float(len(p.choices) - 1) if isinstance(p, CategoricalParam) else 1.0
            for p in self.params
        ]
        return np.array([[0.0] * len(self.params), upper])

    def categorical_combinations(self) -> list[dict[int, float]]:
        """Every categorical assignment, as BoTorch fixed_features dicts."""
        combos: list[dict[int, float]] = [{}]
        for i in self.categorical_dims:
            param = self.params[i]
            assert isinstance(param, CategoricalParam)
            combos = [{**c, i: float(k)} for c in combos for k in range(len(param.choices))]
        return combos

    # --- encoding ---------------------------------------------------------

    def encode(self, vector: ExperimentalConditionVector) -> np.ndarray:
        values = vector.model_dump(mode="json")
        row = []
        for p in self.params:
            if isinstance(p, CategoricalParam):
                row.append(float(p.choices.index(str(values[p.name]))))
            else:
                row.append(p.encode(float(values[p.name])))
        return np.array(row)

    def encode_many(self, vectors: Sequence[ExperimentalConditionVector]) -> np.ndarray:
        return (
            np.vstack([self.encode(v) for v in vectors])
            if vectors
            else np.empty((0, len(self.params)))
        )

    def decode_values(self, row: Sequence[float] | np.ndarray) -> dict[str, object]:
        """Encoded row -> raw parameter values (NOT validated)."""
        values: dict[str, object] = {}
        for p, unit in zip(self.params, row, strict=True):
            if isinstance(p, CategoricalParam):
                index = int(round(float(unit)))
                values[p.name] = p.choices[min(max(index, 0), len(p.choices) - 1)]
            else:
                values[p.name] = p.decode(float(unit))
        return values

    # --- feasibility ------------------------------------------------------

    def violations(self, values: Mapping[str, object]) -> list[str]:
        """Every reason values is infeasible in this space (empty = feasible)."""
        problems = []
        for p in self.params:
            if p.name not in values:
                problems.append(f"{p.name}: missing")
                continue
            value = values[p.name]
            if isinstance(p, CategoricalParam):
                if str(value) not in p.choices:
                    problems.append(f"{p.name}={value!r} not in {list(p.choices)}")
                continue
            number = float(value)  # type: ignore[arg-type]
            if not math.isfinite(number) or not p.low <= number <= p.high:
                problems.append(f"{p.name}={number:g} outside [{p.low:g}, {p.high:g}]")
            elif p.step is not None:
                steps = (number - p.low) / p.step
                if abs(steps - round(steps)) > 1e-6:
                    problems.append(f"{p.name}={number:g} not on a {p.step:g} step from {p.low:g}")
        if problems:
            return problems
        for constraint in self.linear_constraints:
            total = 0.0
            for n, c in constraint.coefficients.items():
                total += c * float(values[n])  # type: ignore[arg-type]
            if total > constraint.rhs + 1e-9:
                problems.append(
                    "linear constraint violated "
                    f"({constraint.description or constraint.coefficients}): "
                    f"{total:g} > {constraint.rhs:g}"
                )
        for rule in self.infeasible_rules:
            if rule.matches(values):
                problems.append(f"known-infeasible combination: {rule.description}")
        return problems

    def is_feasible(self, values: Mapping[str, object]) -> bool:
        return not self.violations(values)

    def to_vector(self, values: Mapping[str, object]) -> ExperimentalConditionVector:
        return ExperimentalConditionVector.model_validate(dict(values))

    def sample(self, n: int, rng: np.random.Generator, max_tries: int = 100) -> np.ndarray:
        """n feasible points in encoded space by rejection sampling."""
        accepted: list[np.ndarray] = []
        upper = self.bounds()[1]
        for _ in range(max_tries):
            raw = rng.uniform(0.0, 1.0, (max(n, 16) * 4, len(self.params)))
            for i in self.categorical_dims:
                raw[:, i] = rng.integers(0, int(upper[i]) + 1, len(raw))
            for row in raw:
                values = self.decode_values(row)
                if self.is_feasible(values):
                    accepted.append(self.encode(self.to_vector(values)))
                    if len(accepted) == n:
                        return np.vstack(accepted)
        raise RuntimeError(f"could not sample {n} feasible points; constraints too tight?")

    def to_json(self) -> str:
        return json.dumps(
            {
                "species": self.species.value,
                "version": self.version,
                "params": [asdict(p) | {"kind": type(p).__name__} for p in self.params],
                "linear_constraints": [asdict(c) for c in self.linear_constraints],
                "infeasible_rules": [asdict(r) for r in self.infeasible_rules],
            },
            sort_keys=True,
            default=list,
        )


def _provisional_space(species: SpeciesCode) -> SearchSpace:
    return SearchSpace(
        species=species,
        params=(
            CategoricalParam(
                "solvent",
                tuple(
                    s.value
                    for s in (
                        SolventType.WATER,
                        SolventType.ETHANOL,
                        SolventType.METHANOL,
                        SolventType.ACETONE,
                        SolventType.ACIDIFIED_ETHANOL,
                        SolventType.MIXED,
                    )
                ),
            ),
            ContinuousParam("solvent_ratio", 0.5, 1.0),
            ContinuousParam("ph", 2.0, 9.0),
            ContinuousParam("extraction_temperature_c", 20.0, 80.0),
            ContinuousParam("extraction_time_min", 15.0, 180.0),
            ContinuousParam("pigment_concentration_mm", 1.0, 30.0, log_scale=True),
            ContinuousParam("adsorption_time_min", 30.0, 300.0),
            ContinuousParam("sensitization_concentration_mm", 0.1, 2.0, log_scale=True),
            ContinuousParam("sensitization_temperature_c", 20.0, 60.0),
            ContinuousParam("tio2_thickness_um", 5.0, 25.0),
            ContinuousParam("sintering_temperature_c", 400.0, 500.0),
            ContinuousParam("sintering_time_min", 15.0, 60.0, step=5.0),
            CategoricalParam(
                "surface_treatment",
                tuple(
                    s.value
                    for s in (
                        SurfaceTreatment.NONE,
                        SurfaceTreatment.TICL4,
                        SurfaceTreatment.MGO_COATING,
                    )
                ),
            ),
        ),
    )


DEFAULT_SEARCH_SPACES: dict[SpeciesCode, SearchSpace] = {
    species: _provisional_space(species) for species in SpeciesCode
}


def search_space_for(species: SpeciesCode) -> SearchSpace:
    return DEFAULT_SEARCH_SPACES[species]
