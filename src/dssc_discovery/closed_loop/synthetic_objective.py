"""Synthetic objective for closed-loop testing (Chunk 7.1).

NOT PHYSICALLY MEANINGFUL -- TEST HARNESS ONLY. It stands in for "the
lab" so the full BO loop can be validated (M7) before real lab time is
risked. Nothing about its shape, optimum or categorical effects is a
claim about DSSC chemistry, and it must never be cited as one.

Shape: over the search space's continuous dimensions normalized to
[0, 1] (u), a global Gaussian peak at GLOBAL_OPTIMUM_U and a lower local
peak at LOCAL_OPTIMUM_U (multimodal, so a greedy optimizer can get
stuck), scaled by multiplicative categorical effects. Bounded in
[0.5, 9.0] "PCE %"; the documented maximum (9.0) is attained at
optimum_vector() (exactly, up to the discrete sintering-time step).
Noise is Gaussian with a seeded generator, so evaluations are
deterministic given seed + noise config.
"""

from __future__ import annotations

import math

import numpy as np

from dssc_discovery.domain.experimental_condition import ExperimentalConditionVector
from dssc_discovery.optimization.search_space import CategoricalParam, ContinuousParam, SearchSpace

MIN_VALUE = 0.5
MAX_VALUE = 9.0

# Continuous coordinates in SearchSpace order (11 continuous dims).
GLOBAL_OPTIMUM_U = np.array([0.7, 0.45, 0.6, 0.35, 0.55, 0.65, 0.4, 0.3, 0.5, 0.6, 0.4])
LOCAL_OPTIMUM_U = np.array([0.2, 0.85, 0.15, 0.8, 0.1, 0.2, 0.85, 0.75, 0.15, 0.1, 0.9])
GLOBAL_WIDTH = 0.06
LOCAL_WIDTH = 0.03
LOCAL_HEIGHT = 0.65

CATEGORICAL_EFFECTS: dict[str, dict[str, float]] = {
    "solvent": {
        "acidified_ethanol": 1.0,
        "ethanol": 0.95,
        "mixed": 0.9,
        "methanol": 0.85,
        "water": 0.75,
        "acetone": 0.7,
    },
    "surface_treatment": {"ticl4": 1.0, "mgo_coating": 0.95, "none": 0.85},
}


class SyntheticObjective:
    def __init__(
        self, space: SearchSpace, noise_std: float = 0.3, optimum_shift: float = 0.0
    ) -> None:
        """optimum_shift moves the global optimum by that amount (in
        normalized units, clipped to [0, 1]) along every continuous
        dimension -- a stand-in for "a related but different species"
        in the exploratory transfer study (10.3)."""
        if len(space.continuous_dims) != len(GLOBAL_OPTIMUM_U):
            raise ValueError("synthetic objective is defined for the 11-continuous-dim space")
        self.space = space
        self.noise_std = noise_std
        self.global_optimum_u = np.clip(GLOBAL_OPTIMUM_U + optimum_shift, 0.0, 1.0)

    def _continuous_u(self, vector: ExperimentalConditionVector) -> np.ndarray:
        encoded = self.space.encode(vector)
        return encoded[self.space.continuous_dims]

    def _categorical_multiplier(self, vector: ExperimentalConditionVector) -> float:
        values = vector.model_dump(mode="json")
        multiplier = 1.0
        for name, effects in CATEGORICAL_EFFECTS.items():
            multiplier *= effects.get(str(values[name]), min(effects.values()))
        return multiplier

    def true_value(self, vector: ExperimentalConditionVector) -> float:
        u = self._continuous_u(vector)
        global_peak = math.exp(-float(np.mean((u - self.global_optimum_u) ** 2)) / GLOBAL_WIDTH)
        local_peak = LOCAL_HEIGHT * math.exp(
            -float(np.mean((u - LOCAL_OPTIMUM_U) ** 2)) / LOCAL_WIDTH
        )
        shape = max(global_peak, local_peak) * self._categorical_multiplier(vector)
        return MIN_VALUE + (MAX_VALUE - MIN_VALUE) * shape

    def evaluate(self, vector: ExperimentalConditionVector, rng: np.random.Generator) -> float:
        noisy = self.true_value(vector) + float(rng.normal(0.0, self.noise_std))
        return float(min(max(noisy, 0.05), 25.0))

    def optimum_vector(self) -> ExperimentalConditionVector:
        values: dict[str, object] = {}
        u = iter(self.global_optimum_u)
        for param in self.space.params:
            if isinstance(param, CategoricalParam):
                effects = CATEGORICAL_EFFECTS[param.name]
                values[param.name] = max(param.choices, key=lambda c: effects.get(c, 0.0))
            else:
                assert isinstance(param, ContinuousParam)
                values[param.name] = param.decode(float(next(u)))
        return self.space.to_vector(values)

    @property
    def optimum_value(self) -> float:
        return MAX_VALUE
