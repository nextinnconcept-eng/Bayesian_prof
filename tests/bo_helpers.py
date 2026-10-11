"""Shared helpers for the M5-M10 tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dssc_discovery.closed_loop.synthetic_objective import SyntheticObjective
from dssc_discovery.domain.enums import SpeciesCode
from dssc_discovery.features.synthetic import _sample_vector, synthetic_pce
from dssc_discovery.optimization.initial_design import latin_hypercube_design
from dssc_discovery.optimization.search_space import search_space_for

TARGET = "power_conversion_efficiency_percent"
FAST = {"num_restarts": 2, "raw_samples": 32, "fallback_pool_size": 200}
GROUND_TRUTH_FEATURES = [
    "ph",
    "pigment_concentration_mm",
    "adsorption_time_min",
    "tio2_thickness_um",
]


def ground_truth_frame(n: int = 40, seed: int = 3) -> pd.DataFrame:
    """Design features + noisy synthetic_pce target, one row per condition."""
    rng = np.random.default_rng(seed)
    vectors = [_sample_vector(rng) for _ in range(n)]
    frame = pd.DataFrame([v.model_dump(mode="json") for v in vectors])
    frame["species_code"] = SpeciesCode.BETA_VULGARIS.value
    frame["group_id"] = [f"cond-{i}" for i in range(n)]
    frame[TARGET] = [synthetic_pce(v) + rng.normal(0, 0.3) for v in vectors]
    return frame


def objective_frame(
    n: int,
    seed: int,
    design_source: str = "initial_doe",
    species: SpeciesCode = SpeciesCode.BETA_VULGARIS,
) -> pd.DataFrame:
    """LHS points evaluated on the 7.1 synthetic objective, in ML-dataset shape."""
    space = search_space_for(species)
    objective = SyntheticObjective(space, noise_std=0.1)
    rng = np.random.default_rng(seed)
    vectors = latin_hypercube_design(space, n, seed)
    frame = pd.DataFrame([v.model_dump(mode="json") for v in vectors])
    frame["species_code"] = species.value
    frame["design_source"] = design_source
    frame["group_id"] = [f"{design_source}-{seed}-{i}" for i in range(n)]
    frame[TARGET] = [objective.evaluate(v, rng) for v in vectors]
    return frame
