"""Chunk 7.4: retraining on cycle N's results must materially change
cycle N+1's recommendations -- otherwise there is no closed loop."""

from __future__ import annotations

import pandas as pd

from dssc_discovery.closed_loop.real_retrospective import compare_with_and_without
from dssc_discovery.closed_loop.synthetic_objective import SyntheticObjective
from dssc_discovery.domain.enums import SpeciesCode
from dssc_discovery.optimization.search_space import search_space_for
from tests.bo_helpers import FAST, TARGET, objective_frame

SPACE = search_space_for(SpeciesCode.BETA_VULGARIS)


def test_cycle_results_change_the_next_recommendations():
    initial = objective_frame(15, seed=0)
    # "Cycle 1" results near the global optimum: high PCE the initial DoE never saw.
    objective = SyntheticObjective(SPACE)
    best = objective.optimum_vector().model_dump(mode="json")
    cycle = pd.DataFrame([best] * 4)
    cycle["species_code"] = SpeciesCode.BETA_VULGARIS.value
    cycle["design_source"] = "bo_cycle_1"
    cycle["group_id"] = [f"c1-{i}" for i in range(4)]
    cycle[TARGET] = [objective.true_value(objective.optimum_vector())] * 4
    data = pd.concat([initial, cycle], ignore_index=True)

    result = compare_with_and_without(
        data, SPACE, "bo_cycle_1", batch_size=3, seed=0, acquisition_options=FAST
    )

    assert result.n_rows_with - result.n_rows_without == 4
    assert result.materially_different, result.summary()
    assert result.prediction_shift > 0
