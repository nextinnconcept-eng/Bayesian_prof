"""Closed-loop claim validation (Chunks 7.4 logic, 8.5): would cycle N+1's
recommendations have been different WITHOUT cycle N's results? The
closed-loop claim is only defensible if the model update demonstrably
changed what the loop recommended -- a single recommend->measure event
isn't a loop.

compare_with_and_without() fits the same surrogate (same space, seed,
acquisition settings) on two datasets -- with and without the stage under
test -- and quantifies the difference between the two recommendation
batches:
- set_distance: mean nearest-neighbour distance between the batches in
  encoded space (0 = identical recommendations);
- prediction_shift: how much the update changed predicted PCE at the
  with-data recommendations;
- materially_different: set_distance above a stated threshold.
Works on real data (8.5) and on simulated cycles (7.4) alike.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dssc_discovery.optimization.acquisition import optimize_acquisition
from dssc_discovery.optimization.search_space import SearchSpace
from dssc_discovery.optimization.surrogate import fit_surrogate

DEFAULT_MATERIAL_THRESHOLD = 0.05


@dataclass(frozen=True)
class RetrospectiveResult:
    held_out_stage: str
    n_rows_with: int
    n_rows_without: int
    set_distance: float
    prediction_shift: float
    materially_different: bool
    threshold: float
    with_recommendations: list[dict[str, object]]
    without_recommendations: list[dict[str, object]]

    def summary(self) -> str:
        verdict = "CHANGED" if self.materially_different else "did NOT materially change"
        return (
            f"Adding stage {self.held_out_stage!r} ({self.n_rows_with - self.n_rows_without} rows) "
            f"{verdict} the next recommendations: mean nearest-neighbour distance "
            f"{self.set_distance:.3f} (threshold {self.threshold}), mean predicted-PCE shift "
            f"{self.prediction_shift:+.2f}%."
        )


def _batch_distance(space: SearchSpace, a: np.ndarray, b: np.ndarray) -> float:
    diff = np.abs(a[:, None, :] - b[None, :, :])
    for i in space.categorical_dims:
        diff[:, :, i] = (diff[:, :, i] > 0.5).astype(float)
    distances = np.sqrt((diff**2).mean(axis=-1))
    return float((distances.min(axis=1).mean() + distances.min(axis=0).mean()) / 2)


def compare_with_and_without(
    data: pd.DataFrame,
    space: SearchSpace,
    held_out_stage: str,
    *,
    batch_size: int = 10,
    seed: int = 0,
    threshold: float = DEFAULT_MATERIAL_THRESHOLD,
    acquisition_options: Mapping[str, int] | None = None,
) -> RetrospectiveResult:
    """data: ML-ready frame including held_out_stage's rows (design_source)."""
    without = data[data["design_source"].astype(str) != held_out_stage]
    if len(without) == len(data):
        raise ValueError(f"no rows from stage {held_out_stage!r} to hold out")

    surrogate_with = fit_surrogate(data, space, seed=seed)
    surrogate_without = fit_surrogate(without, space, seed=seed)
    options = dict(acquisition_options or {})
    batch_with = optimize_acquisition(
        surrogate_with, batch_size=batch_size, seed=seed, **options
    ).candidates
    batch_without = optimize_acquisition(
        surrogate_without, batch_size=batch_size, seed=seed, **options
    ).candidates

    encoded_with = np.vstack([c.encoded for c in batch_with])
    encoded_without = np.vstack([c.encoded for c in batch_without])
    mean_with, _ = surrogate_with.predict_encoded(encoded_with)
    mean_without, _ = surrogate_without.predict_encoded(encoded_with)
    distance = _batch_distance(space, encoded_with, encoded_without)
    return RetrospectiveResult(
        held_out_stage=held_out_stage,
        n_rows_with=len(data),
        n_rows_without=len(without),
        set_distance=distance,
        prediction_shift=float(np.mean(mean_with - mean_without)),
        materially_different=distance > threshold,
        threshold=threshold,
        with_recommendations=[c.vector.model_dump(mode="json") for c in batch_with],
        without_recommendations=[c.vector.model_dump(mode="json") for c in batch_without],
    )
