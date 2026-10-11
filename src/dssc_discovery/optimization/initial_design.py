"""Constrained space-filling initial design (Chunk 6.4, Section J): a
Latin Hypercube over every search-space dimension (categoricals are
stratified too: each LHS column is cut into equal bins, one per choice),
with infeasible points replaced by greedy maximin selection from a large
feasible LHS pool -- so constraints are respected without collapsing the
design onto a narrow region.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from dssc_discovery.domain.experimental_condition import ExperimentalConditionVector
from dssc_discovery.optimization.search_space import CategoricalParam, SearchSpace


def _lhs_points(space: SearchSpace, n: int, seed: int) -> np.ndarray:
    unit = qmc.LatinHypercube(d=len(space.params), seed=seed).random(n)
    encoded = unit.copy()
    for i, param in enumerate(space.params):
        if isinstance(param, CategoricalParam):
            encoded[:, i] = np.minimum(
                np.floor(unit[:, i] * len(param.choices)), len(param.choices) - 1
            )
    return encoded


def _feasible_encoded(space: SearchSpace, points: np.ndarray) -> np.ndarray:
    kept = []
    for row in points:
        values = space.decode_values(row)
        if space.is_feasible(values):
            kept.append(space.encode(space.to_vector(values)))
    return np.vstack(kept) if kept else np.empty((0, len(space.params)))


def _distance_matrix(a: np.ndarray, b: np.ndarray, space: SearchSpace) -> np.ndarray:
    """Distances in encoded space with categoricals as 0/1 mismatch."""
    diff = np.abs(a[:, None, :] - b[None, :, :])
    for i in space.categorical_dims:
        diff[:, :, i] = (diff[:, :, i] > 0.5).astype(float)
    return np.sqrt((diff**2).sum(axis=-1))


def latin_hypercube_design(
    space: SearchSpace, n: int = 40, seed: int = 0, pool_factor: int = 50
) -> list[ExperimentalConditionVector]:
    if n < 1:
        raise ValueError("n must be >= 1")
    design = _feasible_encoded(space, _lhs_points(space, n, seed))
    if len(design) < n:
        pool = _feasible_encoded(space, _lhs_points(space, n * pool_factor, seed + 1))
        if len(design) + len(pool) < n:
            raise RuntimeError("not enough feasible points; constraints too tight?")
        while len(design) < n:
            if len(design) == 0:
                choice = 0
            else:
                choice = int(np.argmax(_distance_matrix(pool, design, space).min(axis=1)))
            design = np.vstack([design, pool[choice]])
            pool = np.delete(pool, choice, axis=0)
    return [space.to_vector(space.decode_values(row)) for row in design[:n]]


def min_pairwise_distance(space: SearchSpace, vectors: list[ExperimentalConditionVector]) -> float:
    """Space-filling diagnostic: smallest distance between design points."""
    encoded = space.encode_many(vectors)
    distances = _distance_matrix(encoded, encoded, space)
    np.fill_diagonal(distances, np.inf)
    return float(distances.min())
