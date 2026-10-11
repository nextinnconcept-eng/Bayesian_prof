"""EXPLORATORY (Chunk 10.3): does data from other species improve
cold-start BO for a new species? Evaluated on synthetic objectives first,
as PLAN.txt asks -- "not worth pursuing" is a valid outcome.

Setup: the target species' objective is SyntheticObjective with its
optimum shifted by target_shift relative to the source species'
(a "related but different" species). Both arms see the same n_init LHS
points of the target; the transfer arm additionally trains on
n_source points evaluated on the source objective. Each arm's GP
recommends one batch; we compare the batches' TRUE target values.
Repeated over seeds; reports mean best-in-batch regret per arm.

This study does not change the production optimizer (Section J keeps BO
per species); it only informs whether a cross-species prior is worth
building.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch

from dssc_discovery.closed_loop.synthetic_objective import SyntheticObjective
from dssc_discovery.domain.enums import SpeciesCode
from dssc_discovery.optimization.acquisition import optimize_acquisition
from dssc_discovery.optimization.initial_design import latin_hypercube_design
from dssc_discovery.optimization.search_space import SearchSpace, search_space_for
from dssc_discovery.optimization.surrogate import Surrogate, fit_gp


@dataclass(frozen=True)
class TransferStudyResult:
    target_shift: float
    n_init: int
    n_source: int
    seeds: int
    cold_start_regret: float
    transfer_regret: float
    transfer_helps: bool

    def summary(self) -> str:
        verdict = "helped" if self.transfer_helps else "did NOT help"
        return (
            f"Cross-species prior {verdict}: best-in-first-batch regret "
            f"{self.transfer_regret:.2f} (transfer) vs {self.cold_start_regret:.2f} (cold start), "
            f"optimum shift {self.target_shift}, {self.seeds} seeds. Synthetic, exploratory only."
        )


def _surrogate(space: SearchSpace, X: np.ndarray, y: np.ndarray, seed: int) -> Surrogate:
    train_X = torch.tensor(X, dtype=torch.double)
    train_Y = torch.tensor(y.reshape(-1, 1), dtype=torch.double)
    return Surrogate(
        model=fit_gp(train_X, train_Y, space.categorical_dims, seed),
        space=space,
        train_X=train_X,
        train_Y=train_Y,
        target="synthetic",
        seed=seed,
    )


def run_transfer_study(
    *,
    target_shift: float = 0.1,
    n_init: int = 8,
    n_source: int = 40,
    batch_size: int = 5,
    seeds: int = 5,
    noise_std: float = 0.3,
    acquisition_options: Mapping[str, int] | None = None,
) -> TransferStudyResult:
    space = search_space_for(SpeciesCode.BETA_VULGARIS)
    source = SyntheticObjective(space, noise_std=noise_std)
    target = SyntheticObjective(space, noise_std=noise_std, optimum_shift=target_shift)
    cold: list[float] = []
    transfer: list[float] = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        init = latin_hypercube_design(space, n_init, seed)
        X_init = space.encode_many(init)
        y_init = np.array([target.evaluate(v, rng) for v in init])
        src = latin_hypercube_design(space, n_source, seed + 1000)
        X_src = space.encode_many(src)
        y_src = np.array([source.evaluate(v, rng) for v in src])

        for arm, (X, y) in (
            ("cold", (X_init, y_init)),
            ("transfer", (np.vstack([X_init, X_src]), np.concatenate([y_init, y_src]))),
        ):
            batch = optimize_acquisition(
                _surrogate(space, X, y, seed),
                batch_size=batch_size,
                seed=seed,
                **(acquisition_options or {}),
            ).candidates
            regret = target.optimum_value - max(target.true_value(c.vector) for c in batch)
            (cold if arm == "cold" else transfer).append(regret)

    cold_regret, transfer_regret = float(np.mean(cold)), float(np.mean(transfer))
    return TransferStudyResult(
        target_shift=target_shift,
        n_init=n_init,
        n_source=n_source,
        seeds=seeds,
        cold_start_regret=cold_regret,
        transfer_regret=transfer_regret,
        transfer_helps=transfer_regret < cold_regret,
    )
