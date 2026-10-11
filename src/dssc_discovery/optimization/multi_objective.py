"""Multi-objective BO, PCE x stability retention (Chunk 9.2).

Two independent GP surrogates (6.2) -- PCE on every measured row,
retention only on rows whose stability_status is "measured" (9.1) --
combined in a ModelListGP, so partial observation needs no imputation:
each objective learns from exactly the data that exists. Acquisition is
qLogNoisyExpectedHypervolumeImprovement over the mixed search space;
candidates are hard-validated exactly like the single-objective path.

Two-tier design (Section Q): stability lags PCE by the ageing time, so
the PCE-only loop (6.x/8.x) keeps driving cycle cadence, and this
multi-objective re-ranking runs periodically as stability data lands --
it routes through the same BOIteration/Recommendation/approval path
(closed_loop/orchestrator.recommend_multi_objective_cycle).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.models import ModelListGP
from botorch.optim import optimize_acqf_mixed
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.pareto import is_non_dominated

from dssc_discovery.evaluation.cv_harness import TARGET_COLUMN
from dssc_discovery.optimization.acquisition import (
    Candidate,
    acquisition_values,
    encoded_linear_constraints,
)
from dssc_discovery.optimization.search_space import SearchSpace
from dssc_discovery.optimization.surrogate import Surrogate, fit_surrogate
from dssc_discovery.optimization.validation import (
    InfeasibleRecommendationError,
    RejectedCandidate,
    validate_candidate,
)
from dssc_discovery.stability.retention import RETENTION_COLUMN, STATUS_COLUMN

ACQUISITION_NAME = "qLogNoisyExpectedHypervolumeImprovement"


@dataclass
class MultiObjectiveSurrogate:
    pce: Surrogate
    retention: Surrogate
    model: ModelListGP
    ref_point: list[float]

    @property
    def space(self) -> SearchSpace:
        return self.pce.space

    def predict_encoded(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pce_mean, pce_std = self.pce.predict_encoded(X)
        ret_mean, ret_std = self.retention.predict_encoded(X)
        return pce_mean, pce_std, ret_mean, ret_std


@dataclass
class MultiObjectiveResult:
    candidates: list[Candidate]
    retention: list[tuple[float, float]]
    rejected: list[RejectedCandidate] = field(default_factory=list)
    acquisition_function: str = ACQUISITION_NAME


def fit_multi_objective(
    data: pd.DataFrame, space: SearchSpace, *, seed: int = 0, min_stability_rows: int = 5
) -> MultiObjectiveSurrogate:
    if STATUS_COLUMN not in data.columns:
        raise ValueError("data must come from build_multi_objective_dataset() (9.1)")
    aged = data[data[STATUS_COLUMN] == "measured"]
    pce = fit_surrogate(data, space, target=TARGET_COLUMN, seed=seed)
    retention = fit_surrogate(
        aged, space, target=RETENTION_COLUMN, seed=seed, min_rows=min_stability_rows
    )
    # Reference point: just below the worst observed value per objective.
    observed = [pce.train_Y.squeeze(-1), retention.train_Y.squeeze(-1)]
    ref_point = [float(y.min() - 0.1 * (y.max() - y.min() + 1e-9)) for y in observed]
    return MultiObjectiveSurrogate(
        pce=pce,
        retention=retention,
        model=ModelListGP(pce.model, retention.model),  # type: ignore[arg-type]
        ref_point=ref_point,
    )


def optimize_multi_objective(
    surrogate: MultiObjectiveSurrogate,
    *,
    batch_size: int = 5,
    seed: int = 0,
    num_restarts: int = 8,
    raw_samples: int = 256,
    fallback_pool_size: int = 2000,
) -> MultiObjectiveResult:
    torch.manual_seed(seed)
    space = surrogate.space
    # Baseline: the conditions where both objectives are observed.
    acq = qLogNoisyExpectedHypervolumeImprovement(
        model=surrogate.model,
        ref_point=surrogate.ref_point,
        X_baseline=surrogate.retention.train_X,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128]), seed=seed),
        prune_baseline=True,
    )
    raw, _ = optimize_acqf_mixed(
        acq_function=acq,
        bounds=torch.tensor(space.bounds(), dtype=torch.double),
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        fixed_features_list=space.categorical_combinations(),
        inequality_constraints=encoded_linear_constraints(space) or None,
    )

    result = MultiObjectiveResult(candidates=[], retention=[])
    seen: set[tuple[float, ...]] = set()

    def accept(row: np.ndarray) -> None:
        vector = validate_candidate(space, space.decode_values(row))
        encoded = space.encode(vector)
        key = tuple(np.round(encoded, 6))
        if key in seen:
            return
        seen.add(key)
        pce_mean, pce_std, ret_mean, ret_std = surrogate.predict_encoded(encoded[None, :])
        result.candidates.append(
            Candidate(
                vector=vector,
                encoded=encoded,
                acquisition_value=float(acquisition_values(acq, encoded[None, :])[0]),
                predicted_mean=float(pce_mean[0]),
                predicted_std=float(pce_std[0]),
            )
        )
        result.retention.append((float(ret_mean[0]), float(ret_std[0])))

    for row in raw.detach().numpy():
        try:
            accept(row)
        except InfeasibleRecommendationError as exc:
            result.rejected.append(RejectedCandidate(exc.values, exc.reasons))

    if len(result.candidates) < batch_size:
        pool = space.sample(fallback_pool_size, np.random.default_rng(seed))
        scores = acquisition_values(acq, pool)
        for index in np.argsort(scores)[::-1]:
            if len(result.candidates) >= batch_size:
                break
            accept(pool[index])
    return result


def pareto_mask(objectives: np.ndarray) -> np.ndarray:
    """Non-dominated rows of an (n, 2) maximize-both objective matrix."""
    return is_non_dominated(torch.as_tensor(objectives, dtype=torch.double)).numpy()
