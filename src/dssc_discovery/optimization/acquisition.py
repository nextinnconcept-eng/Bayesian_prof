"""Acquisition + constrained optimizer (Chunk 6.3): qLogNoisyExpected-
Improvement (Section J -- log-space EI avoids plain qEI's vanishing
gradients at this size) optimized over the mixed search space with
optimize_acqf_mixed (one continuous optimization per categorical
assignment). Linear constraints are passed to the optimizer in encoded
space; infeasible-combination rules and discrete steps can't be, so
every optimizer output is decoded and HARD-validated (validation.py):
an infeasible point is rejected, recorded, and replaced by the best
feasible point from a seeded random pool scored with the same
acquisition function -- never clamped into feasibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf_mixed
from botorch.sampling.normal import SobolQMCNormalSampler

from dssc_discovery.domain.experimental_condition import ExperimentalConditionVector
from dssc_discovery.optimization.search_space import ContinuousParam, SearchSpace
from dssc_discovery.optimization.surrogate import Surrogate
from dssc_discovery.optimization.validation import (
    InfeasibleRecommendationError,
    RejectedCandidate,
    validate_candidate,
)

ACQUISITION_NAME = "qLogNoisyExpectedImprovement"


@dataclass(frozen=True)
class Candidate:
    vector: ExperimentalConditionVector
    encoded: np.ndarray
    acquisition_value: float
    predicted_mean: float
    predicted_std: float


@dataclass
class AcquisitionResult:
    candidates: list[Candidate]
    rejected: list[RejectedCandidate] = field(default_factory=list)
    acquisition_function: str = ACQUISITION_NAME


def build_acquisition(surrogate: Surrogate, seed: int = 0) -> AcquisitionFunction:
    return qLogNoisyExpectedImprovement(
        model=surrogate.model,
        X_baseline=surrogate.train_X,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([256]), seed=seed),
        prune_baseline=True,
    )


def encoded_linear_constraints(
    space: SearchSpace,
) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
    """sum(c_i * x_i) <= rhs in real units, rewritten for encoded
    x_i = low_i + u_i * (high_i - low_i) as BoTorch's
    sum(a_i * u_i) >= b form."""
    constraints = []
    for constraint in space.linear_constraints:
        indices, coefficients = [], []
        offset = 0.0
        for name, c in constraint.coefficients.items():
            param = space.param(name)
            assert isinstance(param, ContinuousParam)
            indices.append(space.param_names.index(name))
            coefficients.append(-c * (param.high - param.low))
            offset += c * param.low
        constraints.append(
            (
                torch.tensor(indices, dtype=torch.long),
                torch.tensor(coefficients, dtype=torch.double),
                float(-(constraint.rhs - offset)),
            )
        )
    return constraints


def acquisition_values(acq: AcquisitionFunction, encoded: np.ndarray) -> np.ndarray:
    """Per-point (q=1) acquisition values."""
    tensor = torch.as_tensor(encoded, dtype=torch.double).unsqueeze(-2)
    with torch.no_grad():
        return acq(tensor).numpy()


def _accept(surrogate: Surrogate, acq: AcquisitionFunction, encoded_row: np.ndarray) -> Candidate:
    values = surrogate.space.decode_values(encoded_row)
    vector = validate_candidate(surrogate.space, values)
    encoded = surrogate.space.encode(vector)
    mean, std = surrogate.predict_encoded(encoded[None, :])
    return Candidate(
        vector=vector,
        encoded=encoded,
        acquisition_value=float(acquisition_values(acq, encoded[None, :])[0]),
        predicted_mean=float(mean[0]),
        predicted_std=float(std[0]),
    )


def optimize_acquisition(
    surrogate: Surrogate,
    *,
    batch_size: int = 1,
    seed: int = 0,
    num_restarts: int = 10,
    raw_samples: int = 256,
    fallback_pool_size: int = 2000,
) -> AcquisitionResult:
    """Best feasible batch (sequential-greedy q-batch) for the surrogate."""
    torch.manual_seed(seed)
    space = surrogate.space
    acq = build_acquisition(surrogate, seed)
    raw, _ = optimize_acqf_mixed(
        acq_function=acq,
        bounds=torch.tensor(space.bounds(), dtype=torch.double),
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        fixed_features_list=space.categorical_combinations(),
        inequality_constraints=encoded_linear_constraints(space) or None,
    )

    result = AcquisitionResult(candidates=[])
    seen: set[tuple[float, ...]] = set()
    for row in raw.detach().numpy():
        try:
            candidate = _accept(surrogate, acq, row)
        except InfeasibleRecommendationError as exc:
            result.rejected.append(RejectedCandidate(exc.values, exc.reasons))
            continue
        key = tuple(np.round(candidate.encoded, 6))
        if key not in seen:
            seen.add(key)
            result.candidates.append(candidate)

    if len(result.candidates) < batch_size:
        pool = space.sample(fallback_pool_size, np.random.default_rng(seed))
        scores = acquisition_values(acq, pool)
        for index in np.argsort(scores)[::-1]:
            if len(result.candidates) >= batch_size:
                break
            key = tuple(np.round(pool[index], 6))
            if key in seen:
                continue
            seen.add(key)
            result.candidates.append(_accept(surrogate, acq, pool[index]))
    return result
