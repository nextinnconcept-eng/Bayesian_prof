"""BoTorch GP surrogate for BO (Chunk 6.2): a MixedSingleTaskGP over a
species' SearchSpace encoding (continuous dims in [0, 1], categoricals
as indices), standardized outputs, maximum-marginal-likelihood fit.
Reuses 4.3's choices: Matern-5/2 on continuous dims (MixedSingleTaskGP's
default continuous kernel) and BoTorch's Standardize outcome transform.
With no categorical dims it falls back to the exact 4.3 configuration
(SingleTaskGP, explicit Matern-5/2).

Only DESIGN-stage information enters the surrogate -- the search-space
parameters -- since BO predicts PCE for conditions that don't exist yet
(features/registry.py).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import MixedSingleTaskGP, SingleTaskGP
from botorch.models.model import Model
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from dssc_discovery.domain.experimental_condition import ExperimentalConditionVector
from dssc_discovery.evaluation.cv_harness import TARGET_COLUMN
from dssc_discovery.optimization.search_space import SearchSpace

SPECIES_COLUMN = "species_code"


@dataclass
class Surrogate:
    model: Model
    space: SearchSpace
    train_X: torch.Tensor
    train_Y: torch.Tensor
    target: str
    seed: int
    dataset_version_id: UUID | None = None
    model_run_id: UUID | None = None

    @property
    def best_observed(self) -> float:
        return float(self.train_Y.max())

    def predict_encoded(self, X: np.ndarray | torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        tensor = torch.as_tensor(np.asarray(X, dtype=np.float64), dtype=torch.double)
        with torch.no_grad():
            posterior = self.model.posterior(tensor)
            mean = posterior.mean.squeeze(-1).numpy()  # type: ignore[attr-defined]
            std = (
                posterior.variance.clamp_min(1e-12)  # type: ignore[attr-defined]
                .sqrt()
                .squeeze(-1)
                .numpy()
            )
        return mean, std

    def predict(
        self, vectors: Sequence[ExperimentalConditionVector]
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.predict_encoded(self.space.encode_many(vectors))


def vectors_from_frame(data: pd.DataFrame) -> list[ExperimentalConditionVector]:
    fields = list(ExperimentalConditionVector.model_fields)
    return [
        ExperimentalConditionVector.model_validate(record)
        for record in data[fields].to_dict(orient="records")
    ]


def species_rows(data: pd.DataFrame, space: SearchSpace, target: str) -> pd.DataFrame:
    rows = data[data[SPECIES_COLUMN].astype(str) == space.species.value]
    return rows[rows[target].notna()]


def fit_gp(
    train_X: torch.Tensor, train_Y: torch.Tensor, categorical_dims: list[int], seed: int
) -> Model:
    torch.manual_seed(seed)
    if categorical_dims:
        model: Model = MixedSingleTaskGP(
            train_X, train_Y, cat_dims=categorical_dims, outcome_transform=Standardize(m=1)
        )
    else:
        model = SingleTaskGP(
            train_X,
            train_Y,
            covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=train_X.shape[-1])),
            outcome_transform=Standardize(m=1),
        )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    model.eval()
    return model


def fit_surrogate(
    data: pd.DataFrame,
    space: SearchSpace,
    *,
    target: str = TARGET_COLUMN,
    seed: int = 0,
    dataset_version_id: UUID | None = None,
    min_rows: int = 3,
) -> Surrogate:
    """Fit on this species' measured rows of an ML-ready frame (one row
    per device; replicates are separate noisy observations of the same
    point, which the GP's noise term absorbs)."""
    rows = species_rows(data, space, target)
    if len(rows) < min_rows:
        raise ValueError(
            f"only {len(rows)} measured rows for {space.species.value}; need >= {min_rows}"
        )
    train_X = torch.tensor(space.encode_many(vectors_from_frame(rows)), dtype=torch.double)
    train_Y = torch.tensor(rows[target].to_numpy(dtype=float).reshape(-1, 1), dtype=torch.double)
    return Surrogate(
        model=fit_gp(train_X, train_Y, space.categorical_dims, seed),
        space=space,
        train_X=train_X,
        train_Y=train_Y,
        target=target,
        seed=seed,
        dataset_version_id=dataset_version_id,
    )
