"""Gaussian Process Regression wrapper (Chunk 4.3): a BoTorch
SingleTaskGP with an explicit Matern-5/2 kernel, usable both as a
candidate predictive model in the evaluation harness (4.4) and, later,
reused unmodified as the BO surrogate spine (6.2) -- PLAN.txt's own
rationale (Section H) for building this now rather than deferring it
to M6.

BoTorch's SingleTaskGP defaults to an RBF kernel, NOT Matern (verified
empirically against this project's installed botorch 0.18.1 before
writing this module -- worth checking rather than assuming, since the
task explicitly requires Matern-5/2), so the kernel is passed
explicitly via covar_module. input_transform=Normalize +
outcome_transform=Standardize give the "standardized I/O" the task
calls for using BoTorch's own built-in, well-tested transforms rather
than a hand-rolled equivalent -- important because they're inverted
automatically inside model.posterior(), and because 6.2's BO surrogate
needs this exact same input/output standardization behavior.

Hyperparameters (lengthscale, outputscale, noise) are fit by maximum
marginal likelihood via botorch.fit.fit_gpytorch_mll (L-BFGS-B under
the hood, not a hand-rolled training loop) -- empirically confirmed
deterministic given torch.manual_seed(), since the only source of
randomness is the initial hyperparameter draw from gpytorch's default
priors.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from dssc_discovery.features.registry import FeatureSpec
from dssc_discovery.models.base import PreprocessingWrappedRegressor


class _SingleTaskGPEstimator(BaseEstimator, RegressorMixin):
    """The raw numpy-array-facing GP estimator -- the "model" step
    inside GPRModel's inherited preprocessing Pipeline (models/base.py),
    matching the same two-layer structure tree_models.py/
    linear_models.py already use. Not exported: GPRModel below is what
    the evaluation harness (4.4) and the BO surrogate (6.2) actually
    use.
    """

    def __init__(self, random_state: int | None = None) -> None:
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: pd.Series | np.ndarray) -> _SingleTaskGPEstimator:
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        n_features = X.shape[1]

        train_X = torch.tensor(X, dtype=torch.double)
        train_Y = torch.tensor(y, dtype=torch.double)

        covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=n_features))
        model = SingleTaskGP(
            train_X,
            train_Y,
            covar_module=covar_module,
            input_transform=Normalize(d=n_features),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        model.eval()

        self.model_ = model
        self.n_features_in_ = n_features
        return self

    def predict(
        self, X: np.ndarray, return_std: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self, "model_")
        test_X = torch.tensor(np.asarray(X, dtype=np.float64), dtype=torch.double)

        with torch.no_grad():
            posterior = self.model_.posterior(test_X)
            mean = posterior.mean.squeeze(-1).numpy()
            if not return_std:
                return mean
            std = posterior.variance.clamp_min(1e-12).sqrt().squeeze(-1).numpy()
            return mean, std


class GPRModel(PreprocessingWrappedRegressor):
    """Matern-5/2 single-task GP over feature_specs, usable identically
    to every other model in the evaluation harness (4.4) via the
    inherited fit(X_raw_df, y) / predict(X_raw_df); pass
    return_std=True to also get the posterior standard deviation
    (Section H's BO surrogate, 6.2, is this class's other consumer, and
    the reason predict()'s uncertainty output exists at all).
    """

    def __init__(
        self,
        feature_specs: Sequence[FeatureSpec],
        random_state: int | None = None,
    ) -> None:
        self.feature_specs = feature_specs
        self.random_state = random_state

    def _make_estimator(self) -> BaseEstimator:
        return _SingleTaskGPEstimator(random_state=self.random_state)
