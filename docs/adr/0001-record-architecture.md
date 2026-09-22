# ADR-0001: Initial architecture decisions

Status: Accepted
Date: 2026-09-22

## Context

This is a bootstrap ADR: it records the architecture decisions made during
planning (see `PLAN.txt`, Sections J/Q/R) and during Chunks 0.1-0.7, in one
place, before the project accumulates enough history that "why did we do
this" would otherwise require re-reading the whole plan or scrolling `git
log`. Later significant decisions get their own numbered ADR (0002, 0003,
...); this one only covers what's already been decided as of Chunk 0.8.

Each decision below follows Decision / Why / Consequences.

## 1. Database engine: SQLite (dev) -> PostgreSQL (production)

**Decision**: use SQLite for local development, target PostgreSQL for the
mature system, via SQLAlchemy so the engine is a connection-string change.

**Why**: matches the original spec; the working directory happens to sit
under `C:\xampp\htdocs`, which has MySQL/MariaDB available, but there was
no reason tied to this project's needs to prefer it over the already-planned
Postgres path.

**Consequences**: schema must stay engine-agnostic (no SQLite- or
Postgres-only column types) once Chunk 1.4 defines it.

## 2. Bayesian optimization: per-species search spaces, not joint

**Decision**: run BO with one independent, constrained search space per
species (Chunk 6.1), rather than one joint space with species as a
categorical dimension.

**Why**: conditional feasibility (concentration ranges differ by plant,
some sintering/surface-treatment combinations are infeasible) is far
easier to encode as N independent bounded spaces than as one space with
conditional parameters, which BoTorch doesn't natively support well.

**Consequences**: cross-species knowledge transfer is handled entirely by
the ML/XAI layer (leave-one-species-out validation, Chunk 4.5/10.x), never
by the optimizer. Joint optimization remains an additive change later if
ever needed, not a rewrite.

## 3. Validation strategy: Leave-One-Condition-Out CV as primary

**Decision**: LOCO-CV grouped by `experimental_condition_id` is the primary
evaluation strategy (Chunk 4.4), not standard k-fold.

**Why**: at N~70-85 experimental conditions, LOCO maximizes training data
per fold while still preventing replicate leakage across the
train/test boundary -- standard k-fold at small k risks folds that are
just single conditions anyway.

**Consequences**: results must be reported with confidence intervals from
repeated resampling, never a single point-estimate R^2.

## 4. Units: Pint at the ingestion boundary only

**Decision**: `to_canonical()` (Chunk 0.5) is the only place a Pint
`Quantity` object is constructed. Everywhere else -- database, features,
ML -- stores and consumes plain floats already in the canonical unit.

**Why**: running Pint quantities through pandas/sklearn pipelines is
friction without benefit at this project's scale; converting once at entry
and storing plain floats is simpler and just as safe.

**Consequences**: any new unit-bearing field must be added to
`CANONICAL_UNITS` and go through `to_canonical()` at ingestion -- there is
no fallback unit-safety net further downstream.

## 5. Optuna vs. BoTorch/GPyTorch: distinct roles, not interchangeable

**Decision**: Optuna (if used at all) is confined to ML hyperparameter
tuning inside nested CV (Chunk 4.4's inner loop). BoTorch/GPyTorch is the
only engine used for the actual closed-loop experimental Bayesian
optimization (Chunk 6.x).

**Why**: conflating the two would mean using a general-purpose HPO tool's
uncertainty model for real lab-experiment recommendations, where BoTorch's
purpose-built GP surrogate and acquisition machinery is what the science
actually requires.

**Consequences**: none of the M6+ closed-loop code should import Optuna;
if it does, that's a sign of scope drift worth stopping on.

## 6. Gaussian Process Regression: a predictive-model candidate, not just a BO risk

**Decision**: GPR is evaluated in Chunk 4.3 as a candidate predictive model
in its own right, not only as the BO surrogate.

**Why**: the original risk register flagged "GP scalability" as a concern;
at N~70-85 this is a non-issue, and GPR is arguably the best-suited model
at this scale. The real risk runs the other way -- tree ensembles
(RF/XGBoost) overfitting a small N with too many features.

**Consequences**: Chunk 3.1's feature-budget guardrail (~N/10 features for
trees, ~N/5 for GPR/linear) applies asymmetrically by model family.

## 7. Local pre-commit hooks instead of external mirror repos

**Decision**: `.pre-commit-config.yaml` (Chunk 0.7) uses `repo: local`
hooks that invoke the venv's own installed ruff/black/mypy directly.

**Why**: external mirror-repo hooks (e.g. `astral-sh/ruff-pre-commit`,
`psf/black`, `pre-commit/mirrors-mypy`) pin a specific git tag; guessing a
tag that's guaranteed to exist without network verification is unreliable
(mypy, for instance, was already at 2.3.1 by the time this was written --
well past what would have been assumed). Local hooks guarantee the exact
same tool versions run locally and in CI, since both install from the same
`pyproject.toml`/lock file.

**Consequences**: contributors must have the project's venv set up (not
just `pre-commit`'s isolated per-hook environments) for hooks to run.
Acceptable for a solo-researcher project at this stage.

## 8. mypy targets the real runtime version, not the compatibility floor

**Decision**: `[tool.mypy] python_version = "3.13"` (the actual dev/CI
Python), while ruff's `target-version = "py311"` still guards the code's
stated 3.11+ compatibility floor.

**Why**: pinning mypy's `python_version` to 3.11 caused it to fail
*parsing* 3.12+ syntax found inside third-party packages (`pint`) it
follows imports into, not just checking our own code -- a mypy limitation
where `exclude` doesn't stop it from analyzing followed imports inside
excluded directories.

**Consequences**: mypy won't itself flag accidental use of 3.12+-only
syntax in our own code; that floor is enforced by ruff instead.

## 9. GitHub repo: private, under the `mcadexina` account

**Decision**: `github.com/mcadexina/Bayesian_prof`, private visibility.

**Why**: unpublished PhD research feeding multiple planned papers --
private avoids any scooping/priority risk before the researcher is ready
to share; `mcadexina` was the active `gh` CLI account and the researcher's
explicit choice among several linked accounts.
