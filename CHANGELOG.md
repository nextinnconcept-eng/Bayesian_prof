# Changelog

Notable changes to this project, grouped by roadmap chunk (see `PLAN.txt`).
Loosely follows [Keep a Changelog](https://keepachangelog.com/); for exact
diffs and dates, `git log` is authoritative -- this file is for *why a
chunk exists*, not a mechanical commit mirror.

## [Unreleased] - M4 Predictive ML (4.4-4.8)

### Added
- **4.4** Leakage-safe grouped nested CV harness (`evaluation/cv_harness.py`)
  -- strategy A: outer LOCO (or shuffled grouped k-fold) by
  experimental condition, inner grouped GridSearchCV on the outer training
  fold only, MAE/RMSE/R² with condition-bootstrap CIs across repeats,
  feature-budget enforcement, measured outcomes (Voc/Jsc/FF) kept out of X.
  Leakage canary test proves ungrouped CV is optimistic on replicate data.
- **4.5** Strategy B, forward extrapolation by `design_source` stage
  (`evaluation/temporal_split.py`), and strategy C, leave-one-species-out
  (`evaluation/cross_species.py`); `build_ml_dataset()` now carries
  `design_source` (provenance column, not a feature).
- **4.6** Out-of-fold residual analysis, heteroscedasticity flag, train-vs-CV
  overfitting gap and grouped learning curves (`evaluation/diagnostics.py`).
- **4.7** MLflow tracking + `model_runs` table (`models/tracking.py`, Alembic
  revision `b3e1f2a4c5d6`): one MLflow run and one ModelRun row per model,
  citing dataset version, feature-set hash, git commit and seed;
  `rerun_model_run()` reproduces recorded metrics.
- **4.8** M4 gate integration test: all models + baselines under A/B/C on
  synthetic data, side-by-side comparison table. The synthetic generator
  gained `design_source`, `n_replicates` and a condition-level offset
  (defaults reproduce previous output exactly).

### Changed
- Local MLflow store: a plain-path `mlflow_tracking_uri` is now an SQLite
  store plus `artifacts/` inside that directory -- MLflow 3.x refuses the
  old filesystem backend by default.

### Fixed
- Audit `before_flush` hook only audits this schema's ORM objects; it
  previously intercepted other libraries' sessions (MLflow's tracking DB)
  and failed on their missing `audit_log` table.

## [Unreleased] - M0 Foundation

### Added
- **0.1** Repository & git initialization -- `dssc_discovery` package,
  src-layout, private GitHub repo.
- **0.2** Pinned scientific dependency stack (pandas, numpy, scipy,
  scikit-learn, xgboost, shap, gpytorch, botorch, mlflow, sqlalchemy,
  alembic, pydantic, pint, matplotlib), `dev`/`ui` extras,
  `requirements-lock.txt`.
- **0.3** Configuration system -- `Config`/`get_config()` layering
  `config/default.yaml` under `DSSC_*` env vars and an optional `.env`.
- **0.4** Structured logging -- `get_logger()` to console + rotating file,
  level driven by config.
- **0.5** Units registry -- `to_canonical()` with Pint-based conversion,
  used only at the ingestion boundary; pH intentionally excluded (not a
  Pint unit; range-validated later instead).
- **0.6** 13 domain enums (`SpeciesCode`, `ClosedLoopState`, `DesignSource`,
  ...) as controlled vocabularies, each `StrEnum` for clean Pydantic
  round-tripping.
- **0.7** Lint/type-check/test automation -- ruff, black, mypy (lenient),
  local pre-commit hooks pinned to the venv's own tool versions, GitHub
  Actions CI.
- **0.8** This changelog and `docs/adr/0001-record-architecture.md`.

See `PLAN.txt` for the full milestone roadmap and `docs/adr/` for the
reasoning behind decisions that deviated from or extended the original
plan.
