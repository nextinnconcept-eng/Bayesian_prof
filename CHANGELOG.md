# Changelog

Notable changes to this project, grouped by roadmap chunk (see `PLAN.txt`).
Loosely follows [Keep a Changelog](https://keepachangelog.com/); for exact
diffs and dates, `git log` is authoritative -- this file is for *why a
chunk exists*, not a mechanical commit mirror.

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
