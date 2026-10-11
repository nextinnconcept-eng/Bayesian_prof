"""add hypothesis_log, bo_iterations, recommendations, closed_loop_state_log

Revision ID: d7a9c1e3f5b2
Revises: b3e1f2a4c5d6
Create Date: 2026-10-11 11:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

import dssc_discovery.database.models
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d7a9c1e3f5b2"
down_revision: str | Sequence[str] | None = "b3e1f2a4c5d6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

GUID = dssc_discovery.database.models.GUID
UTCDateTime = dssc_discovery.database.models.UTCDateTime

_STATES = (
    "proposed",
    "approved",
    "fabricated",
    "measured",
    "validated",
    "ingested",
    "model_updated",
    "rejected",
    "failed_fabrication",
    "failed_measurement",
    "superseded",
)
_SPECIES = (
    "Telfairia occidentalis",
    "Thunbergia erecta",
    "Beta vulgaris",
    "Chromolaena odorata",
)


def _provenance() -> list[sa.Column]:
    return [
        sa.Column("id", GUID(length=36), nullable=False),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("operator", sa.String(length=200), nullable=False),
        sa.Column("is_synthetic", sa.Boolean(), nullable=False),
        sa.Column("experiment_version", sa.String(length=50), nullable=False),
        sa.Column("notes", sa.Text(), nullable=False),
    ]


def _enum(name: str, values: tuple[str, ...]) -> sa.Enum:
    return sa.Enum(*values, name=name, native_enum=False, length=50)


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "hypothesis_log",
        sa.Column("observation", sa.Text(), nullable=False),
        sa.Column("candidate_mechanism", sa.Text(), nullable=False),
        sa.Column("confound_check", sa.Text(), nullable=False),
        sa.Column("proposed_confirmatory_experiment", sa.Text(), nullable=False),
        sa.Column(
            "status",
            _enum("hypothesisstatus", ("open", "testing", "confirmed", "refuted")),
            nullable=False,
        ),
        sa.Column("status_note", sa.Text(), nullable=False),
        sa.Column("shap_features", sa.Text(), nullable=False),
        sa.Column("model_run_id", GUID(length=36), nullable=True),
        sa.Column("interpretation_caveat", sa.Text(), nullable=False),
        *_provenance(),
        sa.ForeignKeyConstraint(["model_run_id"], ["model_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_hypothesis_log_status", "hypothesis_log", ["status"])

    op.create_table(
        "bo_iterations",
        sa.Column("species_code", _enum("speciescode", _SPECIES), nullable=False),
        sa.Column("cycle_number", sa.Integer(), nullable=False),
        sa.Column("model_run_id", GUID(length=36), nullable=False),
        sa.Column("dataset_version_id", GUID(length=36), nullable=False),
        sa.Column("acquisition_function", sa.String(length=100), nullable=False),
        sa.Column("objective", sa.String(length=50), nullable=False),
        sa.Column("search_space_json", sa.Text(), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("batch_size", sa.Integer(), nullable=False),
        sa.Column("best_observed", sa.Float(), nullable=True),
        sa.Column("rejected_candidates_json", sa.Text(), nullable=False),
        sa.Column("convergence_json", sa.Text(), nullable=False),
        *_provenance(),
        sa.ForeignKeyConstraint(["species_code"], ["species.code"]),
        sa.ForeignKeyConstraint(["model_run_id"], ["model_runs.id"]),
        sa.ForeignKeyConstraint(["dataset_version_id"], ["ml_dataset_versions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_bo_iterations_species_code", "bo_iterations", ["species_code"])
    op.create_index("ix_bo_iterations_model_run_id", "bo_iterations", ["model_run_id"])

    op.create_table(
        "recommendations",
        sa.Column("bo_iteration_id", GUID(length=36), nullable=False),
        sa.Column("experimental_condition_id", GUID(length=36), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("parameters_json", sa.Text(), nullable=False),
        sa.Column("predicted_objective", sa.Float(), nullable=False),
        sa.Column("predictive_uncertainty", sa.Float(), nullable=False),
        sa.Column("acquisition_value", sa.Float(), nullable=False),
        sa.Column("predicted_retention", sa.Float(), nullable=True),
        sa.Column("predicted_retention_uncertainty", sa.Float(), nullable=True),
        sa.Column("rationale_text", sa.Text(), nullable=False),
        *_provenance(),
        sa.ForeignKeyConstraint(["bo_iteration_id"], ["bo_iterations.id"]),
        sa.ForeignKeyConstraint(["experimental_condition_id"], ["experimental_conditions.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("experimental_condition_id"),
    )
    op.create_index("ix_recommendations_bo_iteration_id", "recommendations", ["bo_iteration_id"])

    op.create_table(
        "closed_loop_state_log",
        sa.Column("experimental_condition_id", GUID(length=36), nullable=False),
        sa.Column("from_state", _enum("closedloopstate", _STATES), nullable=True),
        sa.Column("to_state", _enum("closedloopstate", _STATES), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("superseding_bo_iteration_id", GUID(length=36), nullable=True),
        *_provenance(),
        sa.ForeignKeyConstraint(["experimental_condition_id"], ["experimental_conditions.id"]),
        sa.ForeignKeyConstraint(["superseding_bo_iteration_id"], ["bo_iterations.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_closed_loop_state_log_experimental_condition_id",
        "closed_loop_state_log",
        ["experimental_condition_id"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_closed_loop_state_log_experimental_condition_id", "closed_loop_state_log")
    op.drop_table("closed_loop_state_log")
    op.drop_index("ix_recommendations_bo_iteration_id", "recommendations")
    op.drop_table("recommendations")
    op.drop_index("ix_bo_iterations_model_run_id", "bo_iterations")
    op.drop_index("ix_bo_iterations_species_code", "bo_iterations")
    op.drop_table("bo_iterations")
    op.drop_index("ix_hypothesis_log_status", "hypothesis_log")
    op.drop_table("hypothesis_log")
