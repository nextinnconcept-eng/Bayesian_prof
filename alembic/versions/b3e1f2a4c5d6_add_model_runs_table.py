"""add model_runs table

Revision ID: b3e1f2a4c5d6
Revises: 7ad238f9594a
Create Date: 2026-10-11 09:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

import dssc_discovery.database.models
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b3e1f2a4c5d6"
down_revision: str | Sequence[str] | None = "7ad238f9594a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "model_runs",
        sa.Column(
            "dataset_version_id", dssc_discovery.database.models.GUID(length=36), nullable=False
        ),
        sa.Column("feature_set_version_id", sa.String(length=64), nullable=False),
        sa.Column("feature_names_json", sa.Text(), nullable=False),
        sa.Column("model_name", sa.String(length=100), nullable=False),
        sa.Column("model_family", sa.String(length=50), nullable=False),
        sa.Column("model_class", sa.String(length=200), nullable=False),
        sa.Column("strategy", sa.String(length=50), nullable=False),
        sa.Column("git_commit_hash", sa.String(length=64), nullable=False),
        sa.Column("is_dirty", sa.Boolean(), nullable=False),
        sa.Column("random_seed", sa.Integer(), nullable=False),
        sa.Column("hyperparameters_json", sa.Text(), nullable=False),
        sa.Column("cv_config_json", sa.Text(), nullable=False),
        sa.Column("metrics_json", sa.Text(), nullable=False),
        sa.Column("mlflow_run_id", sa.String(length=64), nullable=False),
        sa.Column("mlflow_experiment_name", sa.String(length=200), nullable=False),
        sa.Column("id", dssc_discovery.database.models.GUID(length=36), nullable=False),
        sa.Column("created_at", dssc_discovery.database.models.UTCDateTime(), nullable=False),
        sa.Column("operator", sa.String(length=200), nullable=False),
        sa.Column("is_synthetic", sa.Boolean(), nullable=False),
        sa.Column("experiment_version", sa.String(length=50), nullable=False),
        sa.Column("notes", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["dataset_version_id"], ["ml_dataset_versions.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("mlflow_run_id"),
    )
    with op.batch_alter_table("model_runs", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_model_runs_dataset_version_id"), ["dataset_version_id"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_model_runs_feature_set_version_id"),
            ["feature_set_version_id"],
            unique=False,
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("model_runs", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_model_runs_feature_set_version_id"))
        batch_op.drop_index(batch_op.f("ix_model_runs_dataset_version_id"))

    op.drop_table("model_runs")
