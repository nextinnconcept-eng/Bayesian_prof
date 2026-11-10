"""add dssc_simulation_runs table

Revision ID: e8b2d4f6a1c3
Revises: d7a9c1e3f5b2
Create Date: 2026-11-10 21:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

import dssc_discovery.database.models
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e8b2d4f6a1c3"
down_revision: str | Sequence[str] | None = "d7a9c1e3f5b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "dssc_simulation_runs",
        sa.Column("label", sa.String(length=200), nullable=False),
        sa.Column("model", sa.String(length=30), nullable=False),
        sa.Column("temperature_k", sa.Float(), nullable=False),
        sa.Column("irradiance_w_m2", sa.Float(), nullable=False),
        sa.Column("area_cm2", sa.Float(), nullable=False),
        sa.Column("alpha_cm1", sa.Float(), nullable=False),
        sa.Column("uvvis_wavelength_nm", sa.Float(), nullable=True),
        sa.Column("uvvis_json", sa.Text(), nullable=False),
        sa.Column("parameters_json", sa.Text(), nullable=False),
        sa.Column("voc_v", sa.Float(), nullable=False),
        sa.Column("jsc_ma_cm2", sa.Float(), nullable=False),
        sa.Column("isc_ma", sa.Float(), nullable=False),
        sa.Column("fill_factor", sa.Float(), nullable=False),
        sa.Column("vmp_v", sa.Float(), nullable=False),
        sa.Column("imp_ma_cm2", sa.Float(), nullable=False),
        sa.Column("pmax_mw_cm2", sa.Float(), nullable=False),
        sa.Column("efficiency_percent", sa.Float(), nullable=False),
        sa.Column("id", dssc_discovery.database.models.GUID(length=36), nullable=False),
        sa.Column("created_at", dssc_discovery.database.models.UTCDateTime(), nullable=False),
        sa.Column("operator", sa.String(length=200), nullable=False),
        sa.Column("is_synthetic", sa.Boolean(), nullable=False),
        sa.Column("experiment_version", sa.String(length=50), nullable=False),
        sa.Column("notes", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("dssc_simulation_runs")
