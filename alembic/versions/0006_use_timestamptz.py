"""change created_at columns to timestamptz

Revision ID: 0006
Revises: 0005
Create Date: 2026-05-19
"""
import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None

TABLES = ["candidates", "job_posts", "companies", "candidate_job_matches"]


def upgrade() -> None:
    for table in TABLES:
        op.alter_column(
            table,
            "created_at",
            type_=sa.DateTime(timezone=True),
            postgresql_using="created_at AT TIME ZONE 'UTC'",
        )


def downgrade() -> None:
    for table in TABLES:
        op.alter_column(
            table,
            "created_at",
            type_=sa.DateTime(timezone=False),
            postgresql_using="created_at AT TIME ZONE 'UTC'",
        )
