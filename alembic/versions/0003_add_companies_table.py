"""add companies table

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-15
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "companies",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("source", sa.String(), nullable=False, server_default="Resume"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_companies_name", "companies", ["name"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_companies_name", table_name="companies")
    op.drop_table("companies")
