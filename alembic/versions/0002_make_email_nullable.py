"""make candidates.email nullable

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-13
"""
from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("candidates", "email", existing_type=sa.String(), nullable=True)


def downgrade() -> None:
    op.alter_column("candidates", "email", existing_type=sa.String(), nullable=False)
