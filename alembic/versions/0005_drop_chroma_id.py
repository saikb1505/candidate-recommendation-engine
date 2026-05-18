"""drop chroma_id column from candidates

Revision ID: 0005
Revises: 0004
Create Date: 2026-05-19
"""
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("candidates", "chroma_id")


def downgrade() -> None:
    import sqlalchemy as sa
    op.add_column("candidates", sa.Column("chroma_id", sa.String(), nullable=False, server_default=""))
