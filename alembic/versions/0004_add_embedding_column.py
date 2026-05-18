"""add embedding column to candidates

Revision ID: 0004
Revises: 0003
Create Date: 2026-05-18
"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.add_column("candidates", sa.Column("embedding", Vector(768), nullable=True))
    op.add_column("candidates", sa.Column("embedding_text", sa.Text(), nullable=False, server_default=""))
    op.execute(
        "CREATE INDEX candidates_embedding_hnsw ON candidates "
        "USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS candidates_embedding_hnsw")
    op.drop_column("candidates", "embedding_text")
    op.drop_column("candidates", "embedding")
