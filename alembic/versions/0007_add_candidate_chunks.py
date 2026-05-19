"""add candidate_chunks table for field-level embeddings

Revision ID: 0007
Revises: 0006
Create Date: 2026-05-19
"""
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from alembic import op
from pgvector.sqlalchemy import Vector

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "candidate_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "candidate_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("candidates.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_type", sa.String(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(768), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_candidate_chunks_candidate_id", "candidate_chunks", ["candidate_id"])
    op.execute(
        "CREATE INDEX candidate_chunks_embedding_hnsw ON candidate_chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS candidate_chunks_embedding_hnsw")
    op.drop_index("ix_candidate_chunks_candidate_id", table_name="candidate_chunks")
    op.drop_table("candidate_chunks")
