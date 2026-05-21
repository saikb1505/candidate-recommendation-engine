"""drop candidate_chunks table and pgvector columns — vectors moved to Qdrant

Revision ID: 0008
Revises: 0007
Create Date: 2026-05-21
"""
import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop candidate_chunks — Qdrant is now the vector store
    op.execute("DROP INDEX IF EXISTS candidate_chunks_embedding_hnsw")
    op.drop_index("ix_candidate_chunks_candidate_id", table_name="candidate_chunks")
    op.drop_table("candidate_chunks")

    # Drop pgvector columns from candidates — no longer used for search
    op.execute("DROP INDEX IF EXISTS candidates_embedding_hnsw")
    op.drop_column("candidates", "embedding")
    op.drop_column("candidates", "embedding_text")


def downgrade() -> None:
    op.add_column("candidates", sa.Column("embedding_text", sa.Text(), nullable=False, server_default=""))
    op.add_column("candidates", sa.Column("embedding", Vector(768), nullable=True))
    op.execute(
        "CREATE INDEX candidates_embedding_hnsw ON candidates "
        "USING hnsw (embedding vector_cosine_ops)"
    )

    op.create_table(
        "candidate_chunks",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("candidate_id", sa.dialects.postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_type", sa.String(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(768), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_candidate_chunks_candidate_id", "candidate_chunks", ["candidate_id"])
    op.execute(
        "CREATE INDEX candidate_chunks_embedding_hnsw ON candidate_chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )
