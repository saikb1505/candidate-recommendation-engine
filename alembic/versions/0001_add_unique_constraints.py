"""add unique constraints to prevent duplicate records

Revision ID: 0001
Revises:
Create Date: 2026-05-13
"""
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_unique_constraint("uq_candidates_email", "candidates", ["email"])
    op.create_unique_constraint(
        "uq_candidate_job", "candidate_job_matches", ["candidate_id", "job_post_id"]
    )


def downgrade() -> None:
    op.drop_constraint("uq_candidate_job", "candidate_job_matches", type_="unique")
    op.drop_constraint("uq_candidates_email", "candidates", type_="unique")
