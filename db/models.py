import uuid
from datetime import datetime

from sqlalchemy import String, Float, Integer, Text, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped, relationship
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass


class Candidate(Base):
    __tablename__ = "candidates"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String, default="")
    email: Mapped[str | None] = mapped_column(String, default=None, index=True, unique=True)
    phone: Mapped[str] = mapped_column(String, default="")
    location: Mapped[str] = mapped_column(String, default="")
    skills: Mapped[list] = mapped_column(ARRAY(String), default=list)
    total_years_experience: Mapped[float] = mapped_column(Float, default=0.0)
    highest_education: Mapped[str] = mapped_column(String, default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    s3_key: Mapped[str] = mapped_column(String)
    chroma_id: Mapped[str] = mapped_column(String, default="")
    status: Mapped[str] = mapped_column(String, default="pending")
    embedding: Mapped[list | None] = mapped_column(Vector(768), nullable=True)
    embedding_text: Mapped[str] = mapped_column(Text, default="")
    raw_data: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    matches: Mapped[list["CandidateJobMatch"]] = relationship(back_populates="candidate")


class JobPost(Base):
    __tablename__ = "job_posts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(Text)
    required_skills: Mapped[list] = mapped_column(ARRAY(String), default=list)
    min_years: Mapped[float] = mapped_column(Float, default=0.0)
    location: Mapped[str] = mapped_column(String, default="")
    status: Mapped[str] = mapped_column(String, default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    matches: Mapped[list["CandidateJobMatch"]] = relationship(back_populates="job_post")


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String, unique=True, index=True)
    source: Mapped[str] = mapped_column(String, default="Resume")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class CandidateJobMatch(Base):
    __tablename__ = "candidate_job_matches"
    __table_args__ = (
        UniqueConstraint("candidate_id", "job_post_id", name="uq_candidate_job"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("candidates.id"))
    job_post_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("job_posts.id"))
    match_score: Mapped[int] = mapped_column(Integer)
    match_explanation: Mapped[str] = mapped_column(Text, default="")
    similarity_score: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    candidate: Mapped["Candidate"] = relationship(back_populates="matches")
    job_post: Mapped["JobPost"] = relationship(back_populates="matches")
