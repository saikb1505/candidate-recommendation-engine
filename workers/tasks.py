import asyncio
import tempfile
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

from workers.celery_app import celery_app
from db.models import Candidate, JobPost, CandidateJobMatch
from storage.s3 import download_file, delete_file
from ingestion.pipeline import process_single
from embeddings.vector_store import add_resume
from recommendation.ranker import smart_match
from config.settings import config


def _make_session():
    """
    Create a fresh engine + session factory per task call.

    NullPool disables connection pooling so there is no state
    tied to a previous event loop. Each asyncio.run() call gets
    its own engine, which is disposed before the coroutine returns.
    """
    engine = create_async_engine(config.database_url, poolclass=NullPool)
    return async_sessionmaker(engine, expire_on_commit=False), engine


# ──────────────────────────────────────────────
#  FLOW 1: Ingest a resume from S3
# ──────────────────────────────────────────────

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def ingest_resume(self, s3_key: str, candidate_id: str):
    try:
        asyncio.run(_ingest_resume(s3_key, candidate_id))
    except IntegrityError:
        # Duplicate email is permanent — retrying will never resolve it.
        # The inner handler already deleted the row and S3 file.
        pass
    except Exception as exc:
        raise self.retry(exc=exc)


async def _ingest_resume(s3_key: str, candidate_id: str):
    Session, engine = _make_session()

    suffix = Path(s3_key).suffix.lower()
    if suffix == ".pdf":
        file_type = "pdf"
    elif suffix in (".docx", ".doc"):
        file_type = "docx"
    else:
        file_type = "image"

    tmp_path = None
    try:
        file_bytes = download_file(s3_key)

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        resume, error = await process_single(tmp_path, file_type)

        async with Session() as session:
            result = await session.execute(
                select(Candidate).where(Candidate.id == uuid.UUID(candidate_id))
            )
            candidate = result.scalar_one_or_none()
            if not candidate:
                print(f"[ingest_resume] candidate {candidate_id} not found in DB")
                return

            if error:
                print(f"[ingest_resume] extraction failed: {error}")
                candidate.status = "failed"
            else:
                # Check for any existing candidate with this email (any status)
                if resume.contact.email:
                    dup_result = await session.execute(
                        select(Candidate).where(
                            Candidate.email == resume.contact.email,
                            Candidate.id != uuid.UUID(candidate_id),
                        )
                    )
                    existing = dup_result.scalar_one_or_none()
                    if existing:
                        print(f"[ingest_resume] duplicate email {resume.contact.email!r}, "
                              f"deleting {candidate_id} (duplicate of {existing.id})")
                        await session.delete(candidate)
                        await session.commit()
                        delete_file(s3_key)
                        return

                candidate.name = resume.contact.name
                candidate.email = resume.contact.email
                candidate.phone = resume.contact.phone
                candidate.location = resume.contact.location
                candidate.skills = resume.skills
                candidate.total_years_experience = resume.total_years_experience
                candidate.highest_education = resume.highest_education
                candidate.summary = resume.summary
                candidate.chroma_id = candidate_id
                candidate.raw_data = resume.model_dump()
                candidate.status = "processed"

                add_resume(resume, candidate_id)

            try:
                await session.commit()
                print(f"[ingest_resume] committed candidate {candidate_id} status={candidate.status}")
            except IntegrityError:
                # Race condition: another worker committed the same email just before us.
                # Discard the broken session and use a fresh one to mark as duplicate.
                await session.rollback()
                async with Session() as session2:
                    result2 = await session2.execute(
                        select(Candidate).where(Candidate.id == uuid.UUID(candidate_id))
                    )
                    candidate2 = result2.scalar_one_or_none()
                    if candidate2:
                        await session2.delete(candidate2)
                        await session2.commit()
                delete_file(s3_key)
                print(f"[ingest_resume] race-condition duplicate for {candidate_id}, deleted row and S3 file")
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
        await engine.dispose()


# ──────────────────────────────────────────────
#  FLOW 2: Match candidates to a job post
# ──────────────────────────────────────────────

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def process_job_post(self, job_post_id: str):
    try:
        asyncio.run(_process_job_post(job_post_id))
    except Exception as exc:
        raise self.retry(exc=exc)


async def _process_job_post(job_post_id: str):
    Session, engine = _make_session()

    try:
        async with Session() as session:
            result = await session.execute(
                select(JobPost).where(JobPost.id == uuid.UUID(job_post_id))
            )
            job_post = result.scalar_one_or_none()
            if not job_post:
                print(f"[process_job_post] job_post {job_post_id} not found in DB")
                return

            matches = await smart_match(job_description=job_post.description, top_k=20)

            for match in matches:
                score = min(100, round(match.similarity_score * 100))

                candidate_result = await session.execute(
                    select(Candidate).where(Candidate.chroma_id == match.resume_id)
                )
                candidate = candidate_result.scalar_one_or_none()
                if not candidate:
                    continue

                existing_match_result = await session.execute(
                    select(CandidateJobMatch).where(
                        CandidateJobMatch.candidate_id == candidate.id,
                        CandidateJobMatch.job_post_id == job_post.id,
                    )
                )
                existing_match = existing_match_result.scalar_one_or_none()
                if existing_match:
                    existing_match.match_score = score
                    existing_match.match_explanation = match.match_explanation
                    existing_match.similarity_score = match.similarity_score
                else:
                    session.add(CandidateJobMatch(
                        candidate_id=candidate.id,
                        job_post_id=job_post.id,
                        match_score=score,
                        match_explanation=match.match_explanation,
                        similarity_score=match.similarity_score,
                    ))

            job_post.status = "processed"
            await session.commit()
            print(f"[process_job_post] committed {len(matches)} matches for job {job_post_id}")
    finally:
        await engine.dispose()
