import tempfile
import uuid
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from workers.celery_app import celery_app
from db.models import Candidate, Company, JobPost, CandidateJobMatch
from storage.s3 import download_file, delete_file
from ingestion.pipeline import process_single
from embeddings.embedder import embed_chunks_batch
from embeddings.vector_store import make_qdrant_client, upsert_chunks
from recommendation.ranker import smart_match
from config.settings import config


def _make_session():
    engine = create_engine(config.sync_database_url, poolclass=NullPool)
    return sessionmaker(engine, expire_on_commit=False), engine


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def ingest_resume(self, s3_key: str, candidate_id: str):
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

        resume, error = process_single(tmp_path, file_type)

        with Session() as session:
            if error:
                print(f"[ingest_resume] extraction failed: {error}")
                session.add(Candidate(id=uuid.UUID(candidate_id), s3_key=s3_key, status="failed"))
                session.commit()
                return

            assert resume is not None
            if resume.contact.email:
                existing = session.scalar(
                    select(Candidate).where(Candidate.email == resume.contact.email)
                )
                if existing:
                    print(f"[ingest_resume] duplicate email {resume.contact.email!r}, "
                          f"skipping {candidate_id} (duplicate of {existing.id})")
                    delete_file(s3_key)
                    return

            candidate = Candidate(
                id=uuid.UUID(candidate_id),
                s3_key=s3_key,
                name=resume.contact.name,
                email=resume.contact.email,
                phone=resume.contact.phone,
                location=resume.contact.location,
                skills=resume.skills,
                total_years_experience=resume.total_years_experience,
                highest_education=resume.highest_education,
                summary=resume.summary,
                raw_data=resume.model_dump(),
                status="processed",
            )
            session.add(candidate)

            for company_name in resume.companies:
                existing_company = session.scalar(
                    select(Company).where(Company.name == company_name)
                )
                if not existing_company:
                    session.add(Company(name=company_name, source="Resume"))

            chunks = resume.to_chunks()
            chunk_embeddings = embed_chunks_batch(chunks) if chunks else []

            try:
                session.commit()
                print(f"[ingest_resume] committed candidate {candidate_id} status=processed")

                if chunks and chunk_embeddings:
                    qdrant = make_qdrant_client()
                    try:
                        upsert_chunks(
                            candidate_id=candidate_id,
                            chunks=chunks,
                            embeddings=chunk_embeddings,
                            payload_meta={
                                "name":                   candidate.name,
                                "email":                  candidate.email,
                                "location":               candidate.location,
                                "skills":                 candidate.skills or [],
                                "total_years_experience": candidate.total_years_experience,
                                "highest_education":      candidate.highest_education,
                                "status":                 "processed",
                            },
                            client=qdrant,
                        )
                        print(f"[ingest_resume] upserted {len(chunks)} chunks to Qdrant for {candidate_id}")
                    finally:
                        qdrant.close()
            except IntegrityError:
                session.rollback()
                delete_file(s3_key)
                print(f"[ingest_resume] race-condition duplicate for {candidate_id}, deleted")

    except IntegrityError:
        pass
    except Exception as exc:
        raise self.retry(exc=exc)
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
        engine.dispose()


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def process_job_post(self, job_post_id: str):
    Session, engine = _make_session()
    qdrant = make_qdrant_client()

    try:
        with Session() as session:
            job_post = session.scalar(
                select(JobPost).where(JobPost.id == uuid.UUID(job_post_id))
            )
            if not job_post:
                print(f"[process_job_post] job_post {job_post_id} not found in DB")
                return

            matches = smart_match(job_description=job_post.description, top_k=20, client=qdrant)

            for match in matches:
                score = min(100, round(match.similarity_score * 100))

                candidate = session.scalar(
                    select(Candidate).where(Candidate.id == uuid.UUID(match.resume_id))
                )
                if not candidate:
                    continue

                existing_match = session.scalar(
                    select(CandidateJobMatch).where(
                        CandidateJobMatch.candidate_id == candidate.id,
                        CandidateJobMatch.job_post_id == job_post.id,
                    )
                )
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
            session.commit()
            print(f"[process_job_post] committed {len(matches)} matches for job {job_post_id}")

    except Exception as exc:
        raise self.retry(exc=exc)
    finally:
        qdrant.close()
        engine.dispose()
