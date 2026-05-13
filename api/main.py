from dotenv import load_dotenv
load_dotenv()

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import config
from db.session import get_db, engine
from db.models import Base, Candidate, JobPost, CandidateJobMatch
from storage.s3 import upload_file
from workers.tasks import ingest_resume, process_job_post
from workers.celery_app import celery_app


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(title="Resume Matcher API", lifespan=lifespan)


# ──────────────────────────────────────────────
#  Flow 1: Resume upload
# ──────────────────────────────────────────────

@app.post("/resumes/upload", status_code=202)
async def upload_resume(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if not suffix or suffix not in config.supported_extensions:
        raise HTTPException(400, f"Unsupported file type: {suffix or '(unknown)'}")

    candidate_id = str(uuid.uuid4())
    s3_key = f"{candidate_id}{suffix}"

    content = await file.read(config.max_upload_bytes + 1)
    if len(content) > config.max_upload_bytes:
        raise HTTPException(413, f"File exceeds maximum size of {config.max_upload_bytes // (1024 * 1024)} MB")

    upload_file(content, s3_key, file.content_type or "application/octet-stream")

    candidate = Candidate(id=uuid.UUID(candidate_id), s3_key=s3_key, status="pending")
    db.add(candidate)
    await db.commit()

    task = ingest_resume.delay(s3_key, candidate_id)  # type: ignore[attr-defined]

    return {"candidate_id": candidate_id, "task_id": task.id, "status": "processing"}


@app.get("/candidates/{candidate_id}")
async def get_candidate(candidate_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Candidate).where(Candidate.id == uuid.UUID(candidate_id))
    )
    candidate = result.scalar_one_or_none()
    if not candidate:
        raise HTTPException(404, "Candidate not found")

    return {
        "id": str(candidate.id),
        "name": candidate.name,
        "email": candidate.email,
        "location": candidate.location,
        "skills": candidate.skills,
        "total_years_experience": candidate.total_years_experience,
        "status": candidate.status,
    }


# ──────────────────────────────────────────────
#  Flow 2: Job post + matching
# ──────────────────────────────────────────────

class JobPostCreate(BaseModel):
    title: str = ""
    description: str
    required_skills: list[str] = []
    min_years: float = 0.0
    location: str = ""


@app.post("/jobs/", status_code=202)
async def create_job_post(
    body: JobPostCreate,
    db: AsyncSession = Depends(get_db),
):
    job_post_id = str(uuid.uuid4())

    job_post = JobPost(
        id=uuid.UUID(job_post_id),
        title=body.title,
        description=body.description,
        required_skills=body.required_skills,
        min_years=body.min_years,
        location=body.location,
        status="pending",
    )
    db.add(job_post)
    await db.commit()

    task = process_job_post.delay(job_post_id)  # type: ignore[attr-defined]

    return {"job_post_id": job_post_id, "task_id": task.id, "status": "processing"}


@app.get("/jobs/{job_post_id}/matches")
async def get_job_matches(job_post_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(CandidateJobMatch)
        .where(CandidateJobMatch.job_post_id == uuid.UUID(job_post_id))
        .order_by(CandidateJobMatch.match_score.desc())
    )
    matches = result.scalars().all()

    return [
        {
            "candidate_id": str(m.candidate_id),
            "match_score": m.match_score,
            "similarity_score": m.similarity_score,
            "match_explanation": m.match_explanation,
        }
        for m in matches
    ]


# ──────────────────────────────────────────────
#  Task status (works for both flows)
# ──────────────────────────────────────────────

@app.get("/tasks/{task_id}/status")
async def get_task_status(task_id: str):
    result = celery_app.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": result.status,
        "result": result.result if result.ready() else None,
    }
