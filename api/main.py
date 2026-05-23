from dotenv import load_dotenv
load_dotenv()

import io
import uuid
import zipfile
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
from embeddings.vector_store import setup_collection


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    setup_collection()
    yield


app = FastAPI(title="Resume Matcher API", lifespan=lifespan)


# ──────────────────────────────────────────────
#  Flow 1: Resume upload
# ──────────────────────────────────────────────

@app.post("/resumes/upload", status_code=202)
async def upload_resume(
    file: UploadFile = File(...),
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

    task = ingest_resume.delay(s3_key, candidate_id)
    return {"candidate_id": candidate_id, "task_id": task.id, "status": "processing"}


MAX_ZIP_BYTES = 500 * 1024 * 1024  # 500 MB total ZIP limit


@app.post("/resumes/upload-zip", status_code=202)
async def upload_zip(
    file: UploadFile = File(...),
):
    # Must be a ZIP file
    filename = file.filename or ""
    if not filename.lower().endswith(".zip"):
        raise HTTPException(400, "Only .zip files are accepted")

    content = await file.read(MAX_ZIP_BYTES + 1)
    if len(content) > MAX_ZIP_BYTES:
        raise HTTPException(413, f"ZIP exceeds {MAX_ZIP_BYTES // (1024 * 1024)} MB limit")

    # Validate ZIP integrity
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile:
        raise HTTPException(400, "Invalid or corrupted ZIP file")

    results = []
    accepted = 0
    rejected = 0

    for entry in zf.infolist():
        name = Path(entry.filename).name

        # Skip directories and hidden/system files (e.g. macOS ._* resource forks)
        if entry.is_dir() or not name or name.startswith("."):
            continue

        suffix = Path(name).suffix.lower()
        if suffix not in {".pdf", ".docx", ".doc"}:
            continue

        # ZIP bomb guard: check declared uncompressed size before reading
        if entry.file_size > config.max_upload_bytes:
            results.append({"filename": name, "error": f"File exceeds {config.max_upload_bytes // (1024 * 1024)} MB limit"})
            rejected += 1
            continue

        file_bytes = zf.read(entry.filename)

        # Double-check actual size after decompression
        if len(file_bytes) > config.max_upload_bytes:
            results.append({"filename": name, "error": f"File exceeds {config.max_upload_bytes // (1024 * 1024)} MB limit"})
            rejected += 1
            continue

        candidate_id = uuid.uuid4()
        s3_key = f"{candidate_id}{suffix}"

        upload_file(file_bytes, s3_key, "application/octet-stream")
        task = ingest_resume.delay(s3_key, str(candidate_id))  # type: ignore[attr-defined]

        results.append({
            "filename": name,
            "candidate_id": str(candidate_id),
            "task_id": task.id,
            "status": "processing",
        })
        accepted += 1

    if accepted == 0 and rejected == 0:
        raise HTTPException(400, "ZIP contains no supported resume files")

    return {"accepted": accepted, "rejected": rejected, "results": results}


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
