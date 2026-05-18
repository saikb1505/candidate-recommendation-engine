# Resume Matcher

A production-grade recruitment pipeline that ingests resumes, extracts structured data with LLMs, and matches candidates to job descriptions using semantic search and AI re-ranking.

## Architecture

```
Upload (PDF/DOCX/image)
    ↓
S3 / local storage
    ↓
Celery worker: ingest_resume
    ↓
Docling → Groq (text LLM)  ──(fail/incomplete)──→  GPT-4o mini (vision)
    ↓
Structured JSON (name, skills, experience, education…)
    ↓
Duplicate check (by email in Postgres)
    ↓
pgvector (768-dim vectors via sentence-transformers)
    ↓
POST /jobs/ → Celery worker: process_job_post
    ↓
Semantic search → Claude Haiku re-ranks with explanations
    ↓
GET /jobs/{id}/matches → ranked candidates
```

## API

### Resumes

```
POST /resumes/upload
  multipart/form-data, field: file (PDF/DOCX/DOC/JPG/PNG/WEBP, max 10 MB)
  → { candidate_id, task_id, status: "processing" }

GET /candidates/{candidate_id}
  → { id, name, email, location, skills, total_years_experience, status }
```

### Jobs

```
POST /jobs/
  { title, description, required_skills[], min_years, location }
  → { job_post_id, task_id, status: "processing" }

GET /jobs/{job_post_id}/matches
  → [{ candidate_id, match_score, similarity_score, match_explanation }]
     sorted by match_score desc
```

### Task polling

```
GET /tasks/{task_id}/status
  → { task_id, status, result }
  status: PENDING | SUCCESS | FAILURE
```

## Extraction pipeline

Extraction runs in two stages with automatic fallback:

1. **Docling → Groq** (`llama-3.3-70b-versatile`) — converts the file to markdown then calls a text LLM. Fast and cheap.
2. **GPT-4o mini (vision)** — used if Groq fails or returns a result missing name/email/experience. Converts pages to base64 images and sends them to the vision model.

Both paths return the same JSON schema: contact info, title, summary, skills, total years of experience, work history, and education.

## Matching pipeline

**`fast_match`** — vector search + metadata filters only. Free, sub-100ms. Used for browsing.

**`smart_match`** — used for job posts:
1. Claude Haiku parses the job description → `{ skills, min_years, location, role_summary }`
2. pgvector returns up to 30 candidates filtered by those requirements
3. Claude Haiku re-ranks them with a 1–10 match score and a two-sentence explanation per candidate
4. Top matches are written to `candidate_job_matches` in Postgres

## Setup

### System dependencies (macOS)

```bash
brew install poppler libmagic
brew install --cask libreoffice
```

### Python dependencies

```bash
pip install -r requirements.txt
```

### Environment variables

```env
# LLM providers
GROQ_API_KEY=gsk_...
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost/resume_matcher

# Redis
REDIS_URL=redis://localhost:6379/0

# Storage: "local" (default) or "s3"
STORAGE_BACKEND=local
S3_BUCKET=resume-matcher-files   # only needed if STORAGE_BACKEND=s3
AWS_REGION=us-east-1
```

### Database migrations

```bash
alembic upgrade head
```

### Run

```bash
# API
uvicorn api.main:app --reload

# Worker (separate terminal)
# Note: uses solo pool by default (safe on macOS with PyTorch).
# Switch to prefork in production on Linux.
celery -A workers.celery_app worker --loglevel=info
```

## Bulk CLI scripts

For processing a local folder of resumes outside the API:

```bash
# Extract resumes from data/sample_resumes/ (concurrent, 5 workers)
python run_pipeline.py --concurrent

# Embed extracted resumes into pgvector
python run_embeddings.py

# Interactive recommendation test
python run_recommend.py
```

## Project layout

```
api/main.py                   FastAPI app + all routes
config/settings.py            All config loaded from .env
config/schema.py              Pydantic model for extracted resume JSON
ingestion/
  scanner.py                  File discovery, file-hash deduplication, manifest
  converter.py                PDF/DOCX/image → JPEG page images (for vision fallback)
  pipeline.py                 Orchestrates scan → extract → validate for bulk runs
extraction/
  llm_extractor.py            Groq (primary) + GPT-4o mini vision (fallback)
  validator.py                Cleans and validates LLM output against schema
embeddings/
  embedder.py                 sentence-transformers wrapper (all-mpnet-base-v2)
  vector_store.py             pgvector: semantic search, metadata filter
deduplication/
  dedup.py                    Contact-based dedup (email / phone / name+location)
recommendation/
  ranker.py                   fast_match and smart_match
db/
  models.py                   Candidate, JobPost, CandidateJobMatch ORM models
  session.py                  Async session factory
storage/s3.py                 Local filesystem or S3, same interface
workers/
  celery_app.py               Celery config
  tasks.py                    ingest_resume and process_job_post tasks
```

## Data models

**Candidate** — `id`, `name`, `email` (unique), `phone`, `location`, `skills[]`, `total_years_experience`, `highest_education`, `summary`, `s3_key`, `status`, `embedding` (768-dim), `embedding_text`, `raw_data` (full JSON), `created_at`

**JobPost** — `id`, `title`, `description`, `required_skills[]`, `min_years`, `location`, `status`, `created_at`

**CandidateJobMatch** — `candidate_id`, `job_post_id`, `match_score` (0–100), `similarity_score` (cosine), `match_explanation` (text), unique on `(candidate_id, job_post_id)`

## Deduplication

Two layers run independently:

1. **File hash** (bulk CLI only) — identical files are skipped before any LLM call
2. **Email uniqueness** (API + CLI) — enforced in Postgres with a unique constraint; the Celery task handles race conditions between concurrent workers by catching `IntegrityError` and deleting the losing row and its S3 file
