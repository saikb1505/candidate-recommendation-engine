# Resume Matcher

A production-grade recruitment pipeline that ingests resumes, extracts structured data with LLMs, and matches candidates to job descriptions using semantic search and AI re-ranking.

## Architecture

```
Upload (PDF/DOCX/image or ZIP)
    ↓
Local filesystem or S3
    ↓
Celery worker: ingest_resume
    ↓
Docling → Groq llama-4-scout (text)  ──(fail/incomplete)──→  Groq llama-4-scout (vision)
    ↓
Structured JSON (name, skills, experience, education…)
    ↓
Duplicate check (by email in Postgres)
    ↓
Chunk resume → BAAI/bge-base-en-v1.5 (768-dim) → Qdrant
    ↓
POST /jobs/ → Celery worker: process_job_post
    ↓
Semantic search (Qdrant) → Claude Haiku re-ranks with explanations
    ↓
GET /jobs/{id}/matches → ranked candidates
```

## API

### Resumes

```
POST /resumes/upload
  multipart/form-data, field: file (PDF/DOCX/DOC/JPG/PNG/WEBP, max 10 MB)
  → { candidate_id, task_id, status: "processing" }

POST /resumes/upload-zip
  multipart/form-data, field: file (.zip, max 500 MB)
  Extracts all PDF/DOCX/DOC files and queues each one independently.
  → { accepted, rejected, results: [{ filename, candidate_id, task_id, status }] }

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

Extraction runs in two stages with automatic fallback, both using the same Groq model:

1. **Docling → Groq text** (`meta-llama/llama-4-scout-17b-16e-instruct`) — converts the file to markdown then calls the LLM as a text model. Fast and cheap.
2. **Groq vision** — used if the text pass fails or returns a result missing name/email/experience/skills. Converts pages to base64 JPEG images and sends them to the same model in multimodal mode.

Both paths return the same JSON schema: contact info, title, summary, skills, total years of experience, work history, and education.

## Matching pipeline

**`fast_match`** — vector search + metadata filters only. Free, sub-100ms. Used for browsing.

**`smart_match`** — used for job posts:
1. Claude Haiku parses the job description → `{ skills, min_years, location, role_summary }`
2. Qdrant returns up to 30 candidates filtered by those requirements (grouped by candidate to avoid chunk duplication)
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

### Services

- **PostgreSQL** — candidate/job data
- **Redis** — Celery broker and result backend
- **Qdrant** — vector store for semantic search (self-hosted or Qdrant Cloud)

### Environment variables

```env
# LLM providers
GROQ_API_KEY=gsk_...
ANTHROPIC_API_KEY=sk-ant-...

# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost/resume_matcher

# Redis
REDIS_URL=redis://localhost:6379/0

# Qdrant
QDRANT_URL=http://localhost:6333        # or your Qdrant Cloud URL
QDRANT_API_KEY=                          # leave empty for local Qdrant
QDRANT_COLLECTION=resume_chunks          # default collection name

# Storage: "local" (default) or "s3"
STORAGE_BACKEND=local
LOCAL_UPLOAD_DIR=data/uploads            # only for local storage
S3_BUCKET=resume-matcher-files           # only needed if STORAGE_BACKEND=s3
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

# Embed extracted resumes into Qdrant
python run_embeddings.py

# Interactive recommendation test
python run_recommend.py
```

## Project layout

```
api/main.py                   FastAPI app + all routes
config/settings.py            All config loaded from .env
config/schema.py              Pydantic models: ResumeSchema, ResumeChunk, ContactInfo, etc.
ingestion/
  scanner.py                  File discovery, file-hash deduplication, manifest tracking
  converter.py                PDF/DOCX/image → JPEG page images (for vision fallback)
  pipeline.py                 Orchestrates scan → extract → validate for bulk runs
extraction/
  llm_extractor.py            Groq text (primary) + Groq vision (fallback)
  validator.py                Cleans and validates LLM output against schema
embeddings/
  embedder.py                 BAAI/bge-base-en-v1.5 wrapper (768-dim, BGE query prefix)
  vector_store.py             Qdrant: chunk upsert, grouped semantic search, payload indexes
deduplication/
  dedup.py                    Contact-based dedup (email / phone / name+location) for bulk CLI
recommendation/
  ranker.py                   fast_match and smart_match
db/
  models.py                   Candidate, JobPost, Company, CandidateJobMatch ORM models
  session.py                  Async session factory
storage/s3.py                 Local filesystem or S3, same interface
workers/
  celery_app.py               Celery config (solo pool, Redis broker)
  tasks.py                    ingest_resume and process_job_post tasks
```

## Data models

**Candidate** — `id`, `name`, `email` (unique), `phone`, `location`, `skills[]`, `total_years_experience`, `highest_education`, `summary`, `s3_key`, `status`, `raw_data` (full JSON), `created_at`

**JobPost** — `id`, `title`, `description`, `required_skills[]`, `min_years`, `location`, `status`, `created_at`

**Company** — `id`, `name` (unique), `source`, `created_at` — populated from resume work history

**CandidateJobMatch** — `candidate_id`, `job_post_id`, `match_score` (0–100), `similarity_score` (cosine), `match_explanation` (text), unique on `(candidate_id, job_post_id)`

## Vector storage

Each resume is split into semantic chunks before embedding:

| Chunk type   | Content                                      |
|--------------|----------------------------------------------|
| `summary`    | Professional summary + skills list           |
| `experience` | One chunk per job entry (title, company, description) |
| `education`  | All education entries combined               |

Chunks are stored in Qdrant with payload indexes on `candidate_id`, `skills`, `location`, `total_years_experience`, and `status`. Search uses `query_points_groups` grouped by `candidate_id` so each candidate appears at most once in results regardless of how many chunks match.

## Deduplication

Two layers run independently:

1. **File hash** (bulk CLI only) — identical files are skipped before any LLM call
2. **Email uniqueness** (API + CLI) — enforced in Postgres with a unique constraint; the Celery task handles race conditions between concurrent workers by catching `IntegrityError` and deleting the losing row and its file

The bulk CLI additionally runs three-pass contact deduplication (email → phone → name+location) and keeps the most complete resume when duplicates are found.
