# Resume Matcher - RAG-Powered Recruitment System

## Project Overview

A production-grade RAG (Retrieval-Augmented Generation) application that processes resumes and matches them to job descriptions using semantic search and LLM-powered re-ranking. Built to handle 10K+ resumes with cost optimization and intelligent deduplication.

**Status:** Core pipeline complete (Phases 1-6). Ready for Phase 7 (FastAPI) and production deployment.

**Stack:** Python, Claude (Haiku/Sonnet), sentence-transformers, ChromaDB, Pydantic

## What It Does

1. **Ingestion:** Processes PDF/DOCX/image resumes → converts to page images → LLM extracts structured JSON
2. **Deduplication:** Two-layer approach (file hash + contact info) removes duplicates before embedding
3. **Embedding:** Converts resume text to 768-dim vectors using sentence-transformers (local, free)
4. **Search:** Hybrid search combining semantic similarity + metadata filters (skills, years, location)
5. **Recommendation:** Parses job descriptions, retrieves candidates, re-ranks with LLM explanations

## Project Structure

```
resume-matcher/
├── config/
│   ├── settings.py          # Central config (paths, LLM models, thresholds)
│   └── schema.py            # ResumeSchema (Pydantic) - defines JSON structure
├── ingestion/
│   ├── scanner.py           # Find files, detect types, file hash dedup
│   ├── converter.py         # Any format → page images (PDF/DOCX/IMG)
│   └── pipeline.py          # Orchestrator: scan → convert → extract → validate
├── extraction/
│   ├── llm_extractor.py     # Images → Claude → JSON (with retry + fallback)
│   └── validator.py         # Clean and validate LLM output
├── embeddings/
│   ├── embedder.py          # sentence-transformers wrapper (all-mpnet-base-v2)
│   └── vector_store.py      # ChromaDB interface (add, search, filters)
├── deduplication/
│   └── dedup.py             # Contact-based dedup (email/phone/name+location)
├── recommendation/
│   └── ranker.py            # fast_match + smart_match with LLM re-ranking
├── data/
│   ├── sample_resumes/      # Input: drop resumes here
│   └── output/
│       ├── json/            # Extracted resume JSON files
│       ├── chromadb/        # Vector database storage
│       └── manifest.json    # Processing status tracker
├── run_pipeline.py          # CLI: process resumes (--resume --concurrent flags)
├── run_embeddings.py        # CLI: dedup + embed + store in ChromaDB
└── run_recommend.py         # CLI: test recommendation engine
```

## Key Design Decisions

### Why Vision-Based Extraction?
- **One path for all formats:** PDF/DOCX/images all become page images → Claude vision
- **Handles complex layouts:** Multi-column, tables, charts, creative designs
- **Cost at scale:** 10K resumes = ~$37 with Haiku Batch API (vs hybrid text/vision complexity)

### Why Two-Layer Deduplication?
- **Layer 1 (file hash):** Before extraction, saves API costs on identical files
- **Layer 2 (contact fields):** After extraction, catches same person with different filenames
- **Why not embeddings:** Free vs paid, instant vs slow, works before embedding step

### Why Local Embeddings?
- **Cost:** sentence-transformers is free, runs locally
- **Speed:** 10K resumes embedded in ~5-10 minutes on CPU
- **Quality:** all-mpnet-base-v2 (768 dims) is excellent for resume matching

### Why Hybrid Search?
- **Semantic (embeddings):** Finds conceptual matches ("backend engineer" ≈ "server-side developer")
- **Metadata filters:** Exact requirements (skills contains "Python", years >= 5)
- **ChromaDB workaround:** Post-filter in Python due to v1.5.8 operator limitations

### Why Two Recommendation Modes?
- **fast_match:** Free, instant, vector search + filters only (browsing candidates)
- **smart_match:** ~$0.01, 2-3s, LLM re-ranks with explanations (hiring decisions)

## Current Capabilities (What Works)

✅ Process 700 resumes in ~30 minutes (sequential) or ~7 minutes (concurrent)  
✅ File hash dedup removes exact duplicates before API calls  
✅ Contact-based dedup (email/phone/name+location) removes near-duplicates  
✅ ~400+ unique resumes embedded and stored in ChromaDB  
✅ Semantic search: "Python developer with cloud experience" → relevant candidates  
✅ Filtered search: skills + location + min years combined with semantic similarity  
✅ Job description parsing: messy JD text → structured requirements  
✅ LLM re-ranking: candidates ranked with AI-generated match explanations  

## Next Steps (Phase 7 - FastAPI)

### Endpoints to Build

```python
# 1. Extract resume
POST /extract
Body: multipart/form-data (file upload)
Response: ResumeSchema JSON
Implementation: calls pipeline.process_single()

# 2. Search candidates (fast)
GET /search?q=Python+developer&skills=Python&location=Bangalore&min_years=3
Response: List[CandidateMatch]
Implementation: calls ranker.fast_match()

# 3. Match to job description (smart)
POST /match
Body: {"job_description": "full JD text...", "top_k": 10}
Response: List[CandidateMatch] with explanations
Implementation: calls ranker.smart_match()

# 4. Health check
GET /health
Response: {"status": "ok", "resumes_count": 479}
```

### Implementation Notes

- All core functions are already designed for API use (no refactoring needed)
- Use Pydantic models directly as request/response schemas
- Add CORS middleware for frontend integration
- Consider rate limiting on /match (it costs $0.01 per call)
- Stream results for /match to show progress (LLM call takes 2-3s)

## Production Deployment Considerations

### What Changes for Production

**Current (local):**
- CLI scripts, ChromaDB local files, sentence-transformers on CPU
- Manual runs, print statements for logging
- Single-machine, no auth

**Production:**
- FastAPI + Uvicorn/Gunicorn, Qdrant Cloud or Pinecone
- Logging (structlog), metrics (Prometheus), error tracking (Sentry)
- Docker container, deployed to Railway/AWS/GCP
- JWT auth, rate limiting, file upload validation

### What Stays the Same
Core modules (extractor, embedder, validator, ranker) don't change. Only infrastructure around them.

## Cost Analysis (Actual)

**Extraction (700 resumes):**
- ~$5-6 with Claude Haiku (sequential API calls)
- Can reduce 50% with Batch API (24hr turnaround acceptable for bulk)
- Can reduce 90% on system prompt tokens with prompt caching

**Embeddings (700 resumes):**
- $0 (local sentence-transformers)

**Search:**
- $0 (local ChromaDB)

**Smart match per search:**
- ~$0.01-0.02 (2 Haiku calls: parse JD + re-rank candidates)

**Total to process 10K resumes + run 100 searches:**
- Extraction: ~$75 (one-time)
- Searches: ~$1-2
- Grand total: ~$77

## Known Issues / Technical Debt

1. **ChromaDB 1.5.8 operator limitations:** `$contains` doesn't work on strings, using Python post-filtering instead
2. **No evaluation metrics:** Precision@k, recall not measured (would need labeled test data)
3. **Single-threaded embedding:** Could use GPU or batch parallelization for faster processing
4. **No async API calls:** ThreadPoolExecutor for concurrent extraction, could use `httpx.AsyncClient`
5. **Empty resume handling:** Resumes with no extracted content (score 0.0) are stored but pollute search
6. **Phone normalization:** Only handles Indian format (last 10 digits), needs international support

## Environment Setup

```bash
# System dependencies (macOS)
brew install poppler libmagic
brew install --cask libreoffice

# Python dependencies
pip install pydantic python-magic pdf2image Pillow anthropic sentence-transformers chromadb

# Environment variables
export ANTHROPIC_API_KEY="sk-ant-..."

# Run
python run_pipeline.py --concurrent
python run_embeddings.py
python run_recommend.py
```

## Example Usage

```python
# Example 1: Process resumes
from ingestion.pipeline import process_batch_concurrent
results = process_batch_concurrent(max_workers=5)
# → 479 unique resumes extracted

# Example 2: Search with filters
from recommendation.ranker import fast_match
candidates = fast_match(
    query="Python backend developer",
    required_skills=["Python"],
    location="Hyderabad",
    min_years=3,
    top_k=10
)
# → Instant results, free

# Example 3: Smart match with JD
from recommendation.ranker import smart_match
jd = """
Looking for a Senior Software Engineer with 3+ years Python experience.
Must know REST APIs, AWS, and SQL. Location: Bangalore.
"""
candidates = smart_match(job_description=jd, top_k=10)
# → AI-ranked with explanations, ~$0.01
```

## Testing Strategy

**Current:**
- Manual testing via CLI scripts
- Sample resumes in data/sample_resumes/
- Visual inspection of extracted JSON
- Ad-hoc search queries

**Should Add:**
- Unit tests for each module (pytest)
- Integration tests for full pipeline
- Evaluation metrics (precision@k on labeled test set)
- Load testing (can it handle 10K concurrent searches?)
- Regression tests (does updated code still extract correctly?)

## Performance Characteristics

**Extraction:**
- Sequential: ~2-3 seconds per resume (700 resumes = 30 mins)
- Concurrent (5 workers): ~0.5 seconds per resume (700 resumes = 7 mins)
- Bottleneck: API rate limits (can't go beyond ~10 workers)

**Embedding:**
- CPU: ~5-10 minutes for 500 resumes
- GPU: ~1-2 minutes for 500 resumes
- Bottleneck: Model inference speed

**Search:**
- ChromaDB query: <50ms for 500 resumes
- Python post-filtering: negligible
- Total search latency: <100ms
- Scales to ~100K resumes before needing optimization

## Future Enhancements (Beyond FastAPI)

1. **Evaluation module:** Measure precision@k, recall, NDCG on test data
2. **Streamlit/React frontend:** Visual interface for recruiters
3. **Fine-tuned embeddings:** Train sentence-transformers on (JD, resume) pairs
4. **PostgreSQL metadata store:** Richer queries, better filtering at scale
5. **Batch processing queue:** Celery + Redis for async bulk uploads
6. **Resume parser service:** Reusable extraction API for other applications
7. **Multi-language support:** i18n for global resume databases
8. **Skills taxonomy:** Normalize "Python" vs "Python3" vs "Python programming"
9. **Explainability:** Highlight which resume sections matched which JD requirements
10. **A/B testing:** Compare recommendation quality across different models/strategies

## Learning Outcomes

This project demonstrates:
- End-to-end ML pipeline design (not just model training)
- Cost/benefit tradeoffs in production AI systems
- Hybrid search patterns (semantic + structured filters)
- LLM integration beyond chat (extraction, parsing, re-ranking)
- Error handling and resumable processing at scale
- Practical RAG architecture (retrieve → augment → generate)

## References & Resources

- **Claude API:** https://docs.anthropic.com/
- **sentence-transformers:** https://www.sbert.net/
- **ChromaDB:** https://docs.trychroma.com/
- **RAG pattern:** https://arxiv.org/abs/2005.11401
- **Embedding models comparison:** https://huggingface.co/spaces/mteb/leaderboard

---

**Project started:** April 2026  
**Current phase:** 6/7 (Recommendation engine complete, FastAPI pending)  
**Status:** Production-ready core, needs web interface for deployment