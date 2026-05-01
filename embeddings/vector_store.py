"""
Vector store — stores and searches resume embeddings.

This is the core of your RAG retrieval layer.
It wraps ChromaDB to provide:
  1. Store resume embeddings with metadata
  2. Search by text (semantic similarity)
  3. Search with filters (metadata + similarity combined)

WHY ChromaDB?
  - Runs locally, no server setup needed
  - Built-in support for metadata filtering
  - Handles similarity search internally (no manual cosine math)
  - Persistent storage — survives restarts
  - Free and open source

  For production at 1M+ resumes, you'd consider Qdrant,
  Weaviate, or Pinecone. But for learning and up to ~100K
  resumes, ChromaDB is perfect.

HOW CHROMADB WORKS (mental model):
  Think of it as a table with three column types:
    1. ID        — unique identifier for each resume
    2. Embedding — the 768-dim vector (searchable by similarity)
    3. Metadata  — key-value pairs (filterable by exact match)

  When you query:
    - ChromaDB finds the N closest vectors (semantic search)
    - Then filters those by metadata (exact match)
    - Returns results ranked by similarity score

  This two-step process is why you split data into
  embeddings (to_embedding_text) and metadata (to_metadata)
  back in schema.py. Now you see why that design decision
  mattered.
"""

import chromadb
from pathlib import Path

from config.settings import config
from config.schema import ResumeSchema
from embeddings.embedder import embed_text, embed_resume


# ──────────────────────────────────────────────
#  STORE INITIALIZATION
# ──────────────────────────────────────────────

# Module-level client — initialized once, reused everywhere.
# PersistentClient saves data to disk so it survives restarts.

_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None

COLLECTION_NAME = "resumes"


def get_collection() -> chromadb.Collection:
    """
    Get or create the resume collection.

    WHY lazy initialization?
    Same reason as the embedding model — don't pay the startup
    cost until you actually need it. FastAPI imports this module
    at startup, but the DB only initializes on first request.

    WHY PersistentClient?
    The default Client() stores everything in memory — it
    vanishes when your script exits. PersistentClient saves
    to disk at data/output/chromadb/. Your 10K resume vectors
    persist between runs. No re-embedding needed.
    """
    global _client, _collection

    if _collection is None:
        db_path = str(config.output_dir / "chromadb")
        _client = chromadb.PersistentClient(path=db_path)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"  ChromaDB collection '{COLLECTION_NAME}': "
              f"{_collection.count()} documents")

    return _collection


# ──────────────────────────────────────────────
#  ADDING RESUMES
# ──────────────────────────────────────────────

def add_resume(resume: ResumeSchema, resume_id: str) -> bool:
    """
    Add a single resume to the vector store.

    Args:
        resume: validated ResumeSchema object
        resume_id: unique identifier (usually filename stem)

    Returns:
        True if added successfully, False if failed.

    What gets stored:
      - ID: the resume_id you provide
      - Embedding: 768-dim vector from resume text
      - Metadata: filterable fields (skills, years, location)
      - Document: the raw embedding text (for debugging)

    WHY store the document text too?
    ChromaDB can return it with search results, which is
    useful for debugging. "Why did this resume match?"
    — look at the document text and compare with the query.
    """
    collection = get_collection()

    try:
        embedding = embed_resume(resume)
        metadata = resume.to_metadata()
        document = resume.to_embedding_text()

        collection.upsert(
            ids=[resume_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[document],
        )
        return True

    except Exception as e:
        print(f"  Failed to add {resume_id}: {e}")
        return False


def add_resumes_batch(
    resumes: list[ResumeSchema],
    resume_ids: list[str],
) -> int:
    """
    Add multiple resumes to the vector store at once.

    Args:
        resumes: list of ResumeSchema objects
        resume_ids: list of unique IDs (same order as resumes)

    Returns:
        Number of resumes successfully added.

    WHY batch?
    ChromaDB handles batch inserts more efficiently than
    individual inserts. For 10K resumes, batch insert takes
    seconds while individual inserts take minutes.

    WHY upsert instead of add?
    upsert = update if exists, insert if new.
    If you run the pipeline twice on the same resumes,
    upsert updates them instead of throwing a duplicate error.
    Idempotent operations make pipelines resumable.
    """
    collection = get_collection()

    from embeddings.embedder import embed_resumes_batch

    try:
        embeddings = embed_resumes_batch(resumes)
        metadatas = [r.to_metadata() for r in resumes]
        documents = [r.to_embedding_text() for r in resumes]

        collection.upsert(
            ids=resume_ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

        print(f"  Added {len(resumes)} resumes to vector store")
        return len(resumes)

    except Exception as e:
        print(f"  Batch insert failed: {e}")
        return 0


# ──────────────────────────────────────────────
#  SEARCHING
# ──────────────────────────────────────────────

def search(
    query_text: str,
    top_k: int = 10,
    filters: dict | None = None,
) -> list[dict]:
    """
    Search for resumes matching a query.
    """
    collection = get_collection()

    # Embed the query with the SAME model used for resumes
    query_embedding = embed_text(query_text)

    # Build query arguments
    query_args = {
        "query_embeddings": [query_embedding],  # ← was query_texts
        "n_results": top_k,
    }

    if filters:
        query_args["where"] = filters

    results = collection.query(**query_args)

    # Format results into a clean list of dicts
    formatted = []
    for i in range(len(results["ids"][0])):
        formatted.append({
            "id": results["ids"][0][i],
            "score": round(1 - results["distances"][0][i], 4),
            "metadata": results["metadatas"][0][i],
            "document": results["documents"][0][i],
        })

    return formatted

def search_with_skills(
    query_text: str,
    required_skills: list[str] | None = None,
    min_years: float | None = None,
    location: str | None = None,
    top_k: int = 10,
) -> list[dict]:
    """
    Search with filters applied in Python post-retrieval.

    WHY filter in Python instead of ChromaDB?
    ChromaDB 1.5+ doesn't support substring matching on
    string metadata fields. Rather than fighting the DB,
    we fetch more results than needed and filter in Python.
    
    This is a common pattern in production RAG systems:
      1. Vector DB returns top 50-100 candidates (fast, approximate)
      2. Python applies precise filters (slow but exact)
      3. Return top_k from the filtered set
    
    At 10K resumes, fetching 100 and filtering to 10 is
    still instant. This breaks down at 1M+ resumes where
    you'd need a proper DB with full-text search (PostgreSQL
    with pgvector, or Elasticsearch alongside the vector DB).
    """
    # Fetch more results than needed — we'll filter down
    fetch_k = min(top_k * 10, 100)

    # Only use ChromaDB filters for numeric comparisons ($gte works)
    db_filters = None
    if min_years is not None:
        db_filters = {"total_years_experience": {"$gte": min_years}}

    results = search(query_text, top_k=fetch_k, filters=db_filters)

    # Apply skill and location filters in Python
    filtered = []
    for result in results:
        metadata = result["metadata"]

        # Check skills
        if required_skills:
            resume_skills = metadata.get("skills", "").lower()
            if not all(skill.lower() in resume_skills for skill in required_skills):
                continue

        # Check location
        if location:
            resume_location = metadata.get("location", "").lower()
            if location.lower() not in resume_location:
                continue

        filtered.append(result)

        if len(filtered) >= top_k:
            break

    return filtered

# ──────────────────────────────────────────────
#  UTILITY
# ──────────────────────────────────────────────

def get_count() -> int:
    """How many resumes are in the store."""
    return get_collection().count()


def delete_all():
    """
    Clear the entire collection.

    Useful during development when you want to re-embed
    everything from scratch. In production, you'd never
    call this — you'd use upsert to update individual records.
    """
    global _collection
    if _client is not None:
        _client.delete_collection(COLLECTION_NAME)
        _collection = None
        print(f"  Deleted collection '{COLLECTION_NAME}'")