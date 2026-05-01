"""
Embedding generator — converts resume JSON into vectors.

This is the bridge between structured data and semantic search.
It takes the natural language text from ResumeSchema.to_embedding_text()
and converts it into a dense vector (list of 768 numbers).

KEY CONCEPTS:

  What is an embedding?
    A fixed-size list of numbers (e.g. 768 floats) that captures
    the MEANING of text. Similar meanings → similar numbers.
    "Python backend developer" and "server-side Python engineer"
    produce nearly identical vectors, even though the words differ.

  Why sentence-transformers?
    It's the standard library for generating text embeddings.
    The model runs LOCALLY — no API calls, no cost per resume.
    Once downloaded (~100MB), it processes thousands of texts
    per minute on a laptop CPU.

  Which model?
    all-MiniLM-L6-v2:  384 dimensions, fastest, good enough for most cases
    all-mpnet-base-v2:  768 dimensions, best quality, what we use
    The quality difference matters for search relevance.
    768 dims captures more nuance than 384 — worth it for resumes
    where subtle differences in experience matter.

  What about embedding via Claude API?
    Anthropic doesn't offer an embedding API. You could use
    OpenAI's embedding API, but sentence-transformers is free,
    runs locally, and produces excellent results for this use case.
    No reason to pay per-embedding when local works great.
"""

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from config.settings import config
from config.schema import ResumeSchema


# ──────────────────────────────────────────────
#  MODEL LOADING
# ──────────────────────────────────────────────

# Module-level variable — model loads once, reused across calls.
# First call downloads the model (~100MB), subsequent calls are instant.
#
# WHY module-level?
#   Loading the model takes 2-3 seconds. If you loaded it inside
#   every function call, processing 10K resumes would waste hours
#   just on model loading. Load once, use everywhere.

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """
    Load the embedding model (lazy initialization).

    WHY lazy?
    The model only loads when you first call this function,
    not when you import the module. This means importing
    embedder.py in FastAPI doesn't slow down server startup
    unless an embedding endpoint is actually called.
    """
    global _model
    if _model is None:
        print("  Loading embedding model (first time only)...")
        _model = SentenceTransformer("all-mpnet-base-v2")
        print("  Model loaded.")
    return _model


# ──────────────────────────────────────────────
#  SINGLE RESUME EMBEDDING
# ──────────────────────────────────────────────

def embed_text(text: str) -> list[float]:
    """
    Convert a text string into a vector embedding.

    Args:
        text: any text string (resume summary, job description, etc.)

    Returns:
        List of 768 floats representing the text's meaning.

    This is the lowest-level function. Everything else calls this.
    It works for resumes AND job descriptions — same model,
    same vector space. That's why similarity search works:
    both resume and JD vectors live in the same 768-dim space.
    """
    model = get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def embed_resume(resume: ResumeSchema) -> list[float]:
    """
    Convert a ResumeSchema into a vector embedding.

    Uses to_embedding_text() to flatten the resume into
    natural language first, then embeds that text.

    WHY not embed the raw JSON?
    JSON structure adds noise: {"title": "Engineer"} carries
    meaning in the word "Engineer" but the braces and key name
    dilute the embedding. Natural language is what the model
    was trained on — it produces better vectors.
    """
    text = resume.to_embedding_text()

    if not text.strip():
        # Empty resume — return zero vector
        # This shouldn't happen if extraction worked,
        # but defensive coding prevents downstream crashes
        model = get_model()
        dim = model.get_embedding_dimension()
        return [0.0] * dim

    return embed_text(text)


# ──────────────────────────────────────────────
#  BATCH EMBEDDING
# ──────────────────────────────────────────────

def embed_resumes_batch(resumes: list[ResumeSchema]) -> list[list[float]]:
    """
    Embed multiple resumes at once.

    Args:
        resumes: list of ResumeSchema objects

    Returns:
        List of embeddings (same order as input).

    WHY batch instead of one-by-one?
    sentence-transformers can process multiple texts in
    parallel on the GPU/CPU. Batching 100 resumes is 10x
    faster than embedding them one at a time.

    For 10K resumes, this function processes all of them
    in about 5-10 minutes on a laptop CPU.
    """
    model = get_model()

    texts = []
    for resume in resumes:
        text = resume.to_embedding_text()
        texts.append(text if text.strip() else "empty resume")

    # batch_size=32 is a good default for CPU
    # Increase to 64 or 128 if you have a GPU
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    return embeddings.tolist()


# ──────────────────────────────────────────────
#  UTILITY — LOAD RESUMES FROM JSON FILES
# ──────────────────────────────────────────────

def load_all_resumes() -> list[ResumeSchema]:
    """
    Load all extracted JSON files from the output directory.

    This connects Phase 2 (extraction) to Phase 3 (embedding).
    The extraction pipeline saves JSON files to data/output/json/.
    This function loads them all back as ResumeSchema objects.

    Returns:
        List of ResumeSchema objects ready for embedding.
    """
    json_dir = config.json_dir
    resumes = []

    if not json_dir.exists():
        print(f"  No JSON directory found: {json_dir}")
        return resumes

    for json_file in sorted(json_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text())
            resume = ResumeSchema(**data)
            resumes.append(resume)
        except Exception as e:
            print(f"  Failed to load {json_file.name}: {e}")

    print(f"  Loaded {len(resumes)} resumes from {json_dir}")
    return resumes


# ──────────────────────────────────────────────
#  SIMILARITY — FOR TESTING AND DEDUPLICATION
# ──────────────────────────────────────────────

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Returns a value between -1 and 1:
      1.0  = identical meaning
      0.0  = completely unrelated
     -1.0  = opposite meaning (rare in practice)

    For resumes:
      > 0.95 = likely duplicate (same person, same content)
      > 0.80 = very similar (same role/industry)
      > 0.60 = somewhat related
      < 0.40 = different fields entirely

    WHY implement this manually?
    You could use numpy or scipy, but understanding the math
    helps you reason about similarity thresholds. Cosine
    similarity is just: dot(A,B) / (|A| * |B|)

    This function is also used in deduplication (Phase 5)
    and is useful for debugging search results.
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = sum(a * a for a in vec_a) ** 0.5
    magnitude_b = sum(b * b for b in vec_b) ** 0.5

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)