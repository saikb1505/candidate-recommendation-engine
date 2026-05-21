from sentence_transformers import SentenceTransformer

from config.schema import ResumeChunk


_model: SentenceTransformer | None = None

# BGE models require this prefix on queries but NOT on documents.
# Omitting it on queries silently degrades retrieval quality.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print("  Loading embedding model (first time only)...")
        _model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        print("  Model loaded.")
    return _model


def embed_text(text: str) -> list[float]:
    """Encode a search query. Applies BGE query prefix."""
    model = get_model()
    embedding = model.encode(_BGE_QUERY_PREFIX + text, convert_to_numpy=True)
    return embedding.tolist()


def embed_chunks_batch(chunks: list[ResumeChunk]) -> list[list[float]]:
    """Encode resume chunks as documents. No query prefix."""
    model = get_model()
    texts = [chunk.text for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embeddings.tolist()
