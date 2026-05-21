import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchText,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Range,
    VectorParams,
)

from config.settings import config
from config.schema import ResumeChunk
from embeddings.embedder import embed_text

_client: AsyncQdrantClient | None = None


def get_qdrant() -> AsyncQdrantClient:
    global _client
    if _client is None:
        _client = AsyncQdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key or None,
        )
    return _client


def make_qdrant_client() -> AsyncQdrantClient:
    """Create a fresh client per Celery task (each task runs its own event loop)."""
    return AsyncQdrantClient(
        url=config.qdrant_url,
        api_key=config.qdrant_api_key or None,
    )


async def setup_collection() -> None:
    """Create the collection and payload indexes if they don't exist yet."""
    client = get_qdrant()

    if await client.collection_exists(config.qdrant_collection):
        return

    await client.create_collection(
        collection_name=config.qdrant_collection,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

    indexes = {
        "candidate_id":           PayloadSchemaType.KEYWORD,
        "skills":                 PayloadSchemaType.KEYWORD,
        "status":                 PayloadSchemaType.KEYWORD,
        "location":               PayloadSchemaType.KEYWORD,
        "total_years_experience": PayloadSchemaType.FLOAT,
    }
    for field, schema in indexes.items():
        await client.create_payload_index(config.qdrant_collection, field, schema)

    print(f"  Qdrant collection '{config.qdrant_collection}' created with indexes.")


def _chunk_point_id(candidate_id: str, chunk_type: str, chunk_index: int) -> str:
    """Deterministic UUID per chunk — makes upserts idempotent."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{candidate_id}:{chunk_type}:{chunk_index}"))


async def upsert_chunks(
    candidate_id: str,
    chunks: list[ResumeChunk],
    embeddings: list[list[float]],
    payload_meta: dict,
    client: AsyncQdrantClient | None = None,
) -> None:
    """
    Write chunk vectors to Qdrant. Idempotent — safe to call multiple times
    for the same candidate thanks to deterministic point IDs.

    payload_meta keys: name, email, location, skills (list), total_years_experience,
                       highest_education, status.
    """
    cl = client or get_qdrant()

    points = [
        PointStruct(
            id=_chunk_point_id(candidate_id, chunk.chunk_type, chunk.chunk_index),
            vector=embedding,
            payload={
                "candidate_id": candidate_id,
                "chunk_type":   chunk.chunk_type,
                "chunk_index":  chunk.chunk_index,
                "chunk_text":   chunk.text,
                # Candidate-level metadata denormalised into every chunk so
                # Qdrant can filter without joining back to Postgres.
                "name":                    payload_meta.get("name", ""),
                "email":                   payload_meta.get("email", ""),
                "location":                payload_meta.get("location", ""),
                "skills":                  [s.lower() for s in payload_meta.get("skills", [])],
                "total_years_experience":  payload_meta.get("total_years_experience", 0.0),
                "highest_education":       payload_meta.get("highest_education", ""),
                "status":                  payload_meta.get("status", "processed"),
            },
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]

    await cl.upsert(collection_name=config.qdrant_collection, points=points)


async def delete_candidate_chunks(
    candidate_id: str,
    client: AsyncQdrantClient | None = None,
) -> None:
    """Delete all chunks for a candidate — used when re-ingesting."""
    cl = client or get_qdrant()
    await cl.delete(
        collection_name=config.qdrant_collection,
        points_selector=Filter(
            must=[FieldCondition(key="candidate_id", match=MatchValue(value=candidate_id))]
        ),
    )


def _hit_to_dict(hit) -> dict:
    p = hit.payload or {}
    return {
        "id":    p.get("candidate_id", ""),
        "score": round(float(hit.score), 4),
        "metadata": {
            "name":                   p.get("name", ""),
            "email":                  p.get("email", ""),
            "location":               p.get("location", ""),
            "skills":                 ", ".join(p.get("skills", [])),
            "total_years_experience": p.get("total_years_experience", 0.0),
            "highest_education":      p.get("highest_education", ""),
        },
        "document": p.get("chunk_text", ""),
    }


async def search_by_chunks(
    query_text: str,
    top_k: int = 10,
    min_years: float | None = None,
    session=None,  # unused — kept so callers don't need updating
    client: AsyncQdrantClient | None = None,
) -> list[dict]:
    """
    Search by dense vector similarity. Returns the best-matching chunk per
    candidate (MAX score via group_by), so each candidate appears once.
    """
    query_embedding = embed_text(query_text)
    client = client or get_qdrant()

    conditions = [FieldCondition(key="status", match=MatchValue(value="processed"))]
    if min_years is not None:
        conditions.append(
            FieldCondition(key="total_years_experience", range=Range(gte=min_years))
        )

    result = await client.query_points_groups(
        collection_name=config.qdrant_collection,
        query=query_embedding,
        group_by="candidate_id",
        limit=top_k,
        group_size=1,
        query_filter=Filter(must=conditions),
        with_payload=True,
    )

    return [_hit_to_dict(group.hits[0]) for group in result.groups if group.hits]


async def search_with_skills_by_chunks(
    query_text: str,
    required_skills: list[str] | None = None,
    min_years: float | None = None,
    location: str | None = None,
    top_k: int = 10,
    session=None,  # unused — kept so callers don't need updating
    client: AsyncQdrantClient | None = None,
) -> list[dict]:
    """
    Vector search with native Qdrant payload filtering.
    Filters are applied during HNSW traversal — no Python post-filtering,
    no over-fetching, no missed candidates.

    Skills use AND logic: candidate must have ALL required skills.
    Skills are stored lowercase; query skills are lowercased before matching.
    """
    query_embedding = embed_text(query_text)
    client = client or get_qdrant()

    conditions = [FieldCondition(key="status", match=MatchValue(value="processed"))]

    if min_years is not None:
        conditions.append(
            FieldCondition(key="total_years_experience", range=Range(gte=min_years))
        )

    if required_skills:
        for skill in required_skills:
            conditions.append(
                FieldCondition(key="skills", match=MatchValue(value=skill.lower()))
            )

    if location:
        conditions.append(
            FieldCondition(key="location", match=MatchText(text=location))
        )

    result = await client.query_points_groups(
        collection_name=config.qdrant_collection,
        query=query_embedding,
        group_by="candidate_id",
        limit=top_k,
        group_size=1,
        query_filter=Filter(must=conditions),
        with_payload=True,
    )

    return [_hit_to_dict(group.hits[0]) for group in result.groups if group.hits]


async def get_chunk_count() -> int:
    client = get_qdrant()
    result = await client.count(collection_name=config.qdrant_collection, exact=False)
    return result.count
