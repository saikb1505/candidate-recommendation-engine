from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import AsyncSessionLocal
from embeddings.embedder import embed_text

EMBEDDING_DIM = 768


def _to_pg_vector(embedding: list[float]) -> str:
    """Format a float list as a Postgres vector literal."""
    return "[" + ",".join(str(v) for v in embedding) + "]"


async def search(
    query_text: str,
    top_k: int = 10,
    min_years: float | None = None,
    session: AsyncSession | None = None,
) -> list[dict]:
    query_embedding = embed_text(query_text)
    vec_str = _to_pg_vector(query_embedding)

    where_clause = "WHERE embedding IS NOT NULL AND status = 'processed'"
    params: dict = {"top_k": top_k, "vec": vec_str}

    if min_years is not None:
        where_clause += " AND total_years_experience >= :min_years"
        params["min_years"] = min_years

    sql = text(f"""
        SELECT
            id::text,
            name,
            email,
            location,
            array_to_string(skills, ', ') AS skills,
            total_years_experience,
            highest_education,
            embedding_text,
            1 - (embedding <=> CAST(:vec AS vector)) AS score
        FROM candidates
        {where_clause}
        ORDER BY embedding <=> CAST(:vec AS vector)
        LIMIT :top_k
    """)

    async def _execute(s: AsyncSession) -> list:
        return (await s.execute(sql, params)).mappings().all()

    if session is not None:
        rows = await _execute(session)
    else:
        async with AsyncSessionLocal() as s:
            rows = await _execute(s)

    return [
        {
            "id": row["id"],
            "score": round(float(row["score"]), 4),
            "metadata": {
                "name": row["name"],
                "email": row["email"],
                "location": row["location"],
                "skills": row["skills"],
                "total_years_experience": row["total_years_experience"],
                "highest_education": row["highest_education"],
            },
            "document": row["embedding_text"],
        }
        for row in rows
    ]


async def search_with_skills(
    query_text: str,
    required_skills: list[str] | None = None,
    min_years: float | None = None,
    location: str | None = None,
    top_k: int = 10,
    session: AsyncSession | None = None,
) -> list[dict]:
    """
    Fetch more candidates than needed from pgvector, then filter in Python.
    Skill and location matching use case-insensitive substring checks that
    are simpler to express in Python than in SQL for this schema.
    """
    fetch_k = min(top_k * 10, 100)
    results = await search(query_text, top_k=fetch_k, min_years=min_years, session=session)

    filtered = []
    for result in results:
        metadata = result["metadata"]

        if required_skills:
            resume_skills = metadata.get("skills", "").lower()
            if not all(skill.lower() in resume_skills for skill in required_skills):
                continue

        if location:
            resume_location = metadata.get("location", "").lower()
            if location.lower() not in resume_location:
                continue

        filtered.append(result)

        if len(filtered) >= top_k:
            break

    return filtered


async def get_count() -> int:
    sql = text("SELECT COUNT(*) FROM candidates WHERE embedding IS NOT NULL")
    async with AsyncSessionLocal() as session:
        result = await session.execute(sql)
        return result.scalar_one()
