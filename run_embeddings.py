"""
Backfill embeddings for candidates already in Postgres, then run sample searches.

Usage:
    python run_embeddings.py

Useful after manually inserting candidates that don't have embeddings yet.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import select, text

from config.schema import ResumeSchema
from db.session import AsyncSessionLocal
from db.models import Candidate, CandidateChunk
from embeddings.embedder import embed_resume, embed_chunks_batch
from embeddings.vector_store import search, search_with_skills, get_count, search_by_chunks, get_chunk_count


async def backfill_embeddings():
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Candidate).where(
                Candidate.status == "processed",
                Candidate.embedding == None,  # noqa: E711
            )
        )
        candidates = result.scalars().all()

    if not candidates:
        print("  All processed candidates already have embeddings.")
        return 0

    print(f"  Backfilling {len(candidates)} candidates...")

    updated = 0
    for candidate in candidates:
        if not candidate.raw_data:
            continue
        try:
            resume = ResumeSchema(**candidate.raw_data)
            embedding = embed_resume(resume)
            embedding_text = resume.to_embedding_text()

            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(Candidate).where(Candidate.id == candidate.id)
                )
                c = result.scalar_one()
                c.embedding = embedding
                c.embedding_text = embedding_text
                await session.commit()

            updated += 1
        except Exception as e:
            print(f"  Failed to backfill {candidate.id}: {e}")

    return updated


async def backfill_chunks():
    """Create chunk embeddings for processed candidates that have none yet."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Candidate).where(
                Candidate.status == "processed",
                ~Candidate.id.in_(
                    select(CandidateChunk.candidate_id).distinct()
                ),
            )
        )
        candidates = result.scalars().all()

    if not candidates:
        print("  All processed candidates already have chunks.")
        return 0

    print(f"  Creating chunks for {len(candidates)} candidates...")

    updated = 0
    for candidate in candidates:
        if not candidate.raw_data:
            continue
        try:
            resume = ResumeSchema(**candidate.raw_data)
            chunks = resume.to_chunks()
            if not chunks:
                continue

            chunk_embeddings = embed_chunks_batch(chunks)

            async with AsyncSessionLocal() as session:
                for chunk, emb in zip(chunks, chunk_embeddings):
                    session.add(CandidateChunk(
                        candidate_id=candidate.id,
                        chunk_type=chunk.chunk_type,
                        chunk_index=chunk.chunk_index,
                        chunk_text=chunk.text,
                        embedding=emb,
                    ))
                await session.commit()

            updated += 1
        except Exception as e:
            print(f"  Failed to create chunks for {candidate.id}: {e}")

    return updated


async def run_sample_searches():
    queries = [
        "Python backend developer with API experience",
        "Data scientist with machine learning and NLP",
        "Sales manager with B2B experience",
    ]

    for query in queries:
        print(f'  Query: "{query}"')
        results = await search_by_chunks(query, top_k=3)

        if results:
            for rank, result in enumerate(results, 1):
                name = result["metadata"].get("name", "Unknown")
                score = result["score"]
                years = result["metadata"].get("total_years_experience", 0)
                location = result["metadata"].get("location", "")
                print(f"    #{rank} {name} | score: {score} | {years}y | {location}")
        else:
            print("    No results found")
        print()


async def main():
    print("=" * 50)
    print("  pgvector Backfill + Search Test")
    print("=" * 50)
    print()

    print("[Step 1] Backfilling missing whole-resume embeddings...")
    updated = await backfill_embeddings()
    if updated:
        print(f"  Backfilled {updated} candidates.")

    count = await get_count()
    print(f"  Total candidates with embeddings: {count}")

    print()
    print("[Step 2] Backfilling missing chunk embeddings...")
    chunk_updated = await backfill_chunks()
    if chunk_updated:
        print(f"  Created chunks for {chunk_updated} candidates.")

    chunk_count = await get_chunk_count()
    print(f"  Total candidates with chunks: {chunk_count}")

    if count == 0 and chunk_count == 0:
        print("\n  No embeddings found — ingest some resumes first.")
        return

    print()
    print("[Step 3] Sample searches (chunk-based)...")
    print()
    await run_sample_searches()

    print("=" * 50)
    print("  Done.")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
