"""
Backfill chunk embeddings into Qdrant for candidates already in Postgres,
then run sample searches.

Usage:
    python run_embeddings.py

Run this after migrating to Qdrant or after manually inserting candidates.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import select

from config.schema import ResumeSchema
from db.session import AsyncSessionLocal
from db.models import Candidate
from embeddings.embedder import embed_chunks_batch
from embeddings.vector_store import (
    setup_collection,
    upsert_chunks,
    search_by_chunks,
    get_chunk_count,
    get_qdrant,
)


async def backfill_chunks():
    """
    Upsert chunk embeddings to Qdrant for all processed candidates.
    Idempotent — safe to re-run; deterministic point IDs prevent duplicates.
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Candidate).where(Candidate.status == "processed")
        )
        candidates = result.scalars().all()

        if not candidates:
            print("  No processed candidates found.")
            return 0

        print(f"  Upserting chunks for {len(candidates)} candidates...")

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

                await upsert_chunks(
                    candidate_id=str(candidate.id),
                    chunks=chunks,
                    embeddings=chunk_embeddings,
                    payload_meta={
                        "name":                   candidate.name,
                        "email":                  candidate.email,
                        "location":               candidate.location,
                        "skills":                 candidate.skills or [],
                        "total_years_experience": candidate.total_years_experience,
                        "highest_education":      candidate.highest_education,
                        "status":                 "processed",
                    },
                )
                updated += 1
            except Exception as e:
                print(f"  Failed to upsert chunks for {candidate.id}: {e}")

    return updated


async def run_sample_searches():
    queries = [
        "Python backend developer with API experience",
        "Data scientist with machine learning and NLP",
        "Ruby on Rails developer",
    ]

    for query in queries:
        print(f'  Query: "{query}"')
        results = await search_by_chunks(query, top_k=3)

        if results:
            for rank, result in enumerate(results, 1):
                name  = result["metadata"].get("name", "Unknown")
                score = result["score"]
                years = result["metadata"].get("total_years_experience", 0)
                loc   = result["metadata"].get("location", "")
                print(f"    #{rank} {name} | score: {score} | {years}y | {loc}")
        else:
            print("    No results found")
        print()


async def main():
    print("=" * 50)
    print("  Qdrant Backfill + Search Test")
    print("=" * 50)
    print()

    print("[Step 1] Ensuring Qdrant collection exists...")
    await setup_collection()
    print("  Collection ready.")

    print()
    print("[Step 2] Backfilling chunk embeddings...")
    updated = await backfill_chunks()
    if updated:
        print(f"  Upserted chunks for {updated} candidates.")
    else:
        print("  Nothing to backfill.")

    count = await get_chunk_count()
    print(f"  Total chunk vectors in Qdrant: {count}")

    if count == 0:
        print("\n  No vectors found — ingest some resumes first.")
        await get_qdrant().close()
        return

    print()
    print("[Step 3] Sample searches...")
    print()
    await run_sample_searches()

    await get_qdrant().close()

    print("=" * 50)
    print("  Done.")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
