"""
Recommendation engine runner.

Usage:
    python run_recommend.py

Tests both fast_match (free, instant) and smart_match
(LLM-powered, ~$0.01 per search) with sample job descriptions.

Prerequisites:
    - Embeddings stored in pgvector (run run_embeddings.py first)
    - ANTHROPIC_API_KEY set (for smart_match only)
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from recommendation.ranker import fast_match, smart_match


async def main():
    print("=" * 50)
    print("  Recommendation Engine")
    print("=" * 50)

    # ── Test 1: Fast match (free) ──
    print()
    print("[Test 1] Fast match — vector search + filters")
    print()

    results = await fast_match(
        query="Python backend developer with cloud experience",
        required_skills=["Python"],
        min_years=2,
        location="Hyderabad",
        top_k=50,
    )

    print(f"  Query: Python backend developer, 2+ years")
    print(f"  Results: {len(results)}")
    for c in results:
        print(f"    #{c.rank} {c.name}")
        print(f"       Score: {c.similarity_score} | "
              f"Years: {c.years_experience} | {c.location}")
        print(f"       Skills: {', '.join(c.skills[:5])}")
    print()

    # ── Test 2: Smart match (LLM re-ranked) ──
    print("[Test 2] Smart match — with LLM re-ranking")
    print()

    sample_jd = """
    We are looking for a Senior Software Engineer to join our backend team.

    Requirements:
    - 3+ years of experience in Python and Java
    - Experience building REST APIs and microservices
    - Familiarity with cloud platforms (AWS/GCP/Azure)
    - Experience with SQL and NoSQL databases
    - Good understanding of system design and architecture

    Nice to have:
    - Experience with Docker and Kubernetes
    - CI/CD pipeline experience
    - Experience in an agile team environment

    Location: Bangalore or Hyderabad (hybrid)
    """

    results = await smart_match(job_description=sample_jd, top_k=15)

    print(f"\n  Top {len(results)} candidates:")
    print()
    for c in results:
        print(f"    #{c.rank} {c.name}")
        print(f"       Score: {c.similarity_score} | "
              f"Years: {c.years_experience} | {c.location}")
        print(f"       Skills: {', '.join(c.skills[:5])}")
        if c.match_explanation:
            print(f"       Why: {c.match_explanation}")
        print()

    print("=" * 50)
    print("  Done! Your recommendation engine is working.")
    print("=" * 50)
    print()
    print("  What you just saw:")
    print("    fast_match  → instant, free, uses vector search")
    print("    smart_match → 2-3 seconds, ~$0.01, LLM re-ranks results")
    print()
    print("  Next step: wrap this in FastAPI endpoints")


if __name__ == "__main__":
    asyncio.run(main())