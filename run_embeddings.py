"""
Phase 3 Runner — Dedup + Embeddings + Vector Store + Search

Run this after the extraction pipeline has produced JSON files:
    python run_embeddings.py

What it does:
    1. Loads all JSON files from data/output/json/
    2. Deduplicates using contact info (free, instant)
    3. Generates embeddings for unique resumes only
    4. Stores embeddings + metadata in ChromaDB
    5. Runs sample searches so you can see RAG in action
    6. Debug tests for filter functionality

Prerequisites:
    - JSON files in data/output/json/ (from run_pipeline.py)
    - pip install sentence-transformers chromadb
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import config
from deduplication.dedup import deduplicate_resumes
from embeddings.embedder import load_all_resumes, embed_text
from embeddings.vector_store import (
    add_resumes_batch,
    search,
    search_with_skills,
    get_collection,
    get_count,
    delete_all,
)


def main():
    print("=" * 50)
    print("  Phase 3: Dedup + Embeddings + Vector Store")
    print("=" * 50)
    print()

    # ── Step 1: Load extracted resumes ──
    print("[Step 1] Loading extracted resumes...")
    resumes = load_all_resumes()

    if not resumes:
        print("  No resumes found! Run the extraction pipeline first:")
        print("    python run_pipeline.py")
        return

    print(f"  Loaded {len(resumes)} resumes")

    for r in resumes[:5]:
        name = r.contact.name or "Unknown"
        skills_count = len(r.skills)
        years = r.total_years_experience
        print(f"    - {name} | {skills_count} skills | {years} years")
    if len(resumes) > 5:
        print(f"    ... and {len(resumes) - 5} more")

    # ── Step 2: Deduplicate ──
    print()
    print("[Step 2] Deduplicating by contact info...")

    resume_pairs = [
        (Path(r.source_file).stem if r.source_file else f"resume_{i}", r)
        for i, r in enumerate(resumes)
    ]

    dedup_result = deduplicate_resumes(resume_pairs)

    if dedup_result.duplicates:
        print(f"\n  Duplicates removed:")
        for dup_id, kept_id, reason in dedup_result.duplicates[:10]:
            print(f"    {dup_id} → {kept_id} ({reason})")
        if len(dedup_result.duplicates) > 10:
            print(f"    ... and {len(dedup_result.duplicates) - 10} more")

    # Use only unique resumes going forward
    unique_resumes = [r for _, r in dedup_result.unique]
    unique_ids = [rid for rid, _ in dedup_result.unique]

    # ── Step 3: Embed and store ──
    print()
    print(f"[Step 3] Embedding {len(unique_resumes)} unique resumes...")

    try:
        delete_all()
    except Exception:
        pass

    added = add_resumes_batch(unique_resumes, unique_ids)
    print(f"  Stored {added} resumes in vector DB")
    print(f"  Total in collection: {get_count()}")

    # ── Step 4: Sample searches ──
    print()
    print("[Step 4] Sample searches...")
    print()

    queries = [
        "Looking for a business development manager with solar industry experience",
        "Need a Python backend developer with API development skills",
        "Data scientist with machine learning and NLP experience",
        "Site engineer for solar rooftop project installation",
        "Sales manager with B2B experience in manufacturing",
    ]

    for query in queries:
        print(f'  Query: "{query}"')
        results = search(query, top_k=3)

        if results:
            for rank, result in enumerate(results, 1):
                name = result["metadata"].get("name", "Unknown")
                score = result["score"]
                years = result["metadata"].get("total_years_experience", 0)
                location = result["metadata"].get("location", "")
                print(f"    #{rank} {name} | score: {score} | "
                      f"years: {years} | {location}")
        else:
            print("    No results found")
        print()

    # ── Step 5: Filtered search tests ──
    print("[Step 5] Testing filtered search...")
    print()

    # Test skills only
    print("  Test: Skills filter only (Python)")
    results = search_with_skills(
        query_text="Python developer",
        required_skills=["Python"],
        top_k=5,
    )
    print(f"  Results: {len(results)}")
    for r in results[:3]:
        print(f"    {r['metadata'].get('name', '?')} | {r['score']}")
    print()

    # Test location only
    print("  Test: Location filter only (Bangalore)")
    results = search_with_skills(
        query_text="software developer",
        location="Bangalore",
        top_k=5,
    )
    print(f"  Results: {len(results)}")
    for r in results[:3]:
        print(f"    {r['metadata'].get('name', '?')} | "
              f"{r['metadata'].get('location', '?')}")
    print()

    # Test combined
    print("  Test: Skills + Location (Python + Hyderabad)")
    results = search_with_skills(
        query_text="Python developer with backend experience",
        required_skills=["Python"],
        location="Hyderabad",
        top_k=5,
    )
    print(f"  Results: {len(results)}")
    for r in results[:3]:
        print(f"    {r['metadata'].get('name', '?')} | "
              f"{r['metadata'].get('location', '?')}")

    # ── Step 6: Debug — check what's in ChromaDB ──
    print()
    print("[Step 6] Debug: checking stored metadata...")

    collection = get_collection()

    # Peek at stored data
    sample = collection.peek(limit=2)
    for i in range(len(sample["ids"])):
        print(f"\n  ID: {sample['ids'][i]}")
        print(f"  Skills: {sample['metadatas'][i].get('skills', 'MISSING')[:80]}...")
        print(f"  Location: {sample['metadatas'][i].get('location', 'MISSING')}")

    # Test raw ChromaDB filters
    print("\n  Raw filter test: exact match on location")
    try:
        results = collection.get(
            where={"location": "Bangalore, India"},
            limit=3,
        )
        print(f"    Exact match results: {len(results['ids'])}")
    except Exception as e:
        print(f"    Error: {e}")

    print("\n  Raw filter test: $contains on location")
    try:
        results = collection.get(
            where={"location": {"$contains": "Bangalore"}},
            limit=3,
        )
        print(f"    $contains results: {len(results['ids'])}")
    except Exception as e:
        print(f"    Error: {e}")

    print("\n  Raw filter test: $contains on skills")
    try:
        results = collection.get(
            where={"skills": {"$contains": "Python"}},
            limit=3,
        )
        print(f"    $contains results: {len(results['ids'])}")
    except Exception as e:
        print(f"    Error: {e}")

    # ── Done ──
    print()
    print("=" * 50)
    print("  Phase 3 complete!")
    print("=" * 50)
    print()
    print(f"  Resumes loaded:     {len(resumes)}")
    print(f"  Duplicates found:   {len(dedup_result.duplicates)}")
    print(f"  Unique embedded:    {len(unique_resumes)}")
    print(f"  Stored in ChromaDB: {get_count()}")
    print()
    print("  Next steps:")
    print("    - Phase 6: Build recommendation engine")
    print("    - Phase 7: Add FastAPI endpoints")


if __name__ == "__main__":
    main()