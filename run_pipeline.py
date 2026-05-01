"""
CLI entry point for the resume matcher pipeline.

Usage:
    # Process all resumes (sequential, with dedup)
    python run_pipeline.py

    # Process with concurrent API calls (5x faster)
    python run_pipeline.py --concurrent

    # Resume from where you left off after a crash
    python run_pipeline.py --resume

    # Resume + concurrent (the combo you'll use most)
    python run_pipeline.py --resume --concurrent

WHY a separate runner file?
  pipeline.py contains the logic (importable by FastAPI).
  run_pipeline.py contains the CLI interface (prints, formatting).

  This separation means:
  - FastAPI imports pipeline.py directly, never touches this file
  - You can change CLI output without touching business logic
  - Testing pipeline.py doesn't require simulating CLI args
"""

import sys
from pathlib import Path

# Add project root to Python path so imports work
# when running from project directory: python run_pipeline.py
sys.path.insert(0, str(Path(__file__).parent))

from ingestion.pipeline import process_batch, process_batch_concurrent


def main():
    print("=" * 50)
    print("  Resume Matcher — Ingestion Pipeline")
    print("=" * 50)
    print()

    # Parse flags
    resume_from_manifest = "--resume" in sys.argv
    concurrent = "--concurrent" in sys.argv

    # Show mode
    mode_parts = []
    if resume_from_manifest:
        mode_parts.append("RESUME (skipping already-processed)")
    else:
        mode_parts.append("FRESH (processing all files)")

    if concurrent:
        mode_parts.append("CONCURRENT (5 parallel workers)")
    else:
        mode_parts.append("SEQUENTIAL")

    print(f"  Mode: {' + '.join(mode_parts)}")
    print()

    # Run the pipeline
    if concurrent:
        results = process_batch_concurrent(
            resume_from_manifest=resume_from_manifest,
            max_workers=5,
        )
    else:
        results = process_batch(
            resume_from_manifest=resume_from_manifest,
        )

    # Print final report
    print()
    print("=" * 50)
    print("  FINAL REPORT")
    print("=" * 50)
    print(f"  Total files:  {results['total']}")
    print(f"  Succeeded:    {results['succeeded']}")
    print(f"  Failed:       {results['failed']}")
    print(f"  Duplicates:   {results['duplicates']}")
    print(f"  Skipped:      {results['skipped']}")
    print(f"  Duration:     {results['duration_seconds']}s")

    if results["errors"]:
        print()
        print("  ERRORS:")
        for err in results["errors"][:10]:  # show first 10
            print(f"    {err['file']}: {err['error'][:80]}")
        if len(results["errors"]) > 10:
            print(f"    ... and {len(results['errors']) - 10} more")

    print()

    # Suggest next step
    if results["succeeded"] > 0:
        print("  JSON files saved to: data/output/json/")
        print()
        print("  Next steps:")
        print("    1. Review the extracted JSON files")
        print("    2. Run: python run_embeddings.py")
        print("    3. Search your resumes with natural language!")


if __name__ == "__main__":
    main()