"""
Pipeline orchestrator — runs the full ingestion flow.

Connects all modules in sequence:
  scan → DEDUP → convert → extract → validate → save

WHY a separate orchestrator?
  Each module (scanner, converter, extractor, validator) does
  one thing and knows nothing about the others. The pipeline
  is the only place that knows the full flow. This means:

  - You can test each module independently
  - You can swap modules (different LLM, different vector DB)
    without touching the others
  - FastAPI just calls pipeline functions — it doesn't need
    to know about converters or validators
  - You can run the pipeline from CLI, from an API endpoint,
    or from a Jupyter notebook — same code

WHY process_single + process_batch + process_batch_concurrent?
  process_single: handles one resume end-to-end. This is what
    FastAPI calls when a user uploads a single resume.
  process_batch: handles many resumes sequentially with progress
    tracking. Safe and simple.
  process_batch_concurrent: handles many resumes in parallel.
    5x faster but uses more resources.
"""

import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.settings import config
from config.schema import ResumeSchema
from ingestion.scanner import (
    scan_folder,
    save_manifest,
    load_manifest,
    get_manifest_summary,
    deduplicate_records,
    FileRecord,
    ProcessingStatus,
)
from extraction.llm_extractor import extract_resume_with_retry
from extraction.validator import validate_resume


# ──────────────────────────────────────────────
#  SINGLE RESUME PROCESSING
# ──────────────────────────────────────────────

def process_single(file_path: str, file_type: str) -> tuple[ResumeSchema | None, str]:
    """
    Process one resume: convert → extract → validate.

    Args:
        file_path: path to the resume file
        file_type: detected file type ("pdf", "docx", "image")

    Returns:
        Tuple of (ResumeSchema or None, error_message)

    This is the core function. Everything else calls this.
    FastAPI endpoint will call this directly:

        @app.post("/extract")
        async def extract(file: UploadFile):
            # save file, detect type...
            resume, error = process_single(path, file_type)
            if error:
                raise HTTPException(400, error)
            return resume
    """
    filename = Path(file_path).name

    # Step 1: Extract JSON via LLM (conversion happens lazily inside on fallback)
    raw_data = extract_resume_with_retry(file_path, file_type)

    # Step 2: Validate and clean
    resume, error = validate_resume(raw_data, source_file=filename)

    return resume, error


# ──────────────────────────────────────────────
#  SCAN + DEDUP (shared by both batch modes)
# ──────────────────────────────────────────────

def _get_records(
    resume_dir: Path | None = None,
    resume_from_manifest: bool = False,
) -> tuple[list[FileRecord], list[FileRecord]]:
    """
    Scan folder, deduplicate, and return records.

    Returns:
        Tuple of (all_records, pending_records)
        all_records includes duplicates (for manifest tracking)
        pending_records is what needs processing

    WHY extract this into a helper?
    Both process_batch and process_batch_concurrent need
    the same scan + dedup logic. DRY — don't repeat yourself.
    """
    if resume_from_manifest:
        try:
            all_records = load_manifest()
            print(f"  Loaded manifest: {len(all_records)} files")

            # Split into pending vs already done
            pending = [
                r for r in all_records
                if r.status not in (
                    ProcessingStatus.JSON_EXTRACTED,
                    ProcessingStatus.DUPLICATE,
                )
            ]
            return all_records, pending

        except FileNotFoundError:
            print("  No manifest found, scanning folder...")

    # Fresh scan
    records = scan_folder(resume_dir)

    # Deduplicate before extraction (saves API costs)
    unique, duplicates = deduplicate_records(records)

    # Combine for manifest (track everything)
    all_records = unique + duplicates
    save_manifest(all_records)

    print(f"  Total files: {len(all_records)}")
    print(f"  Unique to process: {len(unique)}")
    print(f"  Duplicates skipped: {len(duplicates)}")

    return all_records, unique


# ──────────────────────────────────────────────
#  SEQUENTIAL BATCH PROCESSING
# ──────────────────────────────────────────────

def process_batch(
    resume_dir: Path | None = None,
    resume_from_manifest: bool = False,
) -> dict:
    """
    Process all resumes sequentially (one at a time).

    Safe, simple, easy to debug. Use this for small batches
    or when you want to watch the output carefully.
    """
    config.ensure_dirs()

    all_records, pending = _get_records(resume_dir, resume_from_manifest)

    results = {
        "total": len(all_records),
        "succeeded": 0,
        "failed": 0,
        "skipped": len(all_records) - len(pending),
        "duplicates": sum(
            1 for r in all_records
            if r.status == ProcessingStatus.DUPLICATE
        ),
        "errors": [],
        "duration_seconds": 0,
    }

    start_time = time.time()

    for i, record in enumerate(pending):
        filename = Path(record.path).name
        print(f"  [{i+1}/{len(pending)}] Processing: {filename}")

        resume, error = process_single(record.path, record.file_type)

        if error:
            record.status = ProcessingStatus.FAILED
            record.error_message = error
            results["failed"] += 1
            results["errors"].append({"file": filename, "error": error})
            print(f"    FAILED: {error}")
        else:
            _save_json(resume, filename)
            record.status = ProcessingStatus.JSON_EXTRACTED
            results["succeeded"] += 1
            print(f"    OK: {resume.contact.name or 'name not found'}")

        # Save manifest after each file (crash recovery)
        save_manifest(all_records)

    results["duration_seconds"] = round(time.time() - start_time, 2)
    _print_summary(results)
    return results


# ──────────────────────────────────────────────
#  CONCURRENT BATCH PROCESSING
# ──────────────────────────────────────────────

def process_batch_concurrent(
    resume_dir: Path | None = None,
    resume_from_manifest: bool = False,
    max_workers: int = 5,
) -> dict:
    """
    Process resumes with concurrent API calls.

    Args:
        max_workers: number of parallel API calls.
            5 is safe for most Anthropic API rate limits.
            Increase to 10 if you have higher tier access.
            Decrease to 3 if you see 429 rate limit errors.

    WHY ThreadPoolExecutor instead of asyncio?
    The Anthropic SDK's client.messages.create() is synchronous.
    ThreadPoolExecutor runs multiple synchronous calls in parallel.
    For I/O-bound tasks (waiting for API responses), threads
    work just as well as async and are simpler to understand.

    WHY limit to 5 workers?
    Anthropic rate-limits API calls per minute. Sending 50
    concurrent requests would get 429 errors. 5 workers keeps
    you safely under the limit while still being ~5x faster.
    """
    config.ensure_dirs()

    all_records, pending = _get_records(resume_dir, resume_from_manifest)

    results = {
        "total": len(all_records),
        "succeeded": 0,
        "failed": 0,
        "skipped": len(all_records) - len(pending),
        "duplicates": sum(
            1 for r in all_records
            if r.status == ProcessingStatus.DUPLICATE
        ),
        "errors": [],
        "duration_seconds": 0,
    }

    if not pending:
        print("  Nothing to process!")
        return results

    print(f"  Processing {len(pending)} files with {max_workers} workers...")

    start_time = time.time()
    completed = 0

    def _process_one(record: FileRecord) -> dict:
        """Process a single resume in a thread."""
        filename = Path(record.path).name
        resume, error = process_single(record.path, record.file_type)

        if error:
            record.status = ProcessingStatus.FAILED
            record.error_message = error
            return {"file": filename, "success": False, "error": error}
        else:
            _save_json(resume, filename)
            record.status = ProcessingStatus.JSON_EXTRACTED
            return {
                "file": filename,
                "success": True,
                "name": resume.contact.name or "name not found",
            }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_record = {
            executor.submit(_process_one, record): record
            for record in pending
        }

        # Collect results as they complete
        for future in as_completed(future_to_record):
            completed += 1
            result = future.result()
            filename = result["file"]

            if result["success"]:
                results["succeeded"] += 1
                print(f"  [{completed}/{len(pending)}] OK: {filename}"
                      f" → {result['name']}")
            else:
                results["failed"] += 1
                results["errors"].append(result)
                print(f"  [{completed}/{len(pending)}] FAILED: {filename}"
                      f" → {result['error'][:80]}")

        # Save manifest once at the end
        # (not per-file like sequential, because threads would conflict)
        save_manifest(all_records)

    results["duration_seconds"] = round(time.time() - start_time, 2)
    _print_summary(results)
    return results


# ──────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────

def _print_summary(results: dict):
    """Print a formatted summary of processing results."""
    print(f"\n  {'=' * 40}")
    print(f"  Done in {results['duration_seconds']}s")
    print(f"  Succeeded:  {results['succeeded']}")
    print(f"  Failed:     {results['failed']}")
    print(f"  Duplicates: {results['duplicates']}")
    print(f"  Skipped:    {results['skipped']}")
    print(f"  {'=' * 40}")


def _save_json(resume: ResumeSchema, filename: str):
    """
    Save validated resume JSON to the output directory.

    File naming: original_name.pdf → original_name.json
    This makes it easy to trace back from JSON to source file.
    """
    json_filename = Path(filename).stem + ".json"
    output_path = config.json_dir / json_filename

    data = resume.model_dump()
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def load_json(filename: str) -> ResumeSchema | None:
    """
    Load a previously extracted resume JSON.

    Useful for:
      - Debugging: inspect what was extracted
      - Re-validation: run updated validator on old data
      - Embedding: load JSON → to_embedding_text() → embed
      - FastAPI: serve resume data without re-extracting

    Returns None if file doesn't exist.
    """
    json_path = config.json_dir / filename
    if not json_path.exists():
        return None

    data = json.loads(json_path.read_text())
    return ResumeSchema(**data)