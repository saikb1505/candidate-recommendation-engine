"""Pipeline orchestrator — scan → dedup → extract → validate → save."""

import asyncio
import json
import time
from pathlib import Path

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


async def process_single(file_path: str, file_type: str) -> tuple[ResumeSchema | None, str]:
    filename = Path(file_path).name
    raw_data = await extract_resume_with_retry(file_path, file_type)
    resume, error = validate_resume(raw_data, source_file=filename)
    return resume, error


def _get_records(
    resume_dir: Path | None = None,
    resume_from_manifest: bool = False,
) -> tuple[list[FileRecord], list[FileRecord]]:
    if resume_from_manifest:
        try:
            all_records = load_manifest()
            print(f"  Loaded manifest: {len(all_records)} files")

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

    records = scan_folder(resume_dir)
    unique, duplicates = deduplicate_records(records)
    all_records = unique + duplicates
    save_manifest(all_records)

    print(f"  Total files: {len(all_records)}")
    print(f"  Unique to process: {len(unique)}")
    print(f"  Duplicates skipped: {len(duplicates)}")

    return all_records, unique


async def process_batch(
    resume_dir: Path | None = None,
    resume_from_manifest: bool = False,
) -> dict:
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

        resume, error = await process_single(record.path, record.file_type)

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

        save_manifest(all_records)

    results["duration_seconds"] = round(time.time() - start_time, 2)
    _print_summary(results)
    return results


async def process_batch_concurrent(
    resume_dir: Path | None = None,
    resume_from_manifest: bool = False,
    max_workers: int = 5,
) -> dict:
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

    print(f"  Processing {len(pending)} files with {max_workers} concurrent workers...")

    start_time = time.time()
    semaphore = asyncio.Semaphore(max_workers)

    async def _process_one(record: FileRecord) -> dict:
        async with semaphore:
            filename = Path(record.path).name
            resume, error = await process_single(record.path, record.file_type)

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

    task_results = await asyncio.gather(
        *[_process_one(record) for record in pending],
        return_exceptions=True,
    )

    for i, result in enumerate(task_results):
        if isinstance(result, Exception):
            filename = Path(pending[i].path).name
            results["failed"] += 1
            results["errors"].append({"file": filename, "error": str(result)})
            print(f"  [{i+1}/{len(pending)}] FAILED: {filename} → {result}")
        elif result["success"]:
            results["succeeded"] += 1
            print(f"  [{i+1}/{len(pending)}] OK: {result['file']} → {result['name']}")
        else:
            results["failed"] += 1
            results["errors"].append(result)
            print(f"  [{i+1}/{len(pending)}] FAILED: {result['file']} → {result['error'][:80]}")

    save_manifest(all_records)

    results["duration_seconds"] = round(time.time() - start_time, 2)
    _print_summary(results)
    return results


def _print_summary(results: dict):
    print(f"\n  {'=' * 40}")
    print(f"  Done in {results['duration_seconds']}s")
    print(f"  Succeeded:  {results['succeeded']}")
    print(f"  Failed:     {results['failed']}")
    print(f"  Duplicates: {results['duplicates']}")
    print(f"  Skipped:    {results['skipped']}")
    print(f"  {'=' * 40}")


def _save_json(resume: ResumeSchema, filename: str):
    json_filename = Path(filename).stem + ".json"
    output_path = config.json_dir / json_filename
    data = resume.model_dump()
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def load_json(filename: str) -> ResumeSchema | None:
    json_path = config.json_dir / filename
    if not json_path.exists():
        return None

    data = json.loads(json_path.read_text())
    return ResumeSchema(**data)
