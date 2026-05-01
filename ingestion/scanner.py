"""
Folder scanner — finds all resumes and builds a tracking manifest.

This module answers: "What do I have to process?"

At 10K files, you can't just loop and hope for the best.
You need to know:
  - How many files and what types?
  - Which ones succeeded, which failed?
  - Where to resume if the pipeline crashes at file 6,437?

The manifest is a JSON file that tracks all of this.
Think of it as a to-do list for your pipeline.

WHY MIME detection instead of just file extensions?
  A file named "resume.pdf" might actually be:
  - A DOCX renamed to .pdf (common in email attachments)
  - A corrupted file
  - An image saved with wrong extension
  python-magic reads the file's binary header (first few bytes)
  to determine what it ACTUALLY is, not what it claims to be.

WHY dataclasses instead of Pydantic here?
  FileRecord is internal plumbing — it never becomes an API
  response or request body. Dataclasses are simpler and faster
  for internal data structures. Pydantic is reserved for things
  that cross boundaries (config, API models, resume schema).
"""

import json
import hashlib
import magic
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum

from config.settings import config


class FileType(str, Enum):
    """
    Supported file categories.

    Using str + Enum so values serialize cleanly to JSON.
    FileType.PDF becomes just "pdf" in the manifest file.
    """
    PDF = "pdf"
    DOCX = "docx"
    IMAGE = "image"
    UNSUPPORTED = "unsupported"


class ProcessingStatus(str, Enum):
    """
    Tracks where each file is in the pipeline.

    The pipeline updates this as it progresses:
      PENDING → IMAGES_READY → JSON_EXTRACTED → FAILED

    WHY track status?
    If you're processing 10K files and it crashes at file 5,000,
    you reload the manifest and skip everything that's already
    JSON_EXTRACTED. No reprocessing, no wasted API calls.
    """
    PENDING = "pending"
    IMAGES_READY = "images_ready"
    JSON_EXTRACTED = "json_extracted"
    DUPLICATE = "duplicate"
    FAILED = "failed"


@dataclass
class FileRecord:
    """
    Represents a single resume file in the manifest.

    This is a simple data container — no business logic.
    The pipeline reads and updates these records as it processes.
    """
    path: str
    file_type: str = FileType.UNSUPPORTED
    size_bytes: int = 0
    status: str = ProcessingStatus.PENDING
    error_message: str = ""
    file_hash: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FileRecord":
        return cls(**data)


# ──────────────────────────────────────────────
#  FILE TYPE DETECTION
# ──────────────────────────────────────────────

def detect_file_type(file_path: Path) -> FileType:
    """
    Detect actual file type using MIME type (binary header).
    Falls back to extension if MIME detection fails.
    """
    # Primary: check binary header
    try:
        mime = magic.from_file(str(file_path), mime=True)
    except Exception:
        mime = ""

    mime_map = {
        "application/pdf": FileType.PDF,
        "application/vnd.openxmlformats-officedocument"
        ".wordprocessingml.document": FileType.DOCX,
        "application/msword": FileType.DOCX,
        "image/jpeg": FileType.IMAGE,
        "image/png": FileType.IMAGE,
        "image/webp": FileType.IMAGE,
    }

    if mime in mime_map:
        return mime_map[mime]

    # Fallback: check extension
    ext = file_path.suffix.lower()
    ext_map = {
        ".pdf": FileType.PDF,
        ".docx": FileType.DOCX,
        ".doc": FileType.DOCX,
        ".jpg": FileType.IMAGE,
        ".jpeg": FileType.IMAGE,
        ".png": FileType.IMAGE,
        ".webp": FileType.IMAGE,
    }

    return ext_map.get(ext, FileType.UNSUPPORTED)


# ──────────────────────────────────────────────
#  FILE HASHING
# ──────────────────────────────────────────────

def compute_file_hash(file_path: str) -> str:
    """
    Compute MD5 hash of a file's contents.

    Two files with identical content produce the same hash,
    regardless of filename. This catches:
      - Resume.docx and Resume(4).docx (same file, renamed)
      - Copies saved in different folders

    WHY MD5 and not SHA-256?
    We're detecting duplicates, not securing passwords.
    MD5 is faster and hash collisions are irrelevant at
    10K files — the probability is astronomically low.

    WHY read in chunks?
    Large files (scanned PDFs can be 10MB+) shouldn't be
    loaded entirely into memory. Reading in 8KB chunks
    keeps memory usage constant regardless of file size.
    """
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ──────────────────────────────────────────────
#  DEDUPLICATION
# ──────────────────────────────────────────────

def deduplicate_records(
    records: list[FileRecord],
) -> tuple[list[FileRecord], list[FileRecord]]:
    """
    Remove duplicate files BEFORE extraction.

    Compares file content hashes — catches exact duplicates
    even when filenames differ:
      Resume.docx, Resume(3).docx, Resume(4).docx
      → keeps first one, marks others as DUPLICATE

    Args:
        records: all scanned FileRecords

    Returns:
        Tuple of (unique_records, duplicate_records)

    WHY deduplicate before extraction?
    Each extraction costs ~$0.007 in API calls (Haiku).
    For 80 duplicates in 700 files, that's $0.56 saved.
    At 10K files with 15% duplicates, it saves ~$10.
    More importantly, it keeps your vector DB clean —
    no duplicate entries in search results.
    """
    seen_hashes: dict[str, str] = {}  # hash → first file path
    unique = []
    duplicates = []

    for record in records:
        file_hash = compute_file_hash(record.path)
        record.file_hash = file_hash

        if file_hash in seen_hashes:
            record.status = ProcessingStatus.DUPLICATE
            record.error_message = (
                f"Duplicate of: {seen_hashes[file_hash]}"
            )
            duplicates.append(record)
        else:
            seen_hashes[file_hash] = Path(record.path).name
            unique.append(record)

    if duplicates:
        print(f"  Found {len(duplicates)} exact duplicates:")
        for d in duplicates[:5]:  # show first 5
            name = Path(d.path).name
            print(f"    {name} → {d.error_message}")
        if len(duplicates) > 5:
            print(f"    ... and {len(duplicates) - 5} more")

    return unique, duplicates


# ──────────────────────────────────────────────
#  FOLDER SCANNING
# ──────────────────────────────────────────────

def scan_folder(resume_dir: Path | None = None) -> list[FileRecord]:
    """
    Walk the resume directory and build a manifest.

    Returns a list of FileRecord objects.
    Unsupported files are skipped (not added to manifest).

    WHY sorted()?
    Deterministic ordering means re-running the scanner
    produces the same manifest. Makes debugging easier.
    """
    resume_dir = resume_dir or config.resume_dir

    if not resume_dir.exists():
        raise FileNotFoundError(
            f"Resume directory not found: {resume_dir}"
        )

    records: list[FileRecord] = []
    skipped: list[str] = []

    for file_path in sorted(resume_dir.rglob("*")):
        # Skip directories and hidden/system files
        if file_path.is_dir():
            continue
        if file_path.name.startswith("."):
            continue
        if file_path.name in {"Thumbs.db", "desktop.ini"}:
            continue

        file_type = detect_file_type(file_path)

        if file_type == FileType.UNSUPPORTED:
            skipped.append(str(file_path))
            continue

        record = FileRecord(
            path=str(file_path),
            file_type=file_type,
            size_bytes=file_path.stat().st_size,
            status=ProcessingStatus.PENDING,
        )
        records.append(record)

    if skipped:
        print(f"  Skipped {len(skipped)} unsupported files")

    return records


# ──────────────────────────────────────────────
#  MANIFEST PERSISTENCE
# ──────────────────────────────────────────────

def save_manifest(records: list[FileRecord], path: Path | None = None):
    """
    Save manifest to JSON file.

    WHY save to disk?
    If the pipeline crashes, you can reload the manifest,
    check each record's status, and resume from where you
    left off. Without this, a crash at file 9,999 means
    reprocessing all 10,000 files.
    """
    path = path or config.manifest_path
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [r.to_dict() for r in records]
    path.write_text(json.dumps(data, indent=2))
    print(f"  Manifest saved: {path} ({len(records)} files)")


def load_manifest(path: Path | None = None) -> list[FileRecord]:
    """Load a previously saved manifest for resumable processing."""
    path = path or config.manifest_path

    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    data = json.loads(path.read_text())
    return [FileRecord.from_dict(d) for d in data]


def get_manifest_summary(records: list[FileRecord]) -> dict:
    """
    Quick summary for display/logging.

    Returns counts by file type and processing status.
    Useful for progress tracking: "4,521 / 10,000 processed"
    """
    summary = {
        "total_files": len(records),
        "by_type": {},
        "by_status": {},
        "total_size_mb": round(
            sum(r.size_bytes for r in records) / (1024 * 1024), 2
        ),
    }

    for r in records:
        summary["by_type"][r.file_type] = (
            summary["by_type"].get(r.file_type, 0) + 1
        )
        summary["by_status"][r.status] = (
            summary["by_status"].get(r.status, 0) + 1
        )

    return summary