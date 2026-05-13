"""Folder scanner — finds all resumes and builds a tracking manifest."""

import json
import hashlib
import magic
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum

from config.settings import config


class FileType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    IMAGE = "image"
    UNSUPPORTED = "unsupported"


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    IMAGES_READY = "images_ready"
    JSON_EXTRACTED = "json_extracted"
    DUPLICATE = "duplicate"
    FAILED = "failed"


@dataclass
class FileRecord:
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


def detect_file_type(file_path: Path) -> FileType:
    """Detect actual file type via binary header; falls back to extension."""
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


def compute_file_hash(file_path: str) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def deduplicate_records(
    records: list[FileRecord],
) -> tuple[list[FileRecord], list[FileRecord]]:
    """Remove exact file-content duplicates before extraction to avoid redundant API calls."""
    seen_hashes: dict[str, str] = {}
    unique = []
    duplicates = []

    for record in records:
        file_hash = compute_file_hash(record.path)
        record.file_hash = file_hash

        if file_hash in seen_hashes:
            record.status = ProcessingStatus.DUPLICATE
            record.error_message = f"Duplicate of: {seen_hashes[file_hash]}"
            duplicates.append(record)
        else:
            seen_hashes[file_hash] = Path(record.path).name
            unique.append(record)

    if duplicates:
        print(f"  Found {len(duplicates)} exact duplicates:")
        for d in duplicates[:5]:
            name = Path(d.path).name
            print(f"    {name} → {d.error_message}")
        if len(duplicates) > 5:
            print(f"    ... and {len(duplicates) - 5} more")

    return unique, duplicates


def scan_folder(resume_dir: Path | None = None) -> list[FileRecord]:
    resume_dir = resume_dir or config.resume_dir

    if not resume_dir.exists():
        raise FileNotFoundError(f"Resume directory not found: {resume_dir}")

    records: list[FileRecord] = []
    skipped: list[str] = []

    for file_path in sorted(resume_dir.rglob("*")):
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


def save_manifest(records: list[FileRecord], path: Path | None = None):
    """Persist manifest to disk so the pipeline can resume after a crash."""
    path = path or config.manifest_path
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [r.to_dict() for r in records]
    path.write_text(json.dumps(data, indent=2))
    print(f"  Manifest saved: {path} ({len(records)} files)")


def load_manifest(path: Path | None = None) -> list[FileRecord]:
    path = path or config.manifest_path

    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    data = json.loads(path.read_text())
    return [FileRecord.from_dict(d) for d in data]


def get_manifest_summary(records: list[FileRecord]) -> dict:
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
