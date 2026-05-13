"""
Deduplication using extracted contact fields — no API calls, no embeddings.

Three passes in order of confidence:
  1. Same email     → definite duplicate
  2. Same phone     → definite duplicate
  3. Same name + location → likely duplicate

When duplicates are found, the most complete resume is kept.
"""

from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path

from config.schema import ResumeSchema


@dataclass
class DedupResult:
    unique: list[tuple[str, ResumeSchema]]
    duplicates: list[tuple[str, str, str]]    # (dup_id, kept_id, reason)


def deduplicate_resumes(
    resumes: list[tuple[str, ResumeSchema]],
) -> DedupResult:
    scored = {
        rid: _completeness_score(resume)
        for rid, resume in resumes
    }

    duplicate_ids: dict[str, tuple[str, str]] = {}
    resume_map = {rid: resume for rid, resume in resumes}

    print("  Pass 1: Checking emails...")
    email_dupes = _dedup_by_field(
        resumes, scored, duplicate_ids,
        field_extractor=lambda r: r.contact.email.lower().strip(),
        field_name="email",
    )

    print("  Pass 2: Checking phone numbers...")
    phone_dupes = _dedup_by_field(
        resumes, scored, duplicate_ids,
        field_extractor=lambda r: _normalize_phone(r.contact.phone),
        field_name="phone",
    )

    print("  Pass 3: Checking name + location...")
    name_dupes = _dedup_by_field(
        resumes, scored, duplicate_ids,
        field_extractor=lambda r: _name_location_key(r),
        field_name="name+location",
    )

    duplicates = [
        (dup_id, kept_id, reason)
        for dup_id, (kept_id, reason) in duplicate_ids.items()
    ]

    unique = [
        (rid, resume)
        for rid, resume in resumes
        if rid not in duplicate_ids
    ]

    print(f"\n  Dedup summary:")
    print(f"    Email duplicates:         {email_dupes}")
    print(f"    Phone duplicates:         {phone_dupes}")
    print(f"    Name+location duplicates: {name_dupes}")
    print(f"    Total duplicates:         {len(duplicates)}")
    print(f"    Unique resumes:           {len(unique)}")

    return DedupResult(unique=unique, duplicates=duplicates)


def _dedup_by_field(
    resumes: list[tuple[str, ResumeSchema]],
    scores: dict[str, int],
    already_duped: dict[str, tuple[str, str]],
    field_extractor,
    field_name: str,
) -> int:
    groups = defaultdict(list)

    for rid, resume in resumes:
        if rid in already_duped:
            continue

        key = field_extractor(resume)
        if key:
            groups[key].append(rid)

    new_dupes = 0

    for key, group_ids in groups.items():
        if len(group_ids) <= 1:
            continue

        group_ids.sort(key=lambda rid: scores[rid], reverse=True)

        best_id = group_ids[0]
        for dup_id in group_ids[1:]:
            if dup_id not in already_duped:
                already_duped[dup_id] = (best_id, f"Same {field_name}: {key}")
                new_dupes += 1
                print(f"    {dup_id} → duplicate of {best_id} "
                      f"(same {field_name})")

    return new_dupes


def _normalize_phone(phone: str) -> str:
    """Normalize to last 10 digits to handle country code and formatting variations."""
    if not phone:
        return ""

    digits = "".join(c for c in phone if c.isdigit())

    if len(digits) >= 10:
        return digits[-10:]

    return digits


def _normalize_name(name: str) -> str:
    if not name:
        return ""

    name = name.lower().strip()

    for prefix in ("mr.", "ms.", "mrs.", "dr.", "mr ", "ms ", "mrs ", "dr "):
        if name.startswith(prefix):
            name = name[len(prefix):].strip()

    return " ".join(name.split())


def _name_location_key(resume: ResumeSchema) -> str:
    """
    Name alone is too ambiguous; pairing with location makes it a strong signal.
    Returns empty string if either field is missing to avoid grouping unknowns.
    """
    name = _normalize_name(resume.contact.name)
    location = resume.contact.location.lower().strip()

    if not name or not location:
        return ""

    return f"{name}|{location}"


def _completeness_score(resume: ResumeSchema) -> int:
    score = 0
    score += len(resume.experience) * 10
    score += len(resume.skills) * 2
    score += len(resume.education) * 5
    score += len(resume.to_embedding_text()) // 100
    if resume.contact.email:
        score += 10
    if resume.contact.phone:
        score += 5
    if resume.summary:
        score += 10
    return score
