"""
Deduplication using structured JSON fields.

Catches duplicate resumes by comparing extracted contact info
BEFORE any embeddings are generated. This means:
  - Zero cost (no API calls)
  - Instant (string comparison, not vector math)
  - Works regardless of embedding provider

Strategy:
  1. Exact match: same email → same person (strongest signal)
  2. Exact match: same phone → same person
  3. Fuzzy match: same normalized name + same location → likely same person

When duplicates are found, keep the most complete resume
(most skills, most experience entries, longest content).

WHY NOT use embeddings for dedup?
  If you're using a paid embedding API (OpenAI, Cohere),
  embedding 500 resumes just to find duplicates costs money.
  And you'd STILL need to embed the unique ones afterward.
  That's paying twice. Structured field comparison is free
  and catches 95%+ of duplicates.
"""

from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path

from config.schema import ResumeSchema


@dataclass
class DedupResult:
    """
    Output of the deduplication process.

    unique: resumes to keep (embed and store these)
    duplicates: list of (duplicate_id, kept_id, reason)
    """
    unique: list[tuple[str, ResumeSchema]]
    duplicates: list[tuple[str, str, str]]    # (dup_id, kept_id, reason)


def deduplicate_resumes(
    resumes: list[tuple[str, ResumeSchema]],
) -> DedupResult:
    """
    Find and remove duplicate resumes using contact info.

    Args:
        resumes: list of (resume_id, ResumeSchema) tuples

    Returns:
        DedupResult with unique resumes and duplicate info.

    Three passes, in order of confidence:
      1. Same email → definite duplicate
      2. Same phone → definite duplicate
      3. Same name + same location → likely duplicate
    """
    # Score all resumes by completeness upfront
    scored = {
        rid: _completeness_score(resume)
        for rid, resume in resumes
    }

    # Track which IDs have been marked as duplicates
    duplicate_ids: dict[str, tuple[str, str]] = {}  # dup_id → (kept_id, reason)

    # Build lookup for quick access
    resume_map = {rid: resume for rid, resume in resumes}

    # ── Pass 1: Email dedup ──
    print("  Pass 1: Checking emails...")
    email_dupes = _dedup_by_field(
        resumes, scored, duplicate_ids,
        field_extractor=lambda r: r.contact.email.lower().strip(),
        field_name="email",
    )

    # ── Pass 2: Phone dedup ──
    print("  Pass 2: Checking phone numbers...")
    phone_dupes = _dedup_by_field(
        resumes, scored, duplicate_ids,
        field_extractor=lambda r: _normalize_phone(r.contact.phone),
        field_name="phone",
    )

    # ── Pass 3: Name + Location dedup ──
    print("  Pass 3: Checking name + location...")
    name_dupes = _dedup_by_field(
        resumes, scored, duplicate_ids,
        field_extractor=lambda r: _name_location_key(r),
        field_name="name+location",
    )

    # Build results
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


# ──────────────────────────────────────────────
#  GENERIC FIELD-BASED DEDUP
# ──────────────────────────────────────────────

def _dedup_by_field(
    resumes: list[tuple[str, ResumeSchema]],
    scores: dict[str, int],
    already_duped: dict[str, tuple[str, str]],
    field_extractor,
    field_name: str,
) -> int:
    """
    Generic dedup: group by a field, keep the best in each group.

    Args:
        resumes: all resumes
        scores: completeness scores per resume_id
        already_duped: IDs already marked duplicate (skip these)
        field_extractor: function that takes a ResumeSchema and
            returns the grouping key (email, phone, name+location)
        field_name: label for logging

    Returns:
        Number of new duplicates found in this pass.

    WHY a generic function?
    Email, phone, and name+location dedup all follow the
    same pattern: group by field → keep best → mark rest.
    Writing it once avoids three copies of the same logic.
    """
    groups = defaultdict(list)

    for rid, resume in resumes:
        # Skip if already marked duplicate
        if rid in already_duped:
            continue

        key = field_extractor(resume)
        if key:  # skip empty values
            groups[key].append(rid)

    new_dupes = 0

    for key, group_ids in groups.items():
        if len(group_ids) <= 1:
            continue

        # Sort by completeness score — best first
        group_ids.sort(key=lambda rid: scores[rid], reverse=True)

        # Keep the best, mark rest as duplicates
        best_id = group_ids[0]
        for dup_id in group_ids[1:]:
            if dup_id not in already_duped:
                already_duped[dup_id] = (best_id, f"Same {field_name}: {key}")
                new_dupes += 1
                print(f"    {dup_id} → duplicate of {best_id} "
                      f"(same {field_name})")

    return new_dupes


# ──────────────────────────────────────────────
#  FIELD NORMALIZATION
# ──────────────────────────────────────────────

def _normalize_phone(phone: str) -> str:
    """
    Normalize phone numbers for comparison.

    "+91-9905761112", "9905761112", "+91 990 576 1112"
    all become "9905761112" (last 10 digits).

    WHY last 10 digits?
    Indian phone numbers are 10 digits. Country code (+91)
    and formatting (dashes, spaces, parens) vary but the
    core number is the same. Taking the last 10 digits
    normalizes all formats.
    """
    if not phone:
        return ""

    # Keep only digits
    digits = "".join(c for c in phone if c.isdigit())

    # Take last 10 digits (handles country code prefix)
    if len(digits) >= 10:
        return digits[-10:]

    return digits


def _normalize_name(name: str) -> str:
    """
    Normalize name for comparison.

    "Sai Harsha Vadde", "SAI HARSHA VADDE", "  sai  harsha vadde  "
    all become "sai harsha vadde".
    """
    if not name:
        return ""

    name = name.lower().strip()

    # Remove common prefixes
    for prefix in ("mr.", "ms.", "mrs.", "dr.", "mr ", "ms ", "mrs ", "dr "):
        if name.startswith(prefix):
            name = name[len(prefix):].strip()

    # Collapse multiple spaces
    return " ".join(name.split())


def _name_location_key(resume: ResumeSchema) -> str:
    """
    Create a compound key from name + location.

    Only name alone is risky — common names like
    "Rahul Sharma" could be different people.
    Name + location together is much stronger:
    "rahul sharma|bangalore" is more likely one person.

    Returns empty string if either field is missing
    (we don't want to group unknowns together).
    """
    name = _normalize_name(resume.contact.name)
    location = resume.contact.location.lower().strip()

    if not name or not location:
        return ""

    return f"{name}|{location}"


# ──────────────────────────────────────────────
#  COMPLETENESS SCORING
# ──────────────────────────────────────────────

def _completeness_score(resume: ResumeSchema) -> int:
    """
    Score a resume by how complete it is.

    When duplicates are found, we keep the resume with
    the highest score — it has the most extracted data.

    This ensures we keep Resume(4) over Resume(1) if
    the person added more details in the newer version.
    """
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