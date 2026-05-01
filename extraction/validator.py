"""
JSON validator for extracted resume data.

Takes the raw dict from llm_extractor and:
  1. Validates it against ResumeSchema
  2. Cleans up common LLM output quirks
  3. Returns a validated ResumeSchema object or an error

WHY a separate validator?
  - Extraction (API call) and validation (data checking) are
    different concerns. If validation fails, you might be able
    to fix the JSON without re-calling the API.
  - You can run validation independently — useful for testing
    and debugging without spending API credits.
  - When you add FastAPI later, the validator becomes middleware
    that ensures every response matches the schema.

COMMON LLM OUTPUT QUIRKS this handles:
  - Skills as a comma-separated string instead of a list
  - Years of experience as a string "5 years" instead of float 5.0
  - Empty nested objects instead of empty strings
  - Extra fields the LLM invented that aren't in the schema
"""

import re

from config.schema import ResumeSchema, ContactInfo, Experience, Education


def validate_resume(
    data: dict,
    source_file: str = "",
) -> tuple[ResumeSchema | None, str]:
    """
    Validate and clean raw extracted data.

    Args:
        data: raw dict from llm_extractor
        source_file: original filename (for tracking)

    Returns:
        Tuple of (ResumeSchema or None, error_message).
        If validation succeeds: (schema, "")
        If validation fails:   (None, "what went wrong")

    WHY return a tuple instead of raising exceptions?
    In a pipeline processing 10K files, you don't want one
    bad resume to crash everything. Returning errors as values
    lets the pipeline log the failure and move on.
    """
    # Check for extraction errors passed through
    if "_error" in data:
        return None, data["_error"]

    try:
        # Clean the data before validation
        cleaned = _clean_raw_data(data)

        # Parse into ResumeSchema (Pydantic validates types)
        resume = ResumeSchema(**cleaned)

        # Add tracking info
        resume.source_file = source_file
        resume.extraction_method = "vision"

        # Run business logic checks
        issues = _check_data_quality(resume)
        if issues:
            # Log issues but don't reject — partial data is
            # still useful. The issues are for debugging.
            pass

        return resume, ""

    except Exception as e:
        return None, f"Validation failed: {str(e)}"


# ──────────────────────────────────────────────
#  DATA CLEANING
# ──────────────────────────────────────────────

def _clean_raw_data(data: dict) -> dict:
    """
    Fix common LLM output quirks before Pydantic validation.

    WHY clean before validating?
    Pydantic is strict — if the LLM returns skills as
    "Python, Java, SQL" (a string) instead of ["Python", "Java", "SQL"]
    (a list), Pydantic rejects it. Cleaning first means fewer
    false validation failures and fewer wasted API retries.
    """
    # --- Fix skills: string → list ---
    if isinstance(data.get("skills"), str):
        data["skills"] = [
            s.strip()
            for s in data["skills"].split(",")
            if s.strip()
        ]

    # --- Fix total_years_experience: string → float ---
    years = data.get("total_years_experience")
    if isinstance(years, str):
        data["total_years_experience"] = _parse_years(years)

    # --- Fix experience entries ---
    if isinstance(data.get("experience"), list):
        for exp in data["experience"]:
            if isinstance(exp, dict):
                # Fix years within experience
                if isinstance(exp.get("years"), str):
                    exp["years"] = _parse_years(exp["years"])
                # Ensure description is a string
                if exp.get("description") is None:
                    exp["description"] = ""

    # --- Fix contact: sometimes LLM nests it differently ---
    if not isinstance(data.get("contact"), dict):
        data["contact"] = {}

    # --- Remove any extra fields LLM invented ---
    # Pydantic would ignore them, but being explicit is cleaner
    known_top_fields = {
        "contact", "summary", "skills", "total_years_experience",
        "highest_education", "experience", "education",
        "source_file", "extraction_method",
    }
    extra_keys = set(data.keys()) - known_top_fields - {"_fallback_model"}
    for key in extra_keys:
        del data[key]

    return data


def _parse_years(value: str) -> float:
    """
    Extract a numeric year value from messy strings.

    Handles:
      "5 years"      → 5.0
      "5.5"          → 5.5
      "5+ years"     → 5.0
      "five years"   → 0.0  (too hard to parse, default)
      ""             → 0.0

    WHY not ask the LLM to always return a float?
    We do (in the system prompt), but LLMs don't always
    follow instructions perfectly. Defensive parsing here
    means one less reason to retry an API call.
    """
    if not value:
        return 0.0

    # Find all numbers (including decimals) in the string
    numbers = re.findall(r"(\d+\.?\d*)", str(value))
    if numbers:
        return float(numbers[0])

    return 0.0


# ──────────────────────────────────────────────
#  DATA QUALITY CHECKS
# ──────────────────────────────────────────────

def _check_data_quality(resume: ResumeSchema) -> list[str]:
    """
    Check for potential data quality issues.

    These are warnings, not errors — the resume is still valid
    but might have incomplete extraction.

    WHY check after validation?
    Pydantic checks types (is skills a list?). This function
    checks semantics (is the skills list empty? does the
    experience total match total_years_experience?).
    """
    issues = []

    # Check: has a name?
    if not resume.contact.name:
        issues.append("Missing name")

    # Check: has any contact info?
    if not resume.contact.email and not resume.contact.phone:
        issues.append("No email or phone found")

    # Check: has skills?
    if not resume.skills:
        issues.append("No skills extracted")

    # Check: has experience?
    if not resume.experience:
        issues.append("No work experience extracted")

    # Check: experience years vs total years
    if resume.experience and resume.total_years_experience > 0:
        exp_years_sum = sum(e.years for e in resume.experience)
        if exp_years_sum > 0:
            diff = abs(exp_years_sum - resume.total_years_experience)
            if diff > 3:
                issues.append(
                    f"Years mismatch: total={resume.total_years_experience}, "
                    f"sum of roles={exp_years_sum}"
                )

    return issues