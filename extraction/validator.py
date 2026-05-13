import re

from config.schema import ResumeSchema, ContactInfo, Experience, Education


def validate_resume(
    data: dict,
    source_file: str = "",
) -> tuple[ResumeSchema | None, str]:
    if "_error" in data:
        return None, data["_error"]

    try:
        extraction_method = data.get("_extraction_method", "unknown")
        cleaned = _clean_raw_data(data)
        resume = ResumeSchema(**cleaned)
        resume.source_file = source_file
        resume.extraction_method = extraction_method
        return resume, ""

    except Exception as e:
        return None, f"Validation failed: {str(e)}"


def _clean_raw_data(data: dict) -> dict:
    if isinstance(data.get("skills"), str):
        data["skills"] = [
            s.strip()
            for s in data["skills"].split(",")
            if s.strip()
        ]

    years = data.get("total_years_experience")
    if isinstance(years, str):
        data["total_years_experience"] = _parse_years(years)

    if isinstance(data.get("experience"), list):
        for exp in data["experience"]:
            if isinstance(exp, dict):
                if isinstance(exp.get("years"), str):
                    exp["years"] = _parse_years(exp["years"])
                if exp.get("description") is None:
                    exp["description"] = ""

    if not isinstance(data.get("contact"), dict):
        data["contact"] = {}

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
    if not value:
        return 0.0

    numbers = re.findall(r"(\d+\.?\d*)", str(value))
    if numbers:
        return float(numbers[0])

    return 0.0


def _check_data_quality(resume: ResumeSchema) -> list[str]:
    issues = []

    if not resume.contact.name:
        issues.append("Missing name")

    if not resume.contact.email and not resume.contact.phone:
        issues.append("No email or phone found")

    if not resume.skills:
        issues.append("No skills extracted")

    if not resume.experience:
        issues.append("No work experience extracted")

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
