"""
LLM-based resume extractor.

Primary path:  file → docling → markdown → Groq (text LLM) → JSON
Fallback path: file → converter.py → base64 images → GPT-4o-mini vision → JSON

Images are only produced if Groq fails — lazy conversion on demand.
"""

import json
from datetime import date

from groq import Groq
from openai import OpenAI

from config.settings import config
from ingestion.converter import convert_to_images


# ──────────────────────────────────────────────
#  SYSTEM PROMPT  (shared by both paths)
# ──────────────────────────────────────────────

def _build_system_prompt() -> str:
    today = date.today().strftime("%b %Y")
    return f"""You are a resume data extractor. Extract structured information into JSON format.

Today's date is {today}. Use this to calculate durations when a role says "Present" or "Current".

RULES:
1. Return ONLY valid JSON. No markdown, no explanation, no ```json blocks.
2. Extract exactly these fields — no more, no less.
3. If a field is not found, use empty string "" or empty list [].
4. For total_years_experience, sum the years across all experience entries. Treat "Present"/"Current" as {today}.
5. Skills should be individual items: ["Python", "AWS"] not ["Python and AWS"].
6. For experience descriptions, capture what the person DID, not their job title again.

REQUIRED JSON STRUCTURE:
{{
    "contact": {{
        "name": "Full Name",
        "email": "email@example.com",
        "phone": "+91-1234567890",
        "location": "City, State",
        "linkedin": "linkedin.com/in/username"
    }},
    "title": "Software Engineer",
    "summary": "Professional summary or objective if present",
    "skills": ["Skill1", "Skill2", "Skill3"],
    "total_years_experience": 5.0,
    "highest_education": "B.Tech / M.Tech / PhD / MBA etc",
    "experience": [
        {{
            "title": "Job Title",
            "company": "Company Name",
            "duration": "Jan 2020 - Mar 2023",
            "years": 3.2,
            "description": "What they accomplished in this role"
        }}
    ],
    "education": [
        {{
            "degree": "Degree Name and Field",
            "institution": "University Name",
            "graduation_year": "2020"
        }}
    ]
}}

Remember: ONLY output the JSON object. Nothing else."""


# ──────────────────────────────────────────────
#  PRIMARY PATH — Groq (markdown text)
# ──────────────────────────────────────────────

def _to_markdown(file_path: str) -> str:
    from docling.document_converter import DocumentConverter
    converter = DocumentConverter()
    result = converter.convert(file_path)
    return result.document.export_to_markdown()


def _extract_with_groq(file_path: str) -> dict:
    try:
        markdown = _to_markdown(file_path)
    except Exception as e:
        return {"_error": f"docling conversion failed: {e}"}

    try:
        client = Groq()
        response = client.chat.completions.create(
            model=config.groq_model,
            messages=[
                {"role": "system", "content": _build_system_prompt()},
                {"role": "user", "content": f"Extract resume data from this markdown:\n\n{markdown}"},
            ],
            max_tokens=config.max_tokens,
        )
        raw_text = response.choices[0].message.content
        result = _parse_json_response(raw_text)
        if "_error" not in result:
            result["_extraction_method"] = "groq"
        return result
    except Exception as e:
        return {"_error": f"Groq API failed: {e}"}


# ──────────────────────────────────────────────
#  FALLBACK PATH — GPT-4o-mini vision (images)
# ──────────────────────────────────────────────

def _extract_with_vision(file_path: str, file_type: str) -> dict:
    conversion = convert_to_images(file_path, file_type)
    if not conversion.success:
        return {"_error": f"Image conversion failed: {conversion.error}"}

    content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img}"},
        }
        for img in conversion.images
    ]
    content.append({
        "type": "text",
        "text": "Extract all information from this resume into the JSON structure defined in your instructions.",
    })

    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model=config.openai_vision_model,
            messages=[
                {"role": "system", "content": _build_system_prompt()},
                {"role": "user", "content": content},
            ],
            max_tokens=config.max_tokens,
        )
        raw_text = response.choices[0].message.content
        result = _parse_json_response(raw_text)
        if "_error" not in result:
            result["_extraction_method"] = "gpt4o-mini-vision"
        return result
    except Exception as e:
        return {"_error": f"OpenAI API failed: {e}"}


# ──────────────────────────────────────────────
#  ORCHESTRATOR
# ──────────────────────────────────────────────

def _groq_result_is_incomplete(result: dict) -> bool:
    contact = result.get("contact", {})
    missing_contact = not (
        contact.get("name") and contact.get("email") and contact.get("phone")
    )
    missing_experience = not result.get("experience")
    return missing_contact or missing_experience


def extract_resume_with_retry(file_path: str, file_type: str) -> dict:
    """
    Try Groq first (docling markdown → text LLM).
    Fall back to GPT-4o-mini vision if Groq fails or returns
    incomplete contact info (name/email/phone) or empty experience.
    """
    result = _extract_with_groq(file_path)

    if "_error" not in result and not _groq_result_is_incomplete(result):
        return result

    return _extract_with_vision(file_path, file_type)


# ──────────────────────────────────────────────
#  JSON PARSING  (shared by both paths)
# ──────────────────────────────────────────────

def _parse_json_response(text: str) -> dict:
    text = text.strip()

    if text.startswith("```"):
        first_newline = text.index("\n")
        text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return {"_error": f"Failed to parse JSON: {text[:200]}"}
