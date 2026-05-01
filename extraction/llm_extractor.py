"""
LLM-based resume extractor.

Takes page images (base64) and sends them to Claude's vision API
to extract structured JSON matching our ResumeSchema.

KEY LEARNING — Prompt engineering for structured extraction:
  The quality of your JSON output depends heavily on:
  1. A clear system prompt that defines the exact schema
  2. Explicit instructions on what to extract vs skip
  3. Telling the model to return ONLY JSON, no markdown
  4. Providing the schema as an example in the prompt

  This is a pattern you'll reuse in many RAG applications —
  converting unstructured data into structured data using LLMs.

WHY a system prompt + user prompt split?
  System prompt: defines the role and output format (same every time)
  User prompt: contains the actual images (changes per resume)

  This split matters for the Batch API later — the system prompt
  can be cached across all 10K requests (90% cost reduction on
  that portion) because it never changes.
"""

import json

from anthropic import Anthropic

from config.settings import config
from config.schema import ResumeSchema


# ──────────────────────────────────────────────
#  SYSTEM PROMPT
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are a resume data extractor. Your job is to look at resume 
images and extract structured information into JSON format.

RULES:
1. Return ONLY valid JSON. No markdown, no explanation, no ```json blocks.
2. Extract exactly these fields — no more, no less.
3. If a field is not found in the resume, use empty string "" or empty list [].
4. For total_years_experience, calculate from the work history dates.
5. Skills should be individual items: ["Python", "AWS"] not ["Python and AWS"].
6. For experience descriptions, capture what the person DID, not their job title again.

REQUIRED JSON STRUCTURE:
{
    "contact": {
        "name": "Full Name",
        "email": "email@example.com",
        "phone": "+91-1234567890",
        "location": "City, State",
        "linkedin": "linkedin.com/in/username"
    },
    "summary": "Professional summary or objective if present",
    "skills": ["Skill1", "Skill2", "Skill3"],
    "total_years_experience": 5.0,
    "highest_education": "B.Tech / M.Tech / PhD / MBA etc",
    "experience": [
        {
            "title": "Job Title",
            "company": "Company Name",
            "duration": "Jan 2020 - Mar 2023",
            "years": 3.2,
            "description": "What they accomplished in this role"
        }
    ],
    "education": [
        {
            "degree": "Degree Name and Field",
            "institution": "University Name",
            "year": "2020"
        }
    ]
}

Remember: ONLY output the JSON object. Nothing else."""


# ──────────────────────────────────────────────
#  EXTRACTOR
# ──────────────────────────────────────────────

def extract_resume(
    images: list[str],
    model: str | None = None,
) -> dict:
    """
    Send resume page images to Claude and get structured JSON back.

    Args:
        images: list of base64-encoded JPEG strings (one per page)
        model: which Claude model to use (defaults to config)

    Returns:
        Parsed dict matching ResumeSchema structure.
        Returns empty dict with "_error" key if extraction fails.

    WHY return a dict instead of ResumeSchema?
    Validation happens in a separate step (validator.py).
    This function's job is: images in → raw JSON out.
    Keeping extraction and validation separate means you can
    retry validation without re-calling the API.
    """
    model = model or config.llm_model
    client = Anthropic()

    # Build the message content: all page images + instruction
    content = []

    for i, img_base64 in enumerate(images):
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": img_base64,
            },
        })

    content.append({
        "type": "text",
        "text": "Extract all information from this resume into the JSON structure defined in your instructions.",
    })

    # Call the API
    try:
        response = client.messages.create(
            model=model,
            max_tokens=config.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": content}
            ],
        )

        # Parse the response text as JSON
        raw_text = response.content[0].text
        return _parse_json_response(raw_text)

    except Exception as e:
        return {"_error": f"API call failed: {str(e)}"}


def extract_resume_with_retry(
    images: list[str],
) -> dict:
    """
    Extract with automatic retry and model fallback.

    Strategy:
      1. Try with Haiku (fast, cheap)
      2. If JSON parsing fails, retry with Haiku once more
      3. If still failing, try with Sonnet (smarter, more expensive)

    WHY retry?
    LLMs occasionally produce malformed JSON — a missing comma,
    an extra bracket, markdown wrapping. A simple retry often
    fixes it because the model's output is non-deterministic.

    WHY fallback to Sonnet?
    Some resumes are genuinely hard — creative layouts, unusual
    formatting, multiple languages. Sonnet's stronger vision
    capabilities handle these edge cases better. At ~3x the cost,
    you only want it as a fallback, not the default.
    """
    # Attempt 1: Primary model (Haiku)
    result = extract_resume(images, model=config.llm_model)
    if "_error" not in result:
        return result

    # Attempt 2: Retry with primary model
    result = extract_resume(images, model=config.llm_model)
    if "_error" not in result:
        return result

    # Attempt 3: Fallback to stronger model (Sonnet)
    result = extract_resume(images, model=config.llm_model_fallback)
    if "_error" not in result:
        result["_fallback_model"] = config.llm_model_fallback
    return result


# ──────────────────────────────────────────────
#  JSON PARSING
# ──────────────────────────────────────────────

def _parse_json_response(text: str) -> dict:
    """
    Parse LLM response text into a Python dict.

    Handles common LLM quirks:
    - Markdown code blocks: ```json ... ```
    - Leading/trailing whitespace
    - Text before or after the JSON object

    WHY not just json.loads()?
    LLMs sometimes wrap JSON in markdown blocks or add
    a brief explanation before/after. This function strips
    all of that to find the actual JSON object.
    """
    text = text.strip()

    # Remove markdown code blocks if present
    if text.startswith("```"):
        # Remove opening ``` or ```json
        first_newline = text.index("\n")
        text = text[first_newline + 1:]
        # Remove closing ```
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object within the text
    # Look for first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return {"_error": f"Failed to parse JSON from response: {text[:200]}"}