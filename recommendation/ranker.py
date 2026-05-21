"""
Recommendation engine: takes a job description, returns ranked candidates.

fast_match:  vector search + metadata filters only (free, instant)
smart_match: vector search + LLM re-ranking with explanations (~$0.01)
"""

import json
from dataclasses import dataclass

from anthropic import Anthropic
from qdrant_client import QdrantClient

from config.settings import config
from embeddings.vector_store import search_by_chunks, search_with_skills_by_chunks


@dataclass
class CandidateMatch:
    rank: int
    name: str
    email: str
    location: str
    skills: list[str]
    years_experience: float
    similarity_score: float
    match_explanation: str = ""
    resume_id: str = ""
    resume_text: str = ""      # embedding text — used by LLM re-ranker, not returned to API

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "name": self.name,
            "email": self.email,
            "location": self.location,
            "skills": self.skills,
            "years_experience": self.years_experience,
            "similarity_score": self.similarity_score,
            "match_explanation": self.match_explanation,
            "resume_id": self.resume_id,
        }


def fast_match(
    query: str,
    required_skills: list[str] | None = None,
    min_years: float | None = None,
    location: str | None = None,
    top_k: int = 10,
    client: QdrantClient | None = None,
) -> list[CandidateMatch]:
    results = search_with_skills_by_chunks(
        query_text=query,
        required_skills=required_skills,
        min_years=min_years,
        location=location,
        top_k=top_k,
        client=client,
    )

    return _format_results(results)


def smart_match(
    job_description: str,
    top_k: int = 10,
    client: QdrantClient | None = None,
) -> list[CandidateMatch]:
    print("  Parsing job description...")
    requirements = _parse_job_description(job_description)

    print(f"    Skills: {requirements.get('skills', [])}")
    print(f"    Min years: {requirements.get('min_years', 'any')}")
    print(f"    Location: {requirements.get('location', 'any')}")

    print("  Searching candidates...")
    fetch_k = min(top_k * 3, 30)

    results = search_with_skills_by_chunks(
        query_text=job_description,
        required_skills=requirements.get("skills"),
        min_years=requirements.get("min_years"),
        location=requirements.get("location"),
        top_k=fetch_k,
        client=client,
    )

    if not results:
        print("  No filtered results, trying without filters...")
        results = search_by_chunks(job_description, fetch_k, client=client)

    if not results:
        print("  No candidates found")
        return []

    print(f"  Re-ranking top {len(results)} candidates with LLM...")
    candidates = _format_results(results)
    ranked = _rerank_with_llm(job_description, candidates, top_k)

    return ranked


JD_PARSE_PROMPT = """You are a job description parser. Extract the key requirements
from the job description below.

Return ONLY valid JSON with this structure:
{
    "skills": ["skill1", "skill2"],
    "min_years": 5,
    "location": "city name or null",
    "role_summary": "one sentence describing the ideal candidate"
}

RULES:
1. skills: list the most important technical skills (max 5).
   Use common names: "Python" not "python programming language".
2. min_years: minimum years of experience mentioned. Use 0 if not specified.
3. location: city/region if mentioned, null if remote or not specified.
4. role_summary: one sentence capturing what this role needs.

Return ONLY the JSON. No markdown, no explanation."""


def _parse_job_description(jd_text: str) -> dict:
    client = Anthropic()

    try:
        response = client.messages.create(
            model=config.llm_model,
            max_tokens=500,
            system=JD_PARSE_PROMPT,
            messages=[
                {"role": "user", "content": jd_text}
            ],
        )

        raw_text = response.content[0].text.strip()
        raw_text = _strip_markdown_json(raw_text)
        parsed = json.loads(raw_text)

        result = {}
        if parsed.get("skills"):
            result["skills"] = parsed["skills"][:5]
        if parsed.get("min_years") and parsed["min_years"] > 0:
            result["min_years"] = float(parsed["min_years"])
        if parsed.get("location") and parsed["location"] != "null":
            result["location"] = parsed["location"]
        if parsed.get("role_summary"):
            result["role_summary"] = parsed["role_summary"]

        return result

    except Exception as e:
        print(f"    JD parsing failed: {e}")
        return {}


RERANK_PROMPT = """You are a recruitment matching expert. Given a job description
and a list of candidates, rank them by how well they match the role.

For each candidate, provide:
1. A match score from 1-10 (10 = perfect match)
2. A brief explanation of why they match or don't (max 2 sentences)

Return ONLY valid JSON as a list:
[
    {
        "resume_id": "candidate_id_here",
        "match_score": 8,
        "explanation": "Strong Python background with 5 years of API development.
                        Lacks AWS experience mentioned in JD."
    }
]

Sort by match_score descending (best match first).
Return ONLY the JSON array. No markdown, no explanation."""


def _rerank_with_llm(
    job_description: str,
    candidates: list[CandidateMatch],
    top_k: int,
) -> list[CandidateMatch]:
    client = Anthropic()

    candidate_info = []
    for c in candidates:
        candidate_info.append({
            "resume_id": c.resume_id,
            "name": c.name,
            "years_experience": c.years_experience,
            "location": c.location,
            "resume": c.resume_text,
        })

    prompt = f"""JOB DESCRIPTION:
{job_description}

CANDIDATES:
{json.dumps(candidate_info, indent=2)}

Rank these candidates by match quality. Return top {top_k}."""

    try:
        response = client.messages.create(
            model=config.llm_model,
            max_tokens=2000,
            system=RERANK_PROMPT,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        raw_text = _strip_markdown_json(response.content[0].text.strip())
        rankings = json.loads(raw_text)

        candidate_map = {c.resume_id: c for c in candidates}
        ranked = []

        for i, ranking in enumerate(rankings[:top_k]):
            rid = ranking.get("resume_id", "")
            if rid in candidate_map:
                candidate = candidate_map[rid]
                candidate.rank = i + 1
                candidate.match_explanation = ranking.get("explanation", "")
                ranked.append(candidate)

        return ranked

    except Exception as e:
        print(f"    Re-ranking failed: {e}")
        for i, c in enumerate(candidates[:top_k]):
            c.rank = i + 1
            c.match_explanation = "Ranked by semantic similarity"
        return candidates[:top_k]


def _format_results(results: list[dict]) -> list[CandidateMatch]:
    candidates = []

    for i, result in enumerate(results):
        metadata = result["metadata"]
        skills_str = metadata.get("skills", "")
        skills_list = [s.strip() for s in skills_str.split(",") if s.strip()]

        candidates.append(CandidateMatch(
            rank=i + 1,
            name=metadata.get("name", "Unknown"),
            email=metadata.get("email", ""),
            location=metadata.get("location", ""),
            skills=skills_list,
            years_experience=metadata.get("total_years_experience", 0),
            similarity_score=result["score"],
            resume_id=result["id"],
            resume_text=result.get("document", ""),
        ))

    return candidates


def _strip_markdown_json(text: str) -> str:
    if text.startswith("```"):
        first_nl = text.index("\n")
        text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()
