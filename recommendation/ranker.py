"""
Recommendation engine — the payoff of your entire pipeline.

Takes a job description as input, returns ranked candidates.

What it does:
  1. Parse the JD to extract requirements (skills, years, location)
  2. Search the vector DB (semantic + filters)
  3. Re-rank results using an LLM for deeper matching
  4. Return candidates with match explanations

WHY re-rank with an LLM?
  Vector search is fast but shallow — it matches on overall
  similarity. A resume might score high because it mentions
  many of the same words, even if the actual experience
  doesn't fit.

  Re-ranking sends the top candidates + the JD to Claude
  and asks: "How well does this candidate actually match?"
  This catches nuances that embeddings miss:
    - "5 years required" but candidate has 2 years
    - "Must lead a team" but candidate was individual contributor
    - "Startup experience preferred" — embeddings don't capture this

  Re-ranking is expensive (one LLM call per search), so we
  only re-rank the top 20 candidates, not all 10K.

TWO MODES:
  fast_match:  vector search + filters only (free, instant)
  smart_match: vector search + LLM re-ranking (costs ~$0.01, 2-3 seconds)

  FastAPI can offer both:
    GET  /match?q=...           → fast_match
    POST /match (with JD body)  → smart_match
"""

import asyncio
import json
from dataclasses import dataclass

from anthropic import AsyncAnthropic

from config.settings import config
from embeddings.vector_store import search, search_with_skills
from embeddings.embedder import embed_text


# ──────────────────────────────────────────────
#  RESULT FORMAT
# ──────────────────────────────────────────────

@dataclass
class CandidateMatch:
    """
    A single candidate result with match details.

    This is what the API returns for each candidate.
    Clean, structured, ready for a frontend to display.
    """
    rank: int
    name: str
    email: str
    location: str
    skills: list[str]
    years_experience: float
    similarity_score: float
    match_explanation: str = ""    # filled by LLM re-ranking
    resume_id: str = ""

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


# ──────────────────────────────────────────────
#  FAST MATCH (vector search only, free)
# ──────────────────────────────────────────────

async def fast_match(
    query: str,
    required_skills: list[str] | None = None,
    min_years: float | None = None,
    location: str | None = None,
    top_k: int = 10,
) -> list[CandidateMatch]:
    """
    Quick candidate matching — vector search + filters.

    Use this for:
      - Instant results (no API call)
      - Simple queries ("Python developer in Bangalore")
      - When cost matters (completely free)

    Args:
        query: natural language description of what you need
        required_skills: must-have skills
        min_years: minimum experience
        location: preferred location
        top_k: number of results

    Returns:
        List of CandidateMatch objects, ranked by similarity.
    """
    results = await asyncio.to_thread(
        search_with_skills,
        query_text=query,
        required_skills=required_skills,
        min_years=min_years,
        location=location,
        top_k=top_k,
    )

    return _format_results(results)


# ──────────────────────────────────────────────
#  SMART MATCH (vector search + LLM re-ranking)
# ──────────────────────────────────────────────

async def smart_match(
    job_description: str,
    top_k: int = 10,
) -> list[CandidateMatch]:
    """
    Full recommendation pipeline — the main feature.

    A recruiter pastes a complete job description.
    The system:
      1. Extracts requirements from the JD (skills, years, location)
      2. Runs vector search with those filters
      3. Sends top candidates to Claude for re-ranking
      4. Returns candidates with match explanations

    Args:
        job_description: full JD text (can be paragraphs long)
        top_k: number of final results to return

    Returns:
        List of CandidateMatch with explanations, best first.

    Cost: ~$0.01-0.02 per search (one Haiku call for
    parsing JD + one for re-ranking). Worth it for the
    quality improvement over pure vector search.
    """
    # Step 1: Parse the JD to extract requirements
    print("  Parsing job description...")
    requirements = await _parse_job_description(job_description)

    print(f"    Skills: {requirements.get('skills', [])}")
    print(f"    Min years: {requirements.get('min_years', 'any')}")
    print(f"    Location: {requirements.get('location', 'any')}")

    # Step 2: Vector search with extracted filters
    print("  Searching candidates...")
    fetch_k = min(top_k * 3, 30)  # fetch more for re-ranking

    results = await asyncio.to_thread(
        search_with_skills,
        query_text=job_description,
        required_skills=requirements.get("skills"),
        min_years=requirements.get("min_years"),
        location=requirements.get("location"),
        top_k=fetch_k,
    )

    if not results:
        # Retry without filters if no results found
        print("  No filtered results, trying without filters...")
        results = await asyncio.to_thread(search, job_description, fetch_k)

    if not results:
        print("  No candidates found")
        return []

    # Step 3: Re-rank with LLM
    print(f"  Re-ranking top {len(results)} candidates with LLM...")
    candidates = _format_results(results)
    ranked = await _rerank_with_llm(job_description, candidates, top_k)

    return ranked


# ──────────────────────────────────────────────
#  JD PARSING
# ──────────────────────────────────────────────

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


async def _parse_job_description(jd_text: str) -> dict:
    """
    Extract structured requirements from a job description.

    Sends the JD to Claude and gets back:
      - Required skills (for metadata filtering)
      - Minimum years (for metadata filtering)
      - Location preference (for metadata filtering)
      - Role summary (for semantic search)

    WHY use an LLM for this?
    Job descriptions are messy and inconsistent:
      "3-5 years of Python experience" → min_years: 3
      "Must be proficient in React and Node" → skills: ["React", "Node.js"]
      "Based in our Bangalore office" → location: "Bangalore"

    Regex would need hundreds of rules. The LLM handles
    all variations in one call.
    """
    client = AsyncAnthropic()

    try:
        response = await client.messages.create(
            model=config.llm_model,
            max_tokens=500,
            system=JD_PARSE_PROMPT,
            messages=[
                {"role": "user", "content": jd_text}
            ],
        )

        raw_text = response.content[0].text.strip()

        # Clean markdown if present
        if raw_text.startswith("```"):
            first_nl = raw_text.index("\n")
            raw_text = raw_text[first_nl + 1:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()

        parsed = json.loads(raw_text)

        # Clean up
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
        # Return empty — search will work without filters
        return {}


# ──────────────────────────────────────────────
#  LLM RE-RANKING
# ──────────────────────────────────────────────

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


async def _rerank_with_llm(
    job_description: str,
    candidates: list[CandidateMatch],
    top_k: int,
) -> list[CandidateMatch]:
    """
    Re-rank candidates using Claude for deeper matching.

    The vector search gives us rough similarity. The LLM
    reads both the JD and each candidate's profile to assess
    actual fit — considering nuances like:
      - Does their experience level match?
      - Are the skills relevant or just keyword matches?
      - Is their career trajectory aligned with this role?

    WHY not use the LLM for initial search?
    Sending all 10K resumes to an LLM would cost hundreds
    of dollars and take hours. Vector search narrows it to
    20-30 candidates in milliseconds for free. Then the LLM
    only evaluates those 20-30 — costing ~$0.01.

    This is the Retrieval-Augmented Generation pattern:
      Retrieve (vector search) → Augment (add context) → Generate (LLM)
    """
    client = AsyncAnthropic()

    # Build candidate summaries for the LLM
    candidate_info = []
    for c in candidates:
        candidate_info.append({
            "resume_id": c.resume_id,
            "name": c.name,
            "skills": c.skills[:15],  # limit to keep prompt short
            "years_experience": c.years_experience,
            "location": c.location,
            "similarity_score": c.similarity_score,
        })

    prompt = f"""JOB DESCRIPTION:
{job_description}

CANDIDATES:
{json.dumps(candidate_info, indent=2)}

Rank these candidates by match quality. Return top {top_k}."""

    try:
        response = await client.messages.create(
            model=config.llm_model,
            max_tokens=2000,
            system=RERANK_PROMPT,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        raw_text = response.content[0].text.strip()

        # Clean markdown
        if raw_text.startswith("```"):
            first_nl = raw_text.index("\n")
            raw_text = raw_text[first_nl + 1:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()

        rankings = json.loads(raw_text)

        # Map LLM rankings back to candidates
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
        # Fall back to vector search order
        for i, c in enumerate(candidates[:top_k]):
            c.rank = i + 1
            c.match_explanation = "Ranked by semantic similarity"
        return candidates[:top_k]


# ──────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────

def _format_results(results: list[dict]) -> list[CandidateMatch]:
    """Convert raw search results to CandidateMatch objects."""
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
        ))

    return candidates