"""
Resume JSON schema — the contract for your entire pipeline.

This file defines WHAT a resume looks like as structured data.
Every other module depends on this:
  - LLM extractor uses it to build the prompt
  - Validator checks extracted data against it
  - Embedder reads it to build embedding text
  - Vector store reads it to build metadata filters
  - FastAPI will use it as response models (later)

WHY Pydantic?
  - Automatic validation: if the LLM returns skills as a string
    instead of a list, Pydantic catches it
  - Default values: missing fields don't crash the pipeline
  - FastAPI uses Pydantic for request/response models, so
    these same classes become your API schemas later — zero rewrite
  - .model_dump() gives you a clean dict for JSON serialization

DESIGN DECISION — what to embed vs what to filter:
  - Embed: summary, role descriptions, project details
    → These carry MEANING. "Built a recommendation engine using
      collaborative filtering" is semantically rich.
  - Filter: skills list, years of experience, location, education
    → These are CATEGORICAL. You don't need semantic similarity
      to match "Python" — exact match is better.
"""

from pydantic import BaseModel, Field


class ContactInfo(BaseModel):
    """Contact details extracted from the resume."""
    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin: str = ""


class Experience(BaseModel):
    """A single work experience entry."""
    title: str = ""              # "Senior Software Engineer"
    company: str = ""            # "Google"
    duration: str = ""           # "Jan 2020 - Mar 2023"
    years: float = 0.0           # 3.2 (computed by LLM)
    description: str = ""        # What they did — THIS gets embedded


class Education(BaseModel):
    """A single education entry."""
    degree: str = ""             # "B.Tech Computer Science"
    institution: str = ""        # "IIT Bombay"
    year: str = ""               # "2018"


class ResumeSchema(BaseModel):
    """
    The target structure for every resume.

    Think of this as a database table definition.
    Every resume becomes one row with these columns.
    """

    # --- Contact (metadata — for display, not search) ---
    contact: ContactInfo = Field(default_factory=ContactInfo)

    # --- Filterable metadata ---
    skills: list[str] = Field(default_factory=list)
    total_years_experience: float = 0.0
    highest_education: str = ""    # "B.Tech", "M.Tech", "PhD"

    # --- Embeddable content (semantic search) ---
    summary: str = ""
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)

    # --- Internal tracking ---
    source_file: str = ""          # original filename
    extraction_method: str = ""    # "vision" for our pipeline

    def to_embedding_text(self) -> str:
        """
        Flatten the resume into natural language for embedding.

        WHY not embed the raw JSON?
        JSON has structural noise: {"skills": ["python"]}
        The braces, quotes, and keys dilute the embedding.

        Instead we produce:
        "Senior backend engineer at Google (2020-2023).
         Built distributed payment systems serving 10M users..."

        This captures MEANING, which is what embeddings encode.
        """
        parts = []

        if self.summary:
            parts.append(self.summary)

        for exp in self.experience:
            exp_text = f"{exp.title} at {exp.company}"
            if exp.duration:
                exp_text += f" ({exp.duration})"
            if exp.description:
                exp_text += f". {exp.description}"
            parts.append(exp_text)

        for edu in self.education:
            edu_text = f"{edu.degree} from {edu.institution}"
            if edu.year:
                edu_text += f" ({edu.year})"
            parts.append(edu_text)

        return ". ".join(parts)

    def to_metadata(self) -> dict:
        """
        Extract filterable metadata for the vector database.
        
        Skills stored as individual metadata keys because
        ChromaDB 1.5+ $contains only works on list fields,
        not substring matching on strings.
        
        For location, we store the full string and use $eq.
        For skills filtering, we use a different approach —
        store each skill as a separate boolean metadata field
        would be impractical for 40+ skills.
        
        Instead, we keep skills as a comma-separated string
        for display, and do skill filtering in Python after
        the vector search returns results.
        """
        return {
            "name": self.contact.name,
            "email": self.contact.email,
            "location": self.contact.location,
            "skills": ", ".join(self.skills),
            "total_years_experience": self.total_years_experience,
            "highest_education": self.highest_education,
            "source_file": self.source_file,
        }