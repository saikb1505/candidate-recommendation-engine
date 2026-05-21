from dataclasses import dataclass

from pydantic import BaseModel, Field


@dataclass
class ResumeChunk:
    chunk_type: str   # "summary", "experience", "education"
    chunk_index: int
    text: str


class ContactInfo(BaseModel):
    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin: str = ""


class Experience(BaseModel):
    title: str = ""
    company: str = ""
    duration: str = ""
    years: float = 0.0
    description: str = ""


class Education(BaseModel):
    degree: str = ""
    institution: str = ""
    year: str = ""


class ResumeSchema(BaseModel):
    contact: ContactInfo = Field(default_factory=ContactInfo)

    # Filterable metadata (exact/range match in vector DB)
    skills: list[str] = Field(default_factory=list)
    companies: list[str] = Field(default_factory=list)
    total_years_experience: float = 0.0
    highest_education: str = ""

    # Embeddable content (semantic search)
    summary: str = ""
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)

    source_file: str = ""
    extraction_method: str = ""

    def to_chunks(self) -> list[ResumeChunk]:
        chunks: list[ResumeChunk] = []

        # Chunk 0: summary + skills — candidate's professional identity
        profile_parts = []
        if self.summary:
            profile_parts.append(self.summary)
        if self.skills:
            profile_parts.append("Skills: " + ", ".join(self.skills))
        if profile_parts:
            chunks.append(ResumeChunk("summary", 0, " ".join(profile_parts)))

        # One chunk per experience entry — stays within 384-token limit
        for i, exp in enumerate(self.experience):
            text = f"{exp.title} at {exp.company}"
            if exp.duration:
                text += f" ({exp.duration})"
            if exp.description:
                text += f". {exp.description}"
            chunks.append(ResumeChunk("experience", i, text))

        # Education chunk — typically short, fine to combine
        edu_parts = []
        for edu in self.education:
            edu_text = f"{edu.degree} from {edu.institution}"
            if edu.year:
                edu_text += f" ({edu.year})"
            edu_parts.append(edu_text)
        if edu_parts:
            chunks.append(ResumeChunk("education", 0, ". ".join(edu_parts)))

        return chunks
