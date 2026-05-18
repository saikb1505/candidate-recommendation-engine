from pydantic import BaseModel, Field


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

    def to_embedding_text(self) -> str:
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
        return {
            "name": self.contact.name,
            "email": self.contact.email,
            "location": self.contact.location,
            "skills": ", ".join(self.skills),
            "total_years_experience": self.total_years_experience,
            "highest_education": self.highest_education,
            "source_file": self.source_file,
        }
