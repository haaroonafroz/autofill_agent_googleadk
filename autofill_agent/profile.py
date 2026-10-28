"""
Structured candidate profile extracted from the CV.

Form fields such as "Phone" or "Current employer" are exact lookups, so the
profile stores them as typed fields. Open questions still go to CV search.
"""

from typing import List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


class Experience(BaseModel):
    company: str = Field(description="Employer name exactly as written in the CV.")
    title: str = Field(description="Job title.")
    start: Optional[str] = Field(None, description="Start date as written, e.g. 'Jan 2023'.")
    end: Optional[str] = Field(None, description="End date as written, or 'Present'.")
    location: Optional[str] = None
    highlights: List[str] = Field(default_factory=list, description="Key responsibilities or achievements.")


class Education(BaseModel):
    institution: str
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    grade: Optional[str] = Field(None, description="Grade, GPA or classification if stated.")


class Project(BaseModel):
    name: str
    description: Optional[str] = None
    url: Optional[str] = None


class CandidateProfile(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = Field(None, description="Current city/country if stated.")
    linkedin: Optional[str] = None
    github: Optional[str] = None
    website: Optional[str] = None
    summary: Optional[str] = Field(None, description="Professional summary or profile statement.")
    work_authorization: Optional[str] = Field(
        None, description="Right to work / visa status only if the CV states it."
    )
    experience: List[Experience] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)


EXTRACTION_PROMPT = """You extract a structured candidate profile from a CV.

Rules:
- Use only information that appears in the CV. Never invent or infer facts.
- If a field is not stated, leave it null (or an empty list). Do not guess.
- Copy names, dates and contact details exactly as written.
- Keep highlights short: one line per achievement or responsibility."""


def extract_profile(llm: BaseChatModel, cv_markdown: str) -> CandidateProfile:
    structured_llm = llm.with_structured_output(CandidateProfile, method="json_schema")
    return structured_llm.invoke(
        [
            SystemMessage(content=EXTRACTION_PROMPT),
            HumanMessage(content=f"CV (Markdown):\n\n{cv_markdown}"),
        ]
    )
