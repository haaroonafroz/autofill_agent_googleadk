"""
Metadata for CV entries, extracted once when the CV is indexed.

One small LLM call sees the whole CV, with its sections numbered, and groups the
sections into entries: each job, degree or project is one entry, even when the
CV spreads its company, title and dates over several headings. Each entry gets a
kind and, where the CV states them, an entity (company, school or project name),
a role, and start and end dates.

The model's values are checked against the CV text. A value that does not appear
in the CV is dropped, so an invented date cannot reach the index. Dates are also
stored as sortable keys ("YYYY-MM"), so a list of entries can be returned in a
fixed order rather than by similarity.
"""

import html
import re
import typing
from typing import Dict, List, Literal, Optional, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

Kind = Literal[
    "summary", "experience", "education", "project", "skills",
    "languages", "certifications", "contact", "other",
]
KINDS = typing.get_args(Kind)

# Kinds whose entries are listed newest first.
DATED_KINDS = {"experience", "education"}

PRESENT_KEY = "9999-12"  # sorts after every real date, so current roles come first
UNKNOWN_KEY = "0000-00"  # sorts before every real date, so undated entries come last


class EntryLabel(BaseModel):
    kind: Kind = Field(description="The type of this entry.")
    entity: Optional[str] = Field(
        None, description="Employer, school or project name, exactly as written. Null if the entry has none."
    )
    role: Optional[str] = Field(
        None, description="Job title, degree with field of study, or project role, exactly as written."
    )
    start: Optional[str] = Field(None, description="Start date exactly as written, e.g. '07/2025'. Null if not stated.")
    end: Optional[str] = Field(
        None, description="End date exactly as written, or 'Present' if still ongoing. Null if not stated."
    )
    sections: List[int] = Field(
        description="The numbers of the sections that belong to this entry, from the [S n] markers."
    )


class CVLabels(BaseModel):
    entries: List[EntryLabel] = Field(
        description="Every section must appear in exactly one entry. Headings and other text use kind 'other'."
    )


LABEL_PROMPT = """You group the sections of a CV for a job-application system.

The CV is split into numbered sections, marked [S0], [S1], and so on. Group them into entries.
- Each job, degree or project is one entry. The CV may spread one entry over several headings
  (for example the company in one heading and the title and dates in the next). Put all of
  those sections in the same entry.
- Every section must belong to exactly one entry. Use kind "other" for headings and text that
  is not an entry.
- Copy entity, role, start and end exactly as the CV writes them. Leave a field null when the
  CV does not state it for that entry. Never infer or estimate a date, and never take a date
  from a different entry."""


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text)).strip().lower()


def label_sections(llm: BaseChatModel, sections: Sequence[str], cv_markdown: str) -> Dict[int, EntryLabel]:
    """
    Returns the entry for each section number that the model placed in an entry.
    Values the CV does not contain are removed before they are returned.
    """
    numbered = "\n\n".join(f"[S{i}]\n{text}" for i, text in enumerate(sections))
    structured = llm.with_structured_output(CVLabels, method="json_schema")
    result = structured.invoke([
        SystemMessage(content=LABEL_PROMPT),
        HumanMessage(content=f"CV sections:\n\n{numbered}"),
    ])

    cv_norm = _norm(cv_markdown)
    by_section: Dict[int, EntryLabel] = {}
    for entry in result.entries:
        grounded = entry.model_copy(update={
            "entity": _grounded(entry.entity, cv_norm),
            "role": _grounded(entry.role, cv_norm),
            "start": _grounded(entry.start, cv_norm),
            "end": _grounded(entry.end, cv_norm),
        })
        for number in entry.sections:
            if 0 <= number < len(sections) and number not in by_section:
                by_section[number] = grounded
    return by_section


def _grounded(value: Optional[str], cv_norm: str) -> Optional[str]:
    if not value or not value.strip():
        return None
    return value if _norm(value) in cv_norm else None


_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1
)}
_NUMERIC = re.compile(r"\b(\d{1,2})[/.\-](\d{4})\b")
_MONTH_NAME = re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{4})\b")
_YEAR = re.compile(r"\b(\d{4})\b")
_PRESENT = re.compile(r"\b(present|current|now|today|ongoing)\b", re.IGNORECASE)


def date_key(text: Optional[str]) -> Optional[str]:
    """Turns a date as written ('07/2025', 'Jul 2025', '2025', 'Present') into 'YYYY-MM'."""
    if not text:
        return None
    if _PRESENT.search(text):
        return PRESENT_KEY
    match = _NUMERIC.search(text)
    if match and 1 <= int(match.group(1)) <= 12:
        return f"{match.group(2)}-{int(match.group(1)):02d}"
    match = _MONTH_NAME.search(text)
    if match and match.group(1)[:3].lower() in _MONTHS:
        return f"{match.group(2)}-{_MONTHS[match.group(1)[:3].lower()]:02d}"
    match = _YEAR.search(text)
    if match:
        return f"{match.group(1)}-01"  # only the year is known, so use January
    return None


def describe(entry: EntryLabel) -> str:
    """A one-line label that is prepended to each chunk before embedding."""
    dates = " - ".join(x for x in (entry.start, entry.end) if x)
    return " | ".join(x for x in (entry.kind, entry.entity, entry.role, dates) if x)
