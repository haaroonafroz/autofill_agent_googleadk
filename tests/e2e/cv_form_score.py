"""
Deterministic scoring for tests/e2e/fixtures/cv_form.html.

Every field is either answerable from the CV (it must be filled with the expected
value) or not in the CV (it must stay empty, or be asked of the candidate). A value
typed into a field that is not in the CV is an invention and is reported separately.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from autofill_agent.profile import CandidateProfile

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FORM_FILE = FIXTURES_DIR / "cv_form.html"
EXPECTED_FILE = FIXTURES_DIR / "cv_form_expected.json"
PROFILE_FILE = FIXTURES_DIR / "cv_profile.json"

# What the live test answers when the agent asks the candidate something.
ANSWER_TO_QUESTIONS = "Not specified"


def load_fields() -> List[dict]:
    return json.loads(EXPECTED_FILE.read_text(encoding="utf-8"))["fields"]


def load_profile() -> CandidateProfile:
    return CandidateProfile(**json.loads(PROFILE_FILE.read_text(encoding="utf-8")))


def read_state(page, fields: List[dict]) -> Dict[str, object]:
    """Returns each field's value: text for text-like fields, bool for checkboxes and radio groups."""
    specs = [{"id": f["id"], "kind": f["kind"], "name": f.get("name")} for f in fields]
    rows = page.evaluate(
        """(specs) => specs.map((s) => {
            if (s.kind === 'radio') {
                return { id: s.id, value: !!document.querySelector(`input[name="${s.name}"]:checked`) };
            }
            const el = document.getElementById(s.id);
            return { id: s.id, value: s.kind === 'checkbox' ? el.checked : el.value };
        })""",
        specs,
    )
    return {row["id"]: row["value"] for row in rows}


def is_correct(spec: dict, value, sentinel: str = ANSWER_TO_QUESTIONS) -> bool:
    if spec.get("ask"):
        if spec["kind"] in ("checkbox", "radio"):
            return value is False  # left unticked
        text = str(value).strip()
        return text == "" or text == sentinel  # empty, or the candidate's own answer
    text = str(value)
    if spec.get("contains"):
        return all(part.lower() in text.lower() for part in spec["contains"])
    return text.strip() == spec["expect"]


@dataclass
class Report:
    answerable: int = 0
    answerable_correct: int = 0
    asked: int = 0
    asked_correct: int = 0
    missed: List[str] = field(default_factory=list)     # answerable, left empty
    wrong: List[str] = field(default_factory=list)      # answerable, filled with something else
    invented: List[str] = field(default_factory=list)   # not in the CV, but filled

    @property
    def fill_rate(self) -> float:
        return self.answerable_correct / self.answerable if self.answerable else 0.0

    def summary(self) -> str:
        lines = [
            f"answerable fields correct: {self.answerable_correct}/{self.answerable} ({self.fill_rate:.0%})",
            f"not-in-CV fields handled (empty or asked): {self.asked_correct}/{self.asked}",
            f"invented values in not-in-CV fields: {len(self.invented)}",
            f"missed answerable fields: {', '.join(self.missed) or 'none'}",
            f"wrong answerable values: {', '.join(self.wrong) or 'none'}",
            f"invented: {', '.join(self.invented) or 'none'}",
        ]
        return "\n".join(lines)


def score(fields: List[dict], state: Dict[str, object], sentinel: str = ANSWER_TO_QUESTIONS) -> Report:
    report = Report()
    for spec in fields:
        value = state[spec["id"]]
        ok = is_correct(spec, value, sentinel)
        if spec.get("ask"):
            report.asked += 1
            report.asked_correct += ok
            if not ok:
                report.invented.append(spec["id"])
        else:
            report.answerable += 1
            report.answerable_correct += ok
            if not ok:
                empty = str(value).strip() == ""
                (report.missed if empty else report.wrong).append(spec["id"])
    return report
