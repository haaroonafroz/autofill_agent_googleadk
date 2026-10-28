"""
Prompts for the agent. The model sees the candidate profile, saved answers,
the current page as a list of fields with refs, and the results of its last turn.
"""

from typing import Dict, List, Optional

from .executor import Page
from .profile import CandidateProfile

MAX_FIELDS_IN_PROMPT = 150
MAX_PAGE_TEXT = 1500
MAX_VALUE_IN_PROMPT = 200

SYSTEM_PROMPT = """You fill out a job application in the candidate's browser. You see the page as a list of fields, each with a ref, and you act only by calling tools.

Rules:
1. Use only refs from the Fields list. Refs change when the page changes, so never reuse an old ref after a click.
2. Only enter information that the candidate profile, the saved answers or search_cv results support. Never invent facts.
3. If a required question has no answer in those sources, call ask_human with a clear question. Do not guess. Examples: salary expectations, notice period, availability, visa sponsorship if the CV does not say, diversity or EEO questions.
4. Legal consent checkboxes (privacy policy, terms, data consent) are the candidate's decision. Ask with ask_human, and do not tick them yourself.
5. Passwords, logins and account creation are out of scope. Never ask the candidate for a password, and never fill one. Call finish with reached_review=false and say what is needed.
6. Never click submit, apply, send or confirm buttons. The candidate submits. Clicking Next or Continue is fine once this page's required fields are done.
7. For dropdowns, pass the option's visible text as `label`. For name fields, split the name: "First name" gets only the given name, "Last name" only the family name.
8. Read the "Already filled" and "Results of your last actions" sections. Fix any value that shows a mismatch, and retry or change approach after a failure.
9. Fill several fields in one turn. Put click_element last in a turn.
10. When every field is filled and the next step is review or submit, call finish with reached_review=true.
11. If a browser action fails the same way twice, stop retrying it and move on to the next field. Do not repeat yourself.
12. Use saved answers. If a saved answer fits a field, enter it into that field. Never ask a question that is already in the saved answers or in the notes, not even in other words. A question the candidate answered "Not specified" stays unanswered: leave that field empty and move on.
13. A field with no usable answer never ends the run. Fill every other field first, and only then call finish with reached_review=false. Name the fields that are still empty in the summary.
14. For a repeated section such as "Job 2" or "Project 3", use the candidate's entries in the order given by the profile or by search_cv with kind. Count the entries and fill each section with the entry at the same position. Do not mix entries from different positions.
Always call exactly one or more tools. Never reply with plain text only."""


def _quote(text: str) -> str:
    return '"' + text.replace('"', "'") + '"'


def compact_profile(profile: Optional[CandidateProfile]) -> str:
    if profile is None:
        return "(no profile stored; use search_cv and ask_human)"
    return profile.model_dump_json(exclude_none=True, exclude_defaults=True)


def compact_field(el: dict) -> str:
    parts: List[str] = [el["ref"], el.get("input_type") or el.get("tag", ""), _quote(el.get("label") or "")]
    if el.get("required"):
        parts.append("required")
    if el.get("disabled"):
        parts.append("disabled")
    input_type = el.get("input_type")
    if input_type in ("checkbox", "radio"):
        parts.append(f"checked: {str(bool(el.get('checked'))).lower()}")
    elif el.get("options"):
        texts = [_quote(o["text"]) for o in el["options"][:25] if o.get("text")]
        parts.append("options: " + ", ".join(texts))
        selected = next((o["text"] for o in el["options"] if o.get("selected")), None)
        if selected is not None:
            parts.append(f"selected: {_quote(selected)}")
    elif input_type not in ("button", "submit", "image", "reset", "a") and el.get("value") is not None:
        value = str(el["value"])
        if len(value) > MAX_VALUE_IN_PROMPT:
            value = value[:MAX_VALUE_IN_PROMPT] + "..."
        parts.append(f"value: {_quote(value)}")
    if el.get("text") and input_type in (None, "button", "submit", "a"):
        parts.append(f"text: {_quote(el['text'])}")
    if el.get("error"):
        parts.append(f"error: {_quote(el['error'])}")
    return " | ".join(parts)


def build_user_message(
    *,
    goal: str,
    step: int,
    max_steps: int,
    profile: Optional[CandidateProfile],
    answers: List[tuple],
    page: Page,
    ledger: Dict[str, dict],
    notes: List[str],
    last_results: List[dict],
) -> str:
    lines: List[str] = []
    lines.append(f"Goal: {goal}")
    lines.append(f"Turn {step} of {max_steps}.")
    lines.append("")
    lines.append("Candidate profile (from CV):")
    lines.append(compact_profile(profile))
    lines.append("")
    lines.append("Saved answers:")
    if answers:
        lines.extend(f"- Q: {q} A: {a}" for q, a in answers)
    else:
        lines.append("- (none yet)")
    lines.append("")
    lines.append(f"Current page: {page.title} ({page.url})")
    if page.alerts:
        lines.append("Alerts on the page: " + " | ".join(page.alerts))
    lines.append(f"Visible text (truncated): {page.text[:MAX_PAGE_TEXT]}")
    lines.append("")

    elements = page.elements[:MAX_FIELDS_IN_PROMPT]
    lines.append(f"Fields (use these refs only; {len(page.elements)} visible):")
    lines.extend(compact_field(el) for el in elements)
    lines.append("")

    lines.append("Already filled by you:")
    if ledger:
        for entry in ledger.values():
            status = entry.get("status", "")
            extra = f" (page shows {_quote(str(entry['observed']))})" if status == "mismatch" else ""
            lines.append(f"- {entry['label']}: {_quote(str(entry['intended']))} [{status}]{extra}")
    else:
        lines.append("- (nothing yet)")
    lines.append("")

    lines.append("Results of your last actions:")
    if last_results:
        for r in last_results:
            tool = r.get("tool") or "turn"
            if r.get("ok"):
                lines.append(f"- {tool}: ok")
            elif r.get("blocked"):
                lines.append(f"- {tool}: BLOCKED ({r.get('error')})")
            else:
                lines.append(f"- {tool}: FAILED ({r.get('error')})")
    else:
        lines.append("- (none)")
    lines.append("")

    if notes:
        lines.append("Notes:")
        lines.extend(f"- {n}" for n in notes)
        lines.append("")

    lines.append("Choose your next tool calls.")
    return "\n".join(lines)
