"""
Tools the model can call. They are the only way the agent acts.

Browser tools are translated into the action format in actions.py and then go
through the same executor and policy as the dev tools. The model never gets
raw browser access.
"""

from typing import Literal, Optional

from langchain_core.tools import tool


@tool
def fill_field(ref: str, value: str) -> str:
    """Type a value into a text, email, number, date or textarea field, using its ref from the Fields list."""
    return "queued"


@tool
def select_option(ref: str, label: Optional[str] = None, value: Optional[str] = None) -> str:
    """Choose an option in a dropdown. Pass the option's visible text as `label` (preferred), or its `value`."""
    return "queued"


@tool
def set_checkbox(ref: str, checked: bool) -> str:
    """Tick (checked=true) or untick (checked=false) a checkbox or radio button."""
    return "queued"


@tool
def click_element(ref: str) -> str:
    """Click a button, link, radio or custom control. Never use it on submit, apply, send or confirm buttons.
    It ends this turn, because the page may change."""
    return "queued"


@tool
def type_text(text: str, ref: Optional[str] = None) -> str:
    """Type keystrokes, after clicking `ref` first if given. Use for custom widgets that ignore fill_field."""
    return "queued"


@tool
def scroll_page(dy: int) -> str:
    """Scroll the page vertically. Positive dy scrolls down."""
    return "queued"


@tool
def search_cv(query: str, kind: Optional[Literal["experience", "education", "project"]] = None) -> str:
    """
    Search the candidate's CV for facts the profile does not have, such as project details or motivations.
    To list every job, degree or project, pass kind="experience", "education" or "project" with any query:
    you get all of them, in a fixed order (jobs and degrees newest first, projects in CV order).
    """
    return "queued"


@tool
def ask_human(question: str, ref: Optional[str] = None) -> str:
    """Ask the candidate a question that the CV and saved answers cannot answer. Never guess such answers."""
    return "queued"


@tool
def finish(reached_review: bool, summary: str) -> str:
    """Stop the run. Use reached_review=true when every field is filled and the next step is the review or submit page.
    Use reached_review=false when you are stuck, or a login or account is needed."""
    return "queued"


ALL_TOOLS = [
    fill_field, select_option, set_checkbox, click_element, type_text,
    scroll_page, search_cv, ask_human, finish,
]

BROWSER_TOOL_NAMES = {"fill_field", "select_option", "set_checkbox", "click_element", "type_text", "scroll_page"}


def browser_action_dict(name: str, args: dict) -> dict:
    """Translates a browser tool call into the action format used by actions.py."""
    if name == "fill_field":
        return {"type": "fill", "ref": args["ref"], "value": args["value"]}
    if name == "select_option":
        return {"type": "select", "ref": args["ref"], "label": args.get("label"), "value": args.get("value")}
    if name == "set_checkbox":
        return {"type": "check", "ref": args["ref"], "checked": bool(args["checked"])}
    if name == "click_element":
        return {"type": "click", "ref": args["ref"]}
    if name == "type_text":
        action = {"type": "type", "text": args["text"]}
        if args.get("ref"):
            action["ref"] = args["ref"]
        return action
    if name == "scroll_page":
        return {"type": "scroll", "dy": int(args["dy"])}
    raise ValueError(f"not a browser tool: {name}")
