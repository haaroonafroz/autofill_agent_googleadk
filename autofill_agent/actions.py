"""
Browser actions the agent can request, plus the safety policy that gates them.

Every action refers to an element by a `ref` (e.g. "e12") that came from the
page snapshot. Actions are validated here before they reach the browser.

Two execution modes exist:
- "dom":  set values and fire events directly on the element (precise, cheap).
- "real": send trusted mouse/keyboard input through Chrome DevTools Protocol
          (works with custom widgets and frameworks that ignore synthetic events).
"""

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter


class Fill(BaseModel):
    type: Literal["fill"]
    ref: str
    value: str


class Select(BaseModel):
    type: Literal["select"]
    ref: str
    value: Optional[str] = None   # the option's value attribute
    label: Optional[str] = None   # the option's visible text

    def model_post_init(self, __context):
        if self.value is None and self.label is None:
            raise ValueError("select needs 'value' or 'label'")


class Check(BaseModel):
    type: Literal["check"]
    ref: str
    checked: bool = True


class Click(BaseModel):
    type: Literal["click"]
    ref: str
    mode: Literal["real", "dom"] = "real"


class Type(BaseModel):
    type: Literal["type"]
    text: str
    ref: Optional[str] = Field(None, description="Element to click (real) before typing.")


class Key(BaseModel):
    type: Literal["key"]
    key: str = Field(description="e.g. Enter, Tab, Escape, ArrowDown")


class Scroll(BaseModel):
    type: Literal["scroll"]
    dy: int = Field(description="Pixels to scroll; positive is down.")


class Wait(BaseModel):
    type: Literal["wait"]
    ms: int = Field(ge=0, le=10_000)


Action = Annotated[
    Union[Fill, Select, Check, Click, Type, Key, Scroll, Wait],
    Field(discriminator="type"),
]
_action_adapter = TypeAdapter(Action)

MAX_ACTIONS_PER_BATCH = 20


def parse_actions(raw: list) -> list[BaseModel]:
    if len(raw) > MAX_ACTIONS_PER_BATCH:
        raise ValueError(f"at most {MAX_ACTIONS_PER_BATCH} actions per batch")
    return [_action_adapter.validate_python(item) for item in raw]


# --- Policy ------------------------------------------------------------------
# Hard rules that do not depend on the model. Anything blocked here must go to a
# human (approval gates are added in phase 5). The extension repeats the submit
# check as a second line of defence.

SUBMIT_WORDS = ("submit", "apply", "send application", "finish", "confirm", "pay", "place order")


def is_submit_like(label: str, input_type: str = "") -> bool:
    text = f"{label} {input_type}".lower()
    return any(word in text for word in SUBMIT_WORDS)


def check_policy(action: BaseModel, element: Optional[dict] = None) -> Optional[str]:
    """
    Returns a reason string if the action must not run, otherwise None.
    `element` is the snapshot entry for the action's ref, when the action has one.
    """
    element = element or {}
    kind = element.get("input_type", "")

    if isinstance(action, Click):
        if is_submit_like(element.get("label", "") or element.get("text", ""), kind):
            return "clicking a submit-like control needs human approval"
        if kind == "file":
            return "file pickers need human approval"

    if isinstance(action, (Fill, Type)) and kind == "password":
        return "passwords are never typed by the agent"
    if isinstance(action, Fill) and kind == "file":
        return "file uploads need human approval"

    # Enter can submit a form, so it is treated as a submit (human approval).
    if isinstance(action, Key) and action.key.lower() in ("enter", "return", "\r", "\n"):
        return "pressing Enter can submit the form; needs human approval"
    if isinstance(action, Type) and "\n" in action.text:
        return "newlines would press Enter and can submit the form; needs human approval"

    return None
