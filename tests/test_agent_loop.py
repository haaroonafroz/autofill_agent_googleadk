"""
Offline tests for the agent loop. A scripted fake model chooses the tool calls and
a fake extension answers the browser requests, so no API calls or browser are needed.
"""

import asyncio
import copy
import uuid
from typing import List

from langchain_core.messages import AIMessage

from autofill_agent.agent_loop import AgentDeps, MAX_FINISH_REFUSALS, MAX_STEPS, run_agent, verify_ledger
from autofill_agent.bridge import BridgeRegistry, Session
from autofill_agent.executor import Page
from autofill_agent.profile import CandidateProfile
from autofill_agent.prompts import build_user_message, compact_field
from autofill_agent.storage import Storage

ELEMENTS = [
    {"ref": "e1", "tag": "input", "input_type": "text", "label": "First name", "value": "", "required": True},
    {"ref": "e2", "tag": "input", "input_type": "email", "label": "Email", "value": ""},
    {"ref": "e3", "tag": "select", "input_type": "select", "label": "Country", "value": "",
     "options": [{"value": "", "text": "Choose...", "selected": True},
                 {"value": "DE", "text": "Germany", "selected": False}]},
    {"ref": "e4", "tag": "input", "input_type": "checkbox", "label": "I agree to the privacy policy", "checked": False},
    {"ref": "e9", "tag": "input", "input_type": "submit", "label": "Submit application", "text": "Submit application"},
]


class FakeExtension:
    """Stands in for the side panel and the service worker."""

    def __init__(self, elements, apply_changes=True, answers=None):
        self.elements = copy.deepcopy(elements)
        self.apply_changes = apply_changes
        self.answers = answers or {}
        self.session = None
        self.requests: List[dict] = []

    async def send_json(self, message):
        if message["type"] != "request":
            return  # agent events are checked through the emit callback instead
        self.requests.append(message)
        command, payload = message["command"], message["payload"]
        if command == "snapshot":
            data = self._snapshot()
        elif command == "act":
            data = self._act(payload["action"])
        elif command == "ask_human":
            data = self.answers.get(payload["question"], "")
        else:
            raise AssertionError(f"unexpected command {command}")
        self.session.resolve({"type": "response", "request_id": message["request_id"],
                              "ok": True, "data": data, "error": None})

    def _snapshot(self):
        return {
            "url": "http://test/form", "title": "Application", "text": "Apply now", "alerts": [],
            "frames": [{"frame_id": 0, "url": "http://test/form", "elements": copy.deepcopy(self.elements)}],
        }

    def _find(self, ref):
        return next((e for e in self.elements if e["ref"] == ref), None)

    def _act(self, action):
        el = self._find(action.get("ref"))
        if el is None and action["type"] != "scroll":
            return {"ok": False, "error": f"unknown ref '{action.get('ref')}'"}
        if action["type"] == "fill":
            if self.apply_changes:
                el["value"] = action["value"]
            return {"ok": True, "value_after": action["value"]}
        if action["type"] == "check":
            if self.apply_changes:
                el["checked"] = action["checked"]
            return {"ok": True, "value_after": action["checked"]}
        if action["type"] == "click":
            return {"ok": True}
        return {"ok": False, "error": f"fake does not support {action['type']}"}


class FakeLLM:
    """Returns one scripted turn of tool calls per decision."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen = []  # the messages sent on each decision

    def bind_tools(self, tools, **kwargs):
        return self

    async def ainvoke(self, messages):
        self.seen.append(messages)
        turn = self.turns.pop(0) if self.turns else [("finish", {"reached_review": False, "summary": "script ended"})]
        return AIMessage(content="", tool_calls=[
            {"name": name, "args": args, "id": f"call_{i}"} for i, (name, args) in enumerate(turn)
        ])


def run(turns, *, elements=ELEMENTS, apply_changes=True, answers=None, max_steps=MAX_STEPS,
        should_stop=lambda: False, cv_hits=None, cv_text=""):
    user = f"test-{uuid.uuid4()}"
    storage = Storage()
    storage.save_profile(user, CandidateProfile(full_name="Ada Lovelace", email="ada@example.com",
                                                skills=["Python"]), source=None)

    ext = FakeExtension(elements, apply_changes=apply_changes, answers=answers)
    bridge = BridgeRegistry()
    session = Session(websocket=ext)
    session.user_id = user
    ext.session = session
    bridge.attach(session)

    llm = FakeLLM(turns)
    emitted = []

    async def emit(kind, data):
        emitted.append((kind, data))

    deps = AgentDeps(
        llm=llm, bridge=bridge, storage=storage, user_id=user, profile=storage.get_profile(user),
        emit=emit, should_stop=should_stop,
        cv_search=lambda q, kind=None: cv_hits if cv_hits is not None else ["Built payment APIs in Python"],
        max_steps=max_steps, cv_text=cv_text,
    )
    result = asyncio.run(run_agent(deps, "Fill the application"))
    return {"result": result, "ext": ext, "llm": llm, "storage": storage, "user": user, "emitted": emitted}


def test_happy_path_fills_verifies_and_finishes():
    out = run([
        [("fill_field", {"ref": "e1", "value": "Ada"}), ("fill_field", {"ref": "e2", "value": "ada@example.com"})],
        [("finish", {"reached_review": True, "summary": "all fields filled"})],
    ])
    result = out["result"]
    assert result["outcome"]["status"] == "done"
    assert result["steps"] == 2
    statuses = {f["label"]: f["status"] for f in result["filled"]}
    assert statuses == {"First name": "verified", "Email": "verified"}
    assert out["ext"].elements[0]["value"] == "Ada"


def test_submit_click_is_blocked_and_question_goes_to_candidate():
    out = run(
        [
            [("click_element", {"ref": "e9"})],
            [("ask_human", {"question": "What is your notice period?"})],
            [("finish", {"reached_review": False, "summary": "waiting for the candidate"})],
        ],
        answers={"What is your notice period?": "Two weeks"},
    )
    assert out["result"]["outcome"]["status"] == "stopped"
    # The submit click never reached the browser.
    clicks = [r for r in out["ext"].requests if r["command"] == "act" and r["payload"]["action"]["type"] == "click"]
    assert clicks == []
    # The answer was saved and shown to the model on the next decision.
    assert out["storage"].get_answer(out["user"], "What is your notice period?") == "Two weeks"
    last_prompt = out["llm"].seen[2][1].content
    assert "The candidate answered 'What is your notice period?': Two weeks" in last_prompt
    assert ("ask", {"question": "What is your notice period?", "ref": None}) in out["emitted"]


def test_changed_value_is_detected_and_reported_to_the_model():
    out = run(
        [
            [("fill_field", {"ref": "e1", "value": "Ada"})],
            [("finish", {"reached_review": False, "summary": "stuck"})],
        ],
        apply_changes=False,  # the page ignores the fill
    )
    assert [f["status"] for f in out["result"]["filled"]] == ["mismatch"]
    assert "Field 'First name' shows ''" in out["llm"].seen[1][1].content


def test_no_progress_stops_the_run():
    out = run([[("search_cv", {"query": "projects"})]] * 10)
    assert out["result"]["outcome"]["status"] == "stopped"
    assert "no progress" in out["result"]["outcome"]["reason"]
    assert out["result"]["steps"] == 3


def test_step_budget_stops_the_run():
    out = run(
        [[("fill_field", {"ref": "e1", "value": "Ada"})], [("fill_field", {"ref": "e2", "value": "a@b.c"})]],
        max_steps=2,
    )
    assert out["result"]["outcome"]["status"] == "stopped"
    assert "limit" in out["result"]["outcome"]["reason"]
    assert out["result"]["steps"] == 2


def test_candidate_stop_request_ends_before_any_action():
    out = run([[("fill_field", {"ref": "e1", "value": "Ada"})]], should_stop=lambda: True)
    assert out["result"]["outcome"]["status"] == "stopped"
    assert out["result"]["steps"] == 0
    assert out["ext"].requests == []


def test_unknown_ref_is_reported_and_the_loop_continues():
    out = run([
        [("fill_field", {"ref": "e42", "value": "x"})],
        [("finish", {"reached_review": True, "summary": "done"})],
    ])
    assert out["result"]["outcome"]["status"] == "done"
    second_prompt = out["llm"].seen[1][1].content
    assert "fill_field: FAILED (unknown ref 'e42'" in second_prompt


def test_empty_fill_is_refused_and_never_sent_to_the_page():
    out = run([
        [("fill_field", {"ref": "e4", "value": "   "})],  # the Live search box, with no real answer
        [("finish", {"reached_review": False, "summary": "nothing to fill"})],
    ])
    fills = [r for r in out["ext"].requests if r["command"] == "act" and r["payload"]["action"]["type"] == "fill"]
    assert fills == []
    assert "fill_field: FAILED (empty value" in out["llm"].seen[1][1].content


def test_empty_fill_is_allowed_to_clear_a_value_this_run_filled():
    out = run([
        [("fill_field", {"ref": "e1", "value": "Wrong"})],
        [("fill_field", {"ref": "e1", "value": ""})],  # correcting its own mistake
        [("finish", {"reached_review": False, "summary": "cleared"})],
    ])
    fills = [r["payload"]["action"] for r in out["ext"].requests
             if r["command"] == "act" and r["payload"]["action"]["type"] == "fill"]
    assert [f["value"] for f in fills] == ["Wrong", ""]
    assert out["ext"].elements[0]["value"] == ""


def test_early_finish_is_refused_while_an_empty_field_was_not_addressed():
    out = run([
        [("fill_field", {"ref": "e1", "value": "Ada"}), ("fill_field", {"ref": "e2", "value": "ada@example.com"}),
         ("finish", {"reached_review": False, "summary": "done"})],
        [("ask_human", {"question": "Which country?", "ref": "e3"}),
         ("finish", {"reached_review": False, "summary": "done"})],
    ], answers={"Which country?": "Germany"})
    assert out["result"]["outcome"]["status"] == "stopped"
    assert out["result"]["steps"] == 2
    assert "finish refused: these fields are still empty and were neither filled nor asked about: Country" \
        in out["llm"].seen[1][1].content


def test_asking_about_an_empty_field_counts_as_addressing_it():
    out = run(
        [
            [("fill_field", {"ref": "e1", "value": "Ada"}),
             ("fill_field", {"ref": "e2", "value": "ada@example.com"}),
             ("ask_human", {"question": "Which country?", "ref": "e3"})],
            [("finish", {"reached_review": False, "summary": "waiting for the candidate"})],
        ],
        answers={"Which country?": "Germany"},
    )
    assert out["result"]["steps"] == 2
    assert out["result"]["outcome"]["status"] == "stopped"


def test_finish_refusals_stop_a_run_that_makes_no_progress():
    out = run([[("finish", {"reached_review": False, "summary": "stop"})]] * 7)
    assert out["result"]["outcome"]["status"] == "stopped"
    assert out["result"]["outcome"]["reason"] == "no progress for 3 turns"


def test_finish_refusals_are_capped_so_the_run_cannot_loop():
    # Each refused finish is followed by one fill, so the no-progress guard never fires.
    # After MAX_FINISH_REFUSALS refusals the next finish is accepted, even with fields left empty.
    elements = [{"ref": f"e{i}", "tag": "input", "input_type": "text", "label": f"Field {i}", "value": ""}
                for i in range(1, 7)]
    finish = ("finish", {"reached_review": False, "summary": "stop"})
    turns = []
    for i in range(1, MAX_FINISH_REFUSALS + 1):
        turns += [[finish], [("fill_field", {"ref": f"e{i}", "value": f"value {i}"})]]
    turns.append([finish])
    out = run(turns, elements=elements)
    assert out["result"]["outcome"]["status"] == "stopped"
    assert out["result"]["steps"] == len(turns)
    assert out["result"]["outcome"]["reason"] == "stop"  # the final finish was accepted, not refused again


CV_WITH_LINKS = "Ada Lovelace\nBerlin, Germany\nGitHub: https://github.com/adalovelace/engine"


def test_url_not_in_the_cv_is_never_filled():
    elements = [{"ref": "e1", "tag": "input", "input_type": "url", "label": "GitHub URL", "value": ""}]
    out = run([
        [("fill_field", {"ref": "e1", "value": "https://github.com/adalovelace/analytics"})],
        [("finish", {"reached_review": False, "summary": "done"})],
    ], elements=elements, cv_text=CV_WITH_LINKS)
    assert not [r for r in out["ext"].requests if r["command"] == "act"]  # never reached the page
    assert "does not appear in the CV" in out["llm"].seen[1][1].content


def test_url_in_the_cv_is_filled():
    elements = [{"ref": "e1", "tag": "input", "input_type": "url", "label": "GitHub URL", "value": ""}]
    out = run([
        [("fill_field", {"ref": "e1", "value": "https://github.com/adalovelace/engine"})],
        [("finish", {"reached_review": True, "summary": "done"})],
    ], elements=elements, cv_text=CV_WITH_LINKS)
    assert "does not appear in the CV" not in out["llm"].seen[1][1].content


def test_location_not_in_the_cv_is_refused_but_a_grounded_one_is_filled():
    elements = [{"ref": "e1", "tag": "input", "input_type": "text", "label": "Location", "value": ""}]
    out = run([
        [("fill_field", {"ref": "e1", "value": "Weimar, Germany"})],
        [("finish", {"reached_review": False, "summary": "done"})],
    ], elements=elements, cv_text=CV_WITH_LINKS)
    assert "'weimar' does not appear in the CV" in out["llm"].seen[1][1].content


def test_grounding_check_ignores_scheme_case_and_trailing_slash():
    from autofill_agent.agent_loop import _not_in_cv, _plain
    cv = _plain(CV_WITH_LINKS)
    assert _not_in_cv("GitHub URL", "https://GitHub.com/adalovelace/engine/", cv) is None
    assert _not_in_cv("GitHub URL", "github.com/adalovelace/engine", cv) is None
    assert _not_in_cv("GitHub URL", "https://github.com/adalovelace/other", cv) is not None
    assert _not_in_cv("Location", "Berlin, Germany", cv) is None
    assert _not_in_cv("Location", "Berlin, France", cv) == "france"
    assert _not_in_cv("Full name", "Anybody", cv) is None  # only links and locations are checked
    assert _not_in_cv("Location", "Berlin", "") is None    # no CV text, nothing to check against


def test_search_cv_result_is_passed_to_the_next_decision():
    out = run([
        [("search_cv", {"query": "payment"})],
        [("finish", {"reached_review": False, "summary": "no more"})],
    ], cv_hits=["Built payment APIs in Python"])
    assert "Built payment APIs in Python" in out["llm"].seen[1][1].content


def test_verify_ledger_handles_select_and_checkbox():
    page = Page({"frames": [{"frame_id": 0, "elements": [
        {"ref": "e3", "label": "Country", "input_type": "select", "value": "DE",
         "options": [{"value": "DE", "text": "Germany", "selected": True}]},
        {"ref": "e4", "label": "I agree", "input_type": "checkbox", "checked": True},
    ]}]})
    ledger = {
        "select|Country": {"label": "Country", "input_type": "select", "kind": "select",
                           "intended": "Germany", "status": "filled"},
        "checkbox|I agree": {"label": "I agree", "input_type": "checkbox", "kind": "check",
                             "intended": True, "status": "filled"},
    }
    updated, issues = verify_ledger(ledger, page)
    assert updated["select|Country"]["status"] == "verified"
    assert updated["checkbox|I agree"]["status"] == "verified"
    assert issues == []


def test_long_text_values_are_verified_in_full_but_shortened_in_the_prompt():
    long_text = "word " * 150  # about 750 characters
    page = Page({"frames": [{"frame_id": 0, "elements": [
        {"ref": "e5", "label": "Why", "input_type": "textarea", "value": long_text},
    ]}]})
    ledger = {"textarea|Why": {"label": "Why", "input_type": "textarea", "kind": "fill",
                               "intended": long_text, "status": "filled"}}
    updated, issues = verify_ledger(ledger, page)
    assert updated["textarea|Why"]["status"] == "verified"
    assert issues == []

    line = compact_field(page.elements[0])
    assert "..." in line and len(line) < 400


def test_prompt_shows_refs_and_hides_passwords():
    line = compact_field({"ref": "e7", "input_type": "password", "label": "Password", "value": "[redacted]"})
    assert "e7" in line and "[redacted]" in line
    page = Page({"title": "Form", "url": "http://t", "frames": [{"frame_id": 0, "elements": ELEMENTS}]})
    message = build_user_message(goal="g", step=1, max_steps=25, profile=None, answers=[], page=page,
                                 ledger={}, notes=[], last_results=[])
    assert '"Country"' in message and "Germany" in message
    assert "e9" in message
