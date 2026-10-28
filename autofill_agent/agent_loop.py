"""
The agent loop, built with LangGraph.

    observe -> decide -> act -> observe -> ...

- observe: read the page, check every field the agent already set against what
  the page now shows (verification), and stop if a budget or limit is reached.
- decide: gpt-6-luna chooses tool calls from the page, the profile and the history.
- act: run browser actions through the executor, answer search_cv and ask_human.

The loop ends when the model calls finish, when the step budget runs out, when
it makes no progress for several turns, when the candidate stops it, or when the
browser stops answering.
"""

import html
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from .actions import parse_actions
from .agent_tools import ALL_TOOLS, BROWSER_TOOL_NAMES, browser_action_dict
from .bridge import BridgeRegistry, ExtensionDisconnected, ExtensionError
from .executor import Page, read_page, run_actions
from .profile import CandidateProfile
from .prompts import SYSTEM_PROMPT, build_user_message
from .storage import Storage

MAX_STEPS = 35               # long forms need more turns once the agent must finish every field
MAX_NO_PROGRESS = 3          # turns in a row without a successful action or answer
MAX_CALLS_PER_TURN = 8
ASK_TIMEOUT = 1800.0         # seconds to wait for the candidate to answer a question
RECURSION_LIMIT = 300        # LangGraph super-steps; well above MAX_STEPS * 3

Emit = Callable[[str, dict], Awaitable[None]]


@dataclass
class AgentDeps:
    llm: Any                              # any chat model with bind_tools() and ainvoke()
    bridge: BridgeRegistry
    storage: Storage
    user_id: str
    profile: Optional[CandidateProfile]
    emit: Emit
    should_stop: Callable[[], bool]
    cv_search: Callable[[str, Optional[str]], List[str]]  # (query, kind) -> text of each hit
    max_steps: int = MAX_STEPS
    cv_text: str = ""  # the indexed CV; when set, URLs and locations must appear in it to be filled


class AgentState(TypedDict, total=False):
    goal: str
    asked_keys: List[str]      # fields the candidate was asked about, by field key
    finish_refusals: int       # how many early finish calls were refused
    step: int
    page: Optional[Page]
    ledger: Dict[str, dict]       # what the agent has set, keyed by "input_type|label"
    notes: List[str]              # extra context for the next decision
    last_results: List[dict]      # results of the last turn's actions
    pending_calls: List[dict]     # tool calls chosen by the model, not yet run
    no_progress: int
    outcome: Optional[dict]       # {"status": "done" | "stopped" | "failed", "reason": str}


def _key(label: str, input_type: str) -> str:
    return f"{input_type}|{label}"


# Fields that need a text, choice or question answer. Checkboxes and radios are excluded because
# leaving them unticked is the safe default, and consent is handled by ask_human.
FILLABLE_TYPES = {"text", "email", "tel", "url", "number", "search", "date", "textarea", "select"}
MAX_FINISH_REFUSALS = 4

LOCATION_LABEL = re.compile(r"location|city|address|country|town|postcode|postal|zip", re.I)
URL_VALUE = re.compile(r"^(https?://|www\.)|^[\w-]+(\.[\w-]+)*\.(com|org|net|io|dev|app|co|ai)(/|$)", re.I)


def _plain(text: str) -> str:
    """Lowercase, scheme and www stripped, whitespace collapsed. Used to compare values with the CV."""
    text = html.unescape(text or "").lower()
    text = re.sub(r"https?://", "", text)
    text = re.sub(r"(^|[\s/])www\.", r"\1", text)
    return re.sub(r"\s+", " ", text).strip().rstrip("/")


def _not_in_cv(label: str, value: str, cv_plain: str) -> Optional[str]:
    """
    Why a value must not be filled, or None. Links and locations are the values the model
    most often makes up (a GitHub URL built from a name, a city it guesses), so they are only
    filled when the CV says them. Returns the offending part, for the error message.
    """
    if not value.strip() or not cv_plain:
        return None
    if URL_VALUE.search(value.strip()):
        url = _plain(value)
        return value if url and url not in cv_plain else None
    if LOCATION_LABEL.search(label or ""):
        for part in (p.strip() for p in _plain(value).split(",")):
            if part and part not in cv_plain:
                return part
    return None


def _unaddressed(page: Page, ledger: Dict[str, dict], asked_keys: set) -> List[str]:
    """Empty fields that were neither filled by this run nor asked about. Labels, in page order."""
    missing: List[str] = []
    for el in page.elements:
        if el.get("input_type") not in FILLABLE_TYPES or el.get("disabled"):
            continue
        if str(el.get("value") or "").strip():
            continue
        key = _key(el.get("label") or "", el.get("input_type") or "")
        if key in ledger or key in asked_keys:
            continue
        missing.append(el.get("label") or el["ref"])
    return missing


def _find_element(page: Page, label: str, input_type: str) -> Optional[dict]:
    for el in page.elements:
        if el.get("label") == label and el.get("input_type") == input_type:
            return el
    return None


def _compare(entry: dict, el: dict) -> tuple:
    """Returns (observed value, whether it matches what the agent set)."""
    kind = entry["kind"]
    if kind == "check":
        observed = bool(el.get("checked"))
        return observed, observed == bool(entry["intended"])
    if kind == "select":
        selected = next((o["text"] for o in el.get("options", []) if o.get("selected")), "")
        return selected, selected.strip().lower() == str(entry["intended"]).strip().lower()
    observed = el.get("value") or ""
    return observed, observed.strip() == str(entry["intended"]).strip()


def verify_ledger(ledger: Dict[str, dict], page: Page) -> tuple:
    """Re-checks every field the agent set against the current page. Returns (ledger, issues)."""
    issues: List[str] = []
    updated: Dict[str, dict] = {}
    for key, entry in ledger.items():
        entry = dict(entry)
        if entry["status"] in ("filled", "verified", "mismatch"):
            el = _find_element(page, entry["label"], entry["input_type"])
            if el is not None:
                observed, ok = _compare(entry, el)
                entry["observed"] = observed
                entry["status"] = "verified" if ok else "mismatch"
                if not ok:
                    issues.append(
                        f"Field '{entry['label']}' shows '{observed}' but was set to "
                        f"'{entry['intended']}'. Fix it or report it."
                    )
        updated[key] = entry
    return updated, issues


def _ledger_meta(page: Page, action: dict, step: int) -> Optional[dict]:
    """Describes a fill, select or check so the verifier can check it on the next observe."""
    if action["type"] not in ("fill", "select", "check"):
        return None
    el = page.by_ref.get(action["ref"])
    if el is None:
        return None
    kind = {"fill": "fill", "select": "select", "check": "check"}[action["type"]]
    return {
        "key": _key(el.get("label") or "", el.get("input_type") or ""),
        "label": el.get("label") or "",
        "input_type": el.get("input_type") or "",
        "kind": kind,
        "step": step,
    }


def _failed(reason: str) -> dict:
    return {"status": "failed", "reason": reason}


def _stopped(reason: str) -> dict:
    return {"status": "stopped", "reason": reason}


def build_agent_graph(deps: AgentDeps):
    llm_with_tools = deps.llm.bind_tools(ALL_TOOLS, tool_choice="required")

    async def observe(state: AgentState) -> dict:
        if deps.should_stop():
            return {"outcome": _stopped("stopped by the candidate")}
        try:
            page = await read_page(deps.bridge)
        except (ExtensionError, ExtensionDisconnected) as exc:
            return {"outcome": _failed(f"could not read the page: {exc}")}

        ledger, issues = verify_ledger(state.get("ledger", {}), page)
        if issues:
            await deps.emit("verify", {"issues": issues})

        step = state.get("step", 0)
        outcome = None
        if step >= deps.max_steps:
            outcome = _stopped(f"reached the limit of {deps.max_steps} turns")
        elif state.get("no_progress", 0) >= MAX_NO_PROGRESS:
            outcome = _stopped(f"no progress for {MAX_NO_PROGRESS} turns")

        await deps.emit("observe", {
            "url": page.url,
            "title": page.title,
            "fields": len(page.elements),
            "step": step,
        })
        return {
            "page": page,
            "ledger": ledger,
            "notes": list(state.get("notes", [])) + issues,
            "outcome": outcome,
        }

    async def decide(state: AgentState) -> dict:
        step = state.get("step", 0) + 1
        page = state["page"]
        message = build_user_message(
            goal=state["goal"],
            step=step,
            max_steps=deps.max_steps,
            profile=deps.profile,
            answers=deps.storage.list_answers(deps.user_id),
            page=page,
            ledger=state.get("ledger", {}),
            notes=state.get("notes", []),
            last_results=state.get("last_results", []),
        )
        try:
            reply = await llm_with_tools.ainvoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=message)])
        except Exception as exc:  # network or API error: end the run cleanly
            return {"step": step, "pending_calls": [], "outcome": _failed(f"model call failed: {exc}")}

        calls = list(reply.tool_calls or [])
        await deps.emit("decide", {
            "step": step,
            "text": (reply.text or "")[:500],
            "calls": [{"name": c["name"], "args": c.get("args", {})} for c in calls],
        })
        return {"step": step, "pending_calls": calls, "notes": [], "last_results": []}

    async def act(state: AgentState) -> dict:
        try:
            return await _act(state)
        except (ExtensionError, ExtensionDisconnected) as exc:
            return {"pending_calls": [], "outcome": _failed(f"browser connection problem: {exc}")}

    async def _act(state: AgentState) -> dict:
        calls = (state.get("pending_calls") or [])[:MAX_CALLS_PER_TURN]
        page: Page = state["page"]
        step = state.get("step", 0)
        ledger = dict(state.get("ledger", {}))
        results: List[dict] = []
        notes: List[str] = []
        batch: List[tuple] = []   # (tool name, action dict, ledger meta)
        outcome: Optional[dict] = None
        progressed = False
        asked_keys = set(state.get("asked_keys", []))
        refusals = state.get("finish_refusals", 0)
        cv_plain = _plain(deps.cv_text)

        async def flush() -> None:
            nonlocal progressed
            if not batch:
                return
            run = await run_actions(deps.bridge, [a for _, a, _ in batch], page=page)
            for i, (tool, action, meta) in enumerate(batch):
                if i >= len(run["results"]):
                    results.append({"tool": tool, "ok": False, "error": "not run: an earlier action failed"})
                    continue
                res = run["results"][i]
                ok = bool(res.get("ok"))
                results.append({
                    "tool": tool,
                    "ok": ok,
                    "error": res.get("error"),
                    "blocked": bool(res.get("blocked")),
                })
                if ok:
                    progressed = True
                    if meta:
                        intended = res.get("value_after")
                        if meta["kind"] == "fill":
                            intended = action["value"]
                        elif meta["kind"] == "check":
                            intended = bool(action["checked"])
                        ledger[meta["key"]] = {
                            "label": meta["label"],
                            "input_type": meta["input_type"],
                            "kind": meta["kind"],
                            "intended": intended,
                            "status": "filled",
                            "step": meta["step"],
                        }
            batch.clear()

        if not calls:
            results.append({"tool": None, "ok": False, "error": "you must call a tool"})

        for call in calls:
            name = call["name"]
            args = call.get("args") or {}

            if name == "finish":
                await flush()
                # Stopping early is the main failure on long forms, so an unfinished run must
                # account for every empty field before it can stop.
                missing = [] if args.get("reached_review") else _unaddressed(page, ledger, asked_keys)
                if missing and refusals < MAX_FINISH_REFUSALS:
                    refusals += 1
                    results.append({
                        "tool": name,
                        "ok": False,
                        "error": "finish refused: these fields are still empty and were neither filled nor "
                                 "asked about: " + "; ".join(missing[:25])
                                 + ". Fill them, or ask_human for the ones you cannot answer.",
                    })
                    continue
                outcome = {
                    "status": "done" if args.get("reached_review") else "stopped",
                    "reason": args.get("summary", ""),
                }
                break

            if name == "search_cv":
                kind = args.get("kind")
                hits = deps.cv_search(args.get("query", ""), kind)
                body = "\n---\n".join(hits) if hits else "no matching CV text"
                label = f"{args.get('query', '')}" + (f", kind={kind}" if kind else "")
                notes.append(f"CV search for '{label}':\n{body}")
                results.append({"tool": name, "ok": True})
                continue

            if name == "ask_human":
                await flush()
                question = args.get("question", "")
                await deps.emit("ask", {"question": question, "ref": args.get("ref")})
                try:
                    answer = await deps.bridge.request(
                        "ask_human", {"question": question, "ref": args.get("ref")}, timeout=ASK_TIMEOUT
                    )
                except ExtensionDisconnected as exc:
                    outcome = _failed(f"lost the connection while waiting for an answer: {exc}")
                    break
                except (ExtensionError, TimeoutError) as exc:
                    answer = ""
                    notes.append(f"No answer to '{question}' ({exc}). Do not guess; finish if you cannot continue.")
                answer = str(answer or "").strip()
                if args.get("ref") and page.by_ref.get(args["ref"]):
                    field = page.by_ref[args["ref"]]
                    asked_keys.add(_key(field.get("label") or "", field.get("input_type") or ""))
                if answer:
                    deps.storage.save_answer(deps.user_id, question, answer)
                    progressed = True
                notes.append(f"The candidate answered '{question}': {answer or '(no answer)'}")
                results.append({"tool": name, "ok": bool(answer), "error": None if answer else "no answer"})
                continue

            if name in BROWSER_TOOL_NAMES:
                try:
                    action = browser_action_dict(name, args)
                    parse_actions([action])
                except (ValueError, KeyError, TypeError) as exc:
                    results.append({"tool": name, "ok": False, "error": f"invalid arguments: {exc}"})
                    continue
                meta = _ledger_meta(page, action, step)
                if action["type"] == "fill" and not str(action.get("value", "")).strip():
                    # An empty value would wipe a field the model has no answer for. Clearing a value
                    # that this run filled earlier is a correction, so that one is allowed.
                    earlier = ledger.get(meta["key"]) if meta else None
                    if earlier is None:
                        results.append({
                            "tool": name,
                            "ok": False,
                            "error": "empty value; leave this field alone or ask_human for the answer",
                        })
                        continue
                if action["type"] == "fill":
                    el = page.by_ref.get(action["ref"]) or {}
                    bad = _not_in_cv(el.get("label") or "", str(action.get("value", "")), cv_plain)
                    if bad:
                        results.append({
                            "tool": name,
                            "ok": False,
                            "error": f"'{bad}' does not appear in the CV. Do not guess it: leave this field "
                                     "empty, or ask_human for it.",
                        })
                        continue
                batch.append((name, action, meta))
                if name == "click_element":
                    await flush()  # the page may change, so later refs would be stale
                continue

            results.append({"tool": name, "ok": False, "error": f"unknown tool '{name}'"})

        if outcome is None:
            await flush()

        no_progress = 0 if progressed else state.get("no_progress", 0) + 1
        await deps.emit("act", {"results": results, "notes": notes})
        update: dict = {
            "ledger": ledger,
            "last_results": results,
            "notes": notes,
            "no_progress": no_progress,
            "pending_calls": [],
            "asked_keys": sorted(asked_keys),
            "finish_refusals": refusals,
        }
        if outcome is not None:
            update["outcome"] = outcome
        return update

    def after_observe(state: AgentState) -> str:
        return END if state.get("outcome") else "decide"

    def after_decide(state: AgentState) -> str:
        return END if state.get("outcome") else "act"

    def after_act(state: AgentState) -> str:
        return END if state.get("outcome") else "observe"

    builder = StateGraph(AgentState)
    builder.add_node("observe", observe)
    builder.add_node("decide", decide)
    builder.add_node("act", act)
    builder.add_edge(START, "observe")
    builder.add_conditional_edges("observe", after_observe, ["decide", END])
    builder.add_conditional_edges("decide", after_decide, ["act", END])
    builder.add_conditional_edges("act", after_act, ["observe", END])
    return builder.compile()


def initial_state(goal: str) -> AgentState:
    return {
        "goal": goal,
        "step": 0,
        "page": None,
        "ledger": {},
        "notes": [],
        "last_results": [],
        "pending_calls": [],
        "no_progress": 0,
        "outcome": None,
        "asked_keys": [],
        "finish_refusals": 0,
    }


async def run_agent(deps: AgentDeps, goal: str) -> dict:
    graph = build_agent_graph(deps)
    final = await graph.ainvoke(initial_state(goal), config={"recursion_limit": RECURSION_LIMIT})
    outcome = final.get("outcome") or _failed("the run ended without an outcome")
    summary = [
        {"label": e["label"], "value": e["intended"], "status": e["status"]}
        for e in final.get("ledger", {}).values()
    ]
    return {"outcome": outcome, "steps": final.get("step", 0), "filled": summary}
