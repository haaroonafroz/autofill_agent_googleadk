"""
Runs a batch of browser actions through the extension.

The backend owns the decisions: it validates each action, checks the policy
against the element it targets, then sends one action at a time to the
extension. It stops at the first failure or blocked action, so the agent can
re-read the page instead of continuing blindly.
"""

from typing import Any, Dict, List, Optional

from .actions import check_policy, parse_actions
from .bridge import BridgeRegistry

SNAPSHOT_TIMEOUT = 20.0
ACTION_TIMEOUT = 20.0


class Page:
    """A snapshot indexed by ref, so the policy can look up each element."""

    def __init__(self, data: Dict[str, Any]):
        self.url: str = data.get("url", "")
        self.title: str = data.get("title", "")
        self.text: str = data.get("text", "")
        self.alerts: List[str] = data.get("alerts", [])
        self.frames: List[dict] = data.get("frames", [])
        self.by_ref: Dict[str, dict] = {}
        for frame in self.frames:
            for element in frame.get("elements", []):
                self.by_ref[element["ref"]] = {**element, "frame_id": frame.get("frame_id", 0)}

    @property
    def elements(self) -> List[dict]:
        return list(self.by_ref.values())


async def read_page(bridge: BridgeRegistry) -> Page:
    data = await bridge.request("snapshot", {}, timeout=SNAPSHOT_TIMEOUT)
    return Page(data)


async def run_actions(bridge: BridgeRegistry, raw_actions: list, page: Optional[Page] = None) -> dict:
    actions = parse_actions(raw_actions)
    page = page or await read_page(bridge)
    results: List[dict] = []

    for action in actions:
        ref = getattr(action, "ref", None)
        element = None
        frame_id = 0

        if ref:
            element = page.by_ref.get(ref)
            if element is None:
                results.append({"ok": False, "ref": ref, "error": f"unknown ref '{ref}'; read the page again"})
                break
            frame_id = element["frame_id"]

        reason = check_policy(action, element)
        if reason:
            results.append({"ok": False, "blocked": True, "ref": ref, "error": reason})
            break

        result = await bridge.request(
            "act",
            {"action": action.model_dump(), "frame_id": frame_id},
            timeout=ACTION_TIMEOUT,
        )
        results.append({"action": action.type, "ref": ref, **result})
        if not result.get("ok"):
            break

    return {"results": results, "completed": len(results) == len(actions) and all(r.get("ok") for r in results)}
