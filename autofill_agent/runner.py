"""
Runs the agent in the background and streams its progress to the side panel.

One run at a time. Events go out as:
    {"type": "agent_event", "run_id": ..., "kind": ..., "data": {...}}
and the run ends with a `done` event carrying the outcome.
"""

import asyncio
import uuid
from typing import Optional

from .agent_loop import AgentDeps, run_agent
from .bridge import BridgeRegistry, Session
from .cv_ingest import describe_hit, search_cv
from .llm import get_chat_model
from .storage import Storage


class AgentRunner:
    def __init__(self, bridge: BridgeRegistry, storage: Storage):
        self.bridge = bridge
        self.storage = storage
        self.run_id: Optional[str] = None
        self._task: Optional[asyncio.Task] = None
        self._stop_requested = False

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self, session: Session, goal: str) -> str:
        if self.running:
            raise RuntimeError("an agent run is already in progress; stop it first")
        if not session.user_id:
            raise RuntimeError("the side panel has not said hello yet")
        if self.storage.get_profile(session.user_id) is None:
            raise RuntimeError("upload a CV first, so the agent has your profile")

        self.run_id = str(uuid.uuid4())
        self._stop_requested = False
        self._task = asyncio.create_task(self._run(session, goal, self.run_id))
        return self.run_id

    def request_stop(self) -> None:
        self._stop_requested = True

    async def _emit(self, session: Session, run_id: str, kind: str, data: dict) -> None:
        try:
            await session.websocket.send_json({"type": "agent_event", "run_id": run_id, "kind": kind, "data": data})
        except Exception:
            pass  # the panel went away; the run stops at its next check

    async def _run(self, session: Session, goal: str, run_id: str) -> None:
        user_id = session.user_id

        async def emit(kind: str, data: dict) -> None:
            await self._emit(session, run_id, kind, data)

        try:
            deps = AgentDeps(
                llm=get_chat_model(),
                bridge=self.bridge,
                storage=self.storage,
                user_id=user_id,
                profile=self.storage.get_profile(user_id),
                cv_text=self.storage.cv_text(user_id),
                emit=emit,
                should_stop=lambda: self._stop_requested or session.closed,
                cv_search=lambda q, kind=None: [
                    describe_hit(h) for h in search_cv(q, user_id=user_id, storage=self.storage, kind=kind)
                ],
            )
            result = await run_agent(deps, goal)
            await emit("done", result)
        except Exception as exc:
            await emit("done", {
                "outcome": {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"},
                "steps": 0,
                "filled": [],
            })
