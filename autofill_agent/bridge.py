"""
Backend side of the extension bridge.

The side panel holds a WebSocket to the backend. The backend sends `request`
messages (snapshot the page, run actions, take a screenshot) and awaits the
matching `response`. The agent loop in later phases uses the same `request()`
call, so the protocol is defined once here.

Message shapes (JSON):

  extension -> backend
    {"type": "hello", "user_id": str}
    {"type": "response", "request_id": str, "ok": bool, "data": any, "error": str|null}
    {"type": "ping"}

  backend -> extension
    {"type": "hello_ack", "user_id": str, "chunks": int, "has_profile": bool}
    {"type": "request", "request_id": str, "command": str, "payload": object}
    {"type": "pong"}
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from fastapi import WebSocket


class ExtensionError(RuntimeError):
    """The extension reported a failure for a request."""


class ExtensionDisconnected(RuntimeError):
    """No extension is connected, or it disconnected while a request was pending."""


@dataclass
class Session:
    websocket: WebSocket
    user_id: Optional[str] = None
    pending: Dict[str, asyncio.Future] = field(default_factory=dict)
    closed: bool = False

    async def request(self, command: str, payload: Optional[dict] = None, timeout: float = 30.0) -> Any:
        if self.closed:
            raise ExtensionDisconnected("the browser extension is not connected")

        request_id = str(uuid.uuid4())
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self.pending[request_id] = future
        try:
            await self.websocket.send_json(
                {"type": "request", "request_id": request_id, "command": command, "payload": payload or {}}
            )
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self.pending.pop(request_id, None)

    def resolve(self, message: dict) -> None:
        future = self.pending.get(message.get("request_id", ""))
        if future is None or future.done():
            return
        if message.get("ok"):
            future.set_result(message.get("data"))
        else:
            future.set_exception(ExtensionError(message.get("error") or "unknown extension error"))

    def fail_all(self) -> None:
        self.closed = True
        for future in self.pending.values():
            if not future.done():
                future.set_exception(ExtensionDisconnected("the browser extension disconnected"))


class BridgeRegistry:
    """Keeps track of the connected extension session (one per browser)."""

    def __init__(self) -> None:
        self._session: Optional[Session] = None

    @property
    def session(self) -> Optional[Session]:
        if self._session and not self._session.closed:
            return self._session
        return None

    def attach(self, session: Session) -> None:
        # A newer side panel replaces the older connection.
        if self._session and not self._session.closed:
            self._session.fail_all()
        self._session = session

    def detach(self, session: Session) -> None:
        session.fail_all()
        if self._session is session:
            self._session = None

    async def request(self, command: str, payload: Optional[dict] = None, timeout: float = 30.0) -> Any:
        session = self.session
        if session is None:
            raise ExtensionDisconnected("the browser extension is not connected (open the side panel)")
        return await session.request(command, payload, timeout=timeout)
