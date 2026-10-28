"""
FastAPI backend for the autofill agent.

- POST /cv            upload a CV PDF for a user (ingests it locally)
- GET  /status        CV status and whether a browser extension is connected
- WS   /ws            the side panel connects here (see bridge.py for the protocol)

Run:
    uvicorn autofill_agent.server:app --port 8000
"""

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .bridge import BridgeRegistry, ExtensionDisconnected, ExtensionError, Session
from .config import DEFAULT_USER_ID
from .cv_ingest import ingest_cv
from .executor import read_page, run_actions
from .runner import AgentRunner
from .storage import Storage

DEFAULT_GOAL = (
    "Fill out this job application using my CV. Answer only what the CV supports, "
    "ask me when something is missing, and stop at the review page. Never submit."
)

app = FastAPI(title="Autofill Agent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local development only; the extension calls this API
    allow_methods=["*"],
    allow_headers=["*"],
)

bridge = BridgeRegistry()
storage = Storage()
runner = AgentRunner(bridge, storage)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status(user_id: str = DEFAULT_USER_ID):
    profile = storage.get_profile(user_id)
    return {
        "user_id": user_id,
        "chunks": storage.chunk_count(user_id),
        "has_profile": profile is not None,
        "extension_connected": bridge.session is not None,
    }


@app.post("/cv")
def upload_cv(file: UploadFile = File(...), user_id: str = Form(DEFAULT_USER_ID)):
    """Plain `def` so FastAPI runs the slow ingestion in a worker thread."""
    suffix = Path(file.filename or "cv.pdf").suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        result = ingest_cv(tmp_path, user_id=user_id, storage=storage)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "user_id": user_id,
        "chunks": result.chunk_count,
        "name": result.profile.full_name,
    }


async def _run_dev_command(session: Session, message: dict) -> None:
    """Handles commands from the side panel's developer tools and sends back a result."""
    msg_id = message.get("id")
    kind = message.get("type")
    try:
        if kind == "dev.snapshot":
            page = await read_page(bridge)
            data = {"url": page.url, "title": page.title, "elements": page.elements, "text": page.text[:500]}
        elif kind == "dev.run":
            data = await run_actions(bridge, message.get("actions", []))
        elif kind == "agent.start":
            run_id = await runner.start(session, message.get("goal") or DEFAULT_GOAL)
            data = {"run_id": run_id}
        elif kind == "agent.stop":
            runner.request_stop()
            data = {"stopping": True}
        else:
            raise ValueError(f"unknown command {kind}")
        reply = {"type": "dev_result", "id": msg_id, "ok": True, "data": data}
    except (ExtensionError, ExtensionDisconnected, ValueError, RuntimeError, asyncio.TimeoutError) as exc:
        reply = {"type": "dev_result", "id": msg_id, "ok": False, "error": str(exc) or type(exc).__name__}
    try:
        await session.websocket.send_json(reply)
    except (WebSocketDisconnect, RuntimeError):
        pass  # the side panel closed while the command was running


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = Session(websocket=websocket)

    try:
        hello = await websocket.receive_json()
        if hello.get("type") != "hello":
            await websocket.close(code=1008)
            return

        session.user_id = hello.get("user_id") or DEFAULT_USER_ID
        bridge.attach(session)
        profile = storage.get_profile(session.user_id)
        await websocket.send_json({
            "type": "hello_ack",
            "user_id": session.user_id,
            "chunks": storage.chunk_count(session.user_id),
            "has_profile": profile is not None,
        })

        while True:
            message = await websocket.receive_json()
            kind = message.get("type")
            if kind == "response":
                session.resolve(message)
            elif kind == "ping":
                await websocket.send_json({"type": "pong"})
            elif kind in ("dev.snapshot", "dev.run", "agent.start", "agent.stop"):
                # Run in a task so this loop keeps reading the extension's responses.
                asyncio.create_task(_run_dev_command(session, message))
            else:
                await websocket.send_json({"type": "error", "error": f"unknown message type {kind}"})
    except WebSocketDisconnect:
        pass
    finally:
        bridge.detach(session)
