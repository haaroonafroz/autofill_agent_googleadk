"""Protocol tests for the WebSocket bridge, with a fake extension driving the socket."""

from fastapi.testclient import TestClient

from autofill_agent.server import app

SNAPSHOT = {
    "url": "http://localhost/form",
    "title": "Form",
    "text": "Apply now",
    "alerts": [],
    "frames": [
        {
            "frame_id": 0,
            "url": "http://localhost/form",
            "elements": [
                {"ref": "e1", "input_type": "text", "label": "First name", "value": ""},
                {"ref": "e2", "input_type": "submit", "label": "Submit application", "text": "Submit application"},
                {"ref": "e3", "input_type": "password", "label": "Password", "value": "[redacted]"},
            ],
        },
        {
            "frame_id": 5,
            "url": "http://localhost/frame",
            "elements": [{"ref": "f5_e4", "input_type": "text", "label": "Referee name", "value": ""}],
        },
    ],
}


def _hello(ws):
    ws.send_json({"type": "hello", "user_id": "tester"})
    ack = ws.receive_json()
    assert ack["type"] == "hello_ack"
    assert ack["user_id"] == "tester"
    return ack


def _answer(ws, request, data=None, ok=True, error=None):
    ws.send_json({"type": "response", "request_id": request["request_id"], "ok": ok, "data": data, "error": error})


def test_hello_and_ping():
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        _hello(ws)
        ws.send_json({"type": "ping"})
        assert ws.receive_json() == {"type": "pong"}


def test_snapshot_round_trip():
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        _hello(ws)
        ws.send_json({"type": "dev.snapshot", "id": "d1"})

        request = ws.receive_json()
        assert request["type"] == "request"
        assert request["command"] == "snapshot"
        _answer(ws, request, data=SNAPSHOT)

        result = ws.receive_json()
        assert result["type"] == "dev_result" and result["id"] == "d1" and result["ok"]
        assert [e["ref"] for e in result["data"]["elements"]] == ["e1", "e2", "e3", "f5_e4"]


def test_run_actions_sends_each_action_to_the_extension():
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        _hello(ws)
        ws.send_json({"type": "dev.run", "id": "r1", "actions": [
            {"type": "fill", "ref": "f5_e4", "value": "Dr. Smith"},
        ]})

        snap_request = ws.receive_json()
        assert snap_request["command"] == "snapshot"
        _answer(ws, snap_request, data=SNAPSHOT)

        act_request = ws.receive_json()
        assert act_request["command"] == "act"
        assert act_request["payload"]["frame_id"] == 5
        assert act_request["payload"]["action"]["value"] == "Dr. Smith"
        _answer(ws, act_request, data={"ok": True, "value_after": "Dr. Smith"})

        result = ws.receive_json()
        assert result["ok"] and result["data"]["completed"] is True
        assert result["data"]["results"][0]["value_after"] == "Dr. Smith"


def test_blocked_submit_never_reaches_the_extension():
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        _hello(ws)
        ws.send_json({"type": "dev.run", "id": "r2", "actions": [
            {"type": "click", "ref": "e2"},
            {"type": "fill", "ref": "e1", "value": "never sent"},
        ]})
        snap_request = ws.receive_json()
        _answer(ws, snap_request, data=SNAPSHOT)

        result = ws.receive_json()
        assert result["ok"]
        data = result["data"]
        assert data["completed"] is False
        assert len(data["results"]) == 1
        assert data["results"][0]["blocked"] is True


def test_extension_error_is_reported():
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        _hello(ws)
        ws.send_json({"type": "dev.snapshot", "id": "d2"})
        request = ws.receive_json()
        _answer(ws, request, ok=False, error="Could not read the page.")

        result = ws.receive_json()
        assert result["ok"] is False
        assert "Could not read the page" in result["error"]
