"""
Live end-to-end test of the agent: real gpt-6-luna, real Chromium with the
extension, a local backend on port 8100 and the fixture form.

It uses API calls, so it is skipped unless OPENAI_API_KEY is set.
It does not use port 8000, so a running dev server is not affected.

    python -m pytest tests/e2e/test_agent_e2e.py -q -s
"""

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect, sync_playwright

from autofill_agent.config import OPENAI_API_KEY
from autofill_agent.profile import CandidateProfile
from autofill_agent.storage import Storage

ROOT = Path(__file__).resolve().parents[2]
EXTENSION_DIR = ROOT / "frontend"
FIXTURES_DIR = Path(__file__).parent / "fixtures"
API_PORT = 8100
FIXTURE_PORT = 8766
USER_ID = "e2e-agent-user"

pytestmark = pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY is not set")


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def _wait_for_port(port: int, timeout: float = 20) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _port_free(port):
            return
        time.sleep(0.2)
    raise RuntimeError(f"port {port} did not open")


@pytest.fixture(scope="module")
def servers():
    if not (_port_free(API_PORT) and _port_free(FIXTURE_PORT)):
        pytest.skip("ports 8100/8766 are busy")
    db_path = Path(tempfile.mkdtemp(prefix="agent-e2e-")) / "agent.db"
    Storage(db_path).save_profile(
        USER_ID,
        CandidateProfile(
            full_name="Haaroon Afroz Ognawala",
            email="haaroon.afroz@gmail.com",
            location="Weimar, Germany",
            summary="Data analyst and machine learning engineer.",
            skills=["Python", "SQL", "PyTorch", "LangChain"],
        ),
        source="e2e",
    )
    env = os.environ.copy()
    env["AUTOFILL_DB_PATH"] = str(db_path)

    api = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "autofill_agent.server:app", "--port", str(API_PORT), "--log-level", "warning"],
        cwd=ROOT, env=env,
    )
    static = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(FIXTURE_PORT), "--bind", "127.0.0.1", "--directory", str(FIXTURES_DIR)],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_port(API_PORT)
        _wait_for_port(FIXTURE_PORT)
        yield
    finally:
        for proc in (api, static):
            proc.terminate()
            proc.wait(timeout=10)


@pytest.fixture(scope="module")
def browser_context(servers):
    profile = tempfile.mkdtemp(prefix="agent-chromium-")
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            profile,
            channel="chromium",
            headless=True,
            args=[f"--disable-extensions-except={EXTENSION_DIR}", f"--load-extension={EXTENSION_DIR}"],
        )
        yield context
        context.close()


@pytest.fixture(scope="module")
def form_page(browser_context) -> Page:
    page = browser_context.new_page()
    page.goto(f"http://localhost:{FIXTURE_PORT}/form.html")
    page.bring_to_front()
    return page


@pytest.fixture(scope="module")
def agent_panel(browser_context, form_page) -> Page:
    workers = browser_context.service_workers or [browser_context.wait_for_event("serviceworker", timeout=20000)]
    ext_id = workers[0].url.split("/")[2]

    panel = browser_context.new_page()
    panel.goto(f"chrome-extension://{ext_id}/sidepanel.html")
    # The panel writes a random user id on its first load. Wait for that write before ours, or it wins.
    expect(panel.locator("#userId")).not_to_have_value("", timeout=10000)
    # Point the panel at the test backend, then reload so it connects there.
    panel.evaluate(
        "() => chrome.storage.local.set({ user_id: '%s', backend_url: 'http://localhost:%d' })" % (USER_ID, API_PORT)
    )
    panel.reload()
    expect(panel.locator("#connection")).to_have_text("Connected", timeout=20000)
    form_page.bring_to_front()
    return panel


def test_agent_fills_the_form_and_never_submits(agent_panel, form_page):
    agent_panel.click("#startBtn")

    deadline = time.time() + 480
    status = ""
    while time.time() < deadline:
        if agent_panel.locator("#askBox").is_visible():
            # The agent asked the candidate something. This test has no answer, so it says so.
            agent_panel.fill("#askAnswer", "Not specified")
            agent_panel.click("#askSend")
        status = agent_panel.inner_text("#agentStatus")
        if status.startswith(("Reached", "Stopped", "Failed")):
            break
        time.sleep(1.0)

    log = agent_panel.inner_text("#agentLog")
    print("\n--- agent feed ---\n" + log)
    assert not status.startswith("Failed"), f"{status}\n{log}"
    assert status.startswith(("Reached", "Stopped")), f"the run did not finish in time: {status}\n{log}"

    assert form_page.input_value("#fname") == "Haaroon", log  # given name only
    assert form_page.input_value("#email") == "haaroon.afroz@gmail.com", log
    assert form_page.evaluate("window.submitted === true") is False, "the agent must never submit"
