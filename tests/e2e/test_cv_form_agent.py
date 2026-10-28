"""
Live test: how much of cv_form.html can the agent fill from the CV, and does it
avoid inventing values for things the CV does not say?

It uses real gpt-6-luna calls and a real Chromium with the extension, so it is
skipped unless OPENAI_API_KEY is set. The page, the expected answers and the CV
profile are fixed files, so the score is reproducible apart from the model's own
variation. It uses ports 8102/8767 and does not touch a running dev server.

    python -m pytest tests/e2e/test_cv_form_agent.py -q -s
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
from autofill_agent.cv_ingest import index_markdown
from autofill_agent.storage import Storage
from tests.e2e.cv_form_score import (
    ANSWER_TO_QUESTIONS,
    FIXTURES_DIR,
    load_fields,
    load_profile,
    read_state,
    score,
)

ROOT = Path(__file__).resolve().parents[2]
EXTENSION_DIR = ROOT / "frontend"
API_PORT = 8102
FIXTURE_PORT = 8767
USER_ID = "cv-form-user"
RUN_TIMEOUT_S = 900

# The agent must fill at least this share of the answerable fields, and must not invent values.
MIN_FILL_RATE = 0.85
MAX_INVENTED = 0

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
        pytest.skip(f"ports {API_PORT}/{FIXTURE_PORT} are busy")
    db_path = Path(tempfile.mkdtemp(prefix="cv-form-")) / "cv_form.db"
    storage = Storage(db_path)
    # Seed the CV profile directly so the run does not depend on PDF extraction.
    storage.save_profile(USER_ID, load_profile(), source="cv_profile.json")
    # Index the same CV for search_cv, so the agent has the RAG path as well as the profile.
    index_markdown(
        (FIXTURES_DIR / "cv_markdown.md").read_text(encoding="utf-8"),
        user_id=USER_ID, source="cv_markdown.md", storage=storage,
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
    profile = tempfile.mkdtemp(prefix="cv-form-chromium-")
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
    page.goto(f"http://localhost:{FIXTURE_PORT}/cv_form.html")
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
    panel.evaluate(
        "() => chrome.storage.local.set({ user_id: '%s', backend_url: 'http://localhost:%d' })" % (USER_ID, API_PORT)
    )
    panel.reload()
    expect(panel.locator("#connection")).to_have_text("Connected", timeout=20000)
    form_page.bring_to_front()
    return panel


def test_agent_fills_the_cv_form_without_inventing_or_submitting(agent_panel, form_page):
    fields = load_fields()
    agent_panel.click("#startBtn")

    deadline = time.time() + RUN_TIMEOUT_S
    status = ""
    while time.time() < deadline:
        if agent_panel.locator("#askBox").is_visible():
            agent_panel.fill("#askAnswer", ANSWER_TO_QUESTIONS)
            agent_panel.click("#askSend")
        status = agent_panel.inner_text("#agentStatus")
        if status.startswith(("Reached", "Stopped", "Failed")):
            break
        time.sleep(1.0)

    log = agent_panel.inner_text("#agentLog")
    print("\n--- agent feed ---\n" + log)
    assert not status.startswith("Failed"), f"{status}\n{log}"
    assert status.startswith(("Reached", "Stopped")), f"the run did not finish in time: {status}\n{log}"

    report = score(fields, read_state(form_page, fields))
    print("\n--- cv form score ---\n" + report.summary())

    assert form_page.evaluate("window.submitted === true") is False, "the agent must never submit"
    assert len(report.invented) <= MAX_INVENTED, report.summary()
    assert report.fill_rate >= MIN_FILL_RATE, report.summary()
