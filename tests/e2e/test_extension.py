"""
End-to-end test: loads the unpacked extension into Chromium, connects the side
panel to a real backend, and runs actions on a local test form.

No OpenAI calls are made. Run with:
    python -m pytest tests/e2e -q
"""

import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect, sync_playwright

ROOT = Path(__file__).resolve().parents[2]
EXTENSION_DIR = ROOT / "frontend"
FIXTURES_DIR = Path(__file__).parent / "fixtures"
# Use a port nobody else connects to. Port 8000 is the developer backend, and any
# browser with this extension installed reconnects there and would take over the session.
API_PORT = 8200
FIXTURE_PORT = 8765


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
    if not _port_free(API_PORT):
        pytest.skip(f"port {API_PORT} is busy; stop the running backend to run the e2e test")
    env = os.environ.copy()
    env["AUTOFILL_DB_PATH"] = str(Path(tempfile.mkdtemp(prefix="e2e-")) / "e2e.db")

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
    profile = tempfile.mkdtemp(prefix="chromium-profile-")
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            profile,
            channel="chromium",  # the full Chromium build, which supports extensions in headless mode
            headless=True,
            args=[
                f"--disable-extensions-except={EXTENSION_DIR}",
                f"--load-extension={EXTENSION_DIR}",
            ],
        )
        yield context
        context.close()


def _extension_id(context) -> str:
    workers = context.service_workers
    if not workers:
        workers = [context.wait_for_event("serviceworker", timeout=20000)]
    return workers[0].url.split("/")[2]


@pytest.fixture(scope="module")
def form_page(browser_context) -> Page:
    page = browser_context.new_page()
    page.goto(f"http://localhost:{FIXTURE_PORT}/form.html")
    page.bring_to_front()  # makes this the last web tab the service worker saw
    return page


@pytest.fixture(scope="module")
def panel(browser_context, form_page) -> Page:
    ext_id = _extension_id(browser_context)
    panel = browser_context.new_page()
    panel.goto(f"chrome-extension://{ext_id}/sidepanel.html")
    # Point the panel at the test backend, then reload so it connects there.
    panel.evaluate(
        "(url) => new Promise((r) => chrome.storage.local.set({ backend_url: url }, r))",
        f"http://localhost:{API_PORT}",
    )
    panel.reload()
    expect(panel.locator("#connection")).to_have_text("Connected", timeout=20000)
    form_page.bring_to_front()
    return panel


def _read_refs(panel: Page) -> dict:
    """Clicks 'Read page' and maps each visible label to its ref."""
    panel.click("#readBtn")
    expect(panel.locator("#devStatus")).to_have_text(re.compile(r"Read \d+ elements\."), timeout=20000)
    # The status and the table are updated together, so check the table holds every element.
    count = int(re.search(r"Read (\d+) elements", panel.inner_text("#devStatus")).group(1))
    expect(panel.locator("#elements tbody tr")).to_have_count(count, timeout=5000)
    rows = panel.locator("#elements tbody tr")
    refs = {}
    for i in range(rows.count()):
        row = rows.nth(i)
        refs[row.locator("td").nth(2).inner_text()] = row.get_attribute("data-ref")
    return refs


def _run(panel: Page, actions: list) -> None:
    panel.fill("#actionsInput", json.dumps(actions))
    panel.click("#runBtn")
    expect(panel.locator("#devStatus")).not_to_have_text("Running actions...", timeout=30000)


def test_side_panel_connects_and_reads_page(panel, form_page):
    refs = _read_refs(panel)
    assert "First name" in refs
    assert "Submit application" in refs
    assert "Referee name" in refs  # element inside the iframe


def test_dom_actions_fill_select_and_check(panel, form_page):
    refs = _read_refs(panel)
    _run(panel, [
        {"type": "fill", "ref": refs["First name"], "value": "Ada"},
        {"type": "select", "ref": refs["Country"], "label": "Germany"},
        {"type": "check", "ref": refs["I agree to the privacy policy"]},
        {"type": "fill", "ref": refs["Why do you want this job?"], "value": "Because I build things"},
    ])
    expect(panel.locator("#devStatus")).to_have_text("All actions completed.", timeout=30000)

    assert form_page.input_value("#fname") == "Ada"
    assert form_page.input_value("#country") == "DE"
    assert form_page.is_checked("#privacy")
    assert form_page.input_value("#why") == "Because I build things"


def test_fill_inside_iframe(panel, form_page):
    refs = _read_refs(panel)
    _run(panel, [{"type": "fill", "ref": refs["Referee name"], "value": "Dr. Smith"}])
    expect(panel.locator("#devStatus")).to_have_text("All actions completed.", timeout=30000)
    assert form_page.frame_locator("#frame").locator("#referee").input_value() == "Dr. Smith"


def test_real_mouse_and_keyboard_typing(panel, form_page):
    refs = _read_refs(panel)
    form_page.bring_to_front()
    _run(panel, [
        {"type": "click", "ref": refs["Live search"]},
        {"type": "type", "text": "hello"},
    ])
    expect(panel.locator("#devStatus")).to_have_text("All actions completed.", timeout=30000)
    assert form_page.input_value("#search") == "hello"
    assert form_page.inner_text("#echo") == "hello"


def test_submit_click_is_blocked_by_policy(panel, form_page):
    refs = _read_refs(panel)
    _run(panel, [{"type": "click", "ref": refs["Submit application"]}])
    expect(panel.locator("#devStatus")).to_have_text("Stopped early. See log.", timeout=30000)
    expect(panel.locator("#log")).to_contain_text("BLOCKED")
    assert form_page.evaluate("window.submitted === true") is False


def test_password_is_never_filled(panel, form_page):
    refs = _read_refs(panel)
    _run(panel, [{"type": "fill", "ref": refs["Password"], "value": "secret"}])
    expect(panel.locator("#devStatus")).to_have_text("Stopped early. See log.", timeout=30000)
    assert form_page.input_value("#password") == ""
