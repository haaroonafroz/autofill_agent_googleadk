"""
Checks the scoring oracle for cv_form.html without the agent or the extension:
expected values come from the CV, a perfect fill scores 100%, an empty form
scores 0%, and values typed into not-in-CV fields are counted as inventions.

    python -m pytest tests/e2e/test_cv_form_oracle.py -q
"""

import json

import pytest
from playwright.sync_api import Page, sync_playwright

from tests.e2e.cv_form_score import (
    ANSWER_TO_QUESTIONS,
    FORM_FILE,
    load_fields,
    load_profile,
    read_state,
    score,
)

ANSWERABLE = [f for f in load_fields() if not f.get("ask")]
ASKED = [f for f in load_fields() if f.get("ask")]
# Derived by inference from "M.Sc." rather than stated in the CV, so it is not checked against the text.
INFERRED = {"highest-education"}


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        yield b
        b.close()


@pytest.fixture
def page(browser) -> Page:
    page = browser.new_page()
    page.goto(FORM_FILE.as_uri())
    yield page
    page.close()


def _fill_expected(page: Page, field: dict) -> None:
    if field["kind"] == "select":
        page.select_option(f"#{field['id']}", field["expect"])
    elif field.get("expect") is not None:
        page.fill(f"#{field['id']}", field["expect"])
    else:  # contains-list: type every required phrase
        page.fill(f"#{field['id']}", "; ".join(field["contains"]))


def test_expected_values_come_from_the_cv():
    cv_text = json.dumps(load_profile().model_dump(), ensure_ascii=False).lower()
    for field in ANSWERABLE:
        if field["id"] in INFERRED:
            continue
        values = [field["expect"]] if field.get("expect") is not None else field["contains"]
        for value in values:
            assert value.lower() in cv_text, f"{field['id']}: '{value}' is not in the CV profile"


def test_perfect_fill_scores_full_marks(page):
    for field in ANSWERABLE:
        _fill_expected(page, field)
    report = score(load_fields(), read_state(page, load_fields()))
    assert report.fill_rate == 1.0, report.summary()
    assert report.invented == [], report.summary()
    assert report.asked_correct == len(ASKED), report.summary()


def test_empty_form_scores_zero_fill(page):
    report = score(load_fields(), read_state(page, load_fields()))
    assert report.answerable_correct == 0
    assert len(report.missed) == len(ANSWERABLE)
    assert report.asked_correct == len(ASKED)


def test_invented_values_in_not_in_cv_fields_are_counted(page):
    for field in ASKED:
        selector = f"#{field['id']}"
        if field["kind"] == "text":
            page.fill(selector, "Invented value")
        elif field["kind"] == "select":
            page.select_option(selector, "Yes")
        elif field["kind"] == "checkbox":
            page.check(selector)
        elif field["kind"] == "radio":
            page.check("#sponsorship-yes")
    report = score(load_fields(), read_state(page, load_fields()))
    assert sorted(report.invented) == sorted(f["id"] for f in ASKED), report.summary()


def test_candidate_answer_to_a_question_is_not_an_invention(page):
    for field in ASKED:
        if field["kind"] == "text":
            page.fill(f"#{field['id']}", ANSWER_TO_QUESTIONS)
    report = score(load_fields(), read_state(page, load_fields()))
    text_asks = [f for f in ASKED if f["kind"] == "text"]
    assert not [i for i in report.invented if i in {f["id"] for f in text_asks}], report.summary()


def test_submit_is_recorded_and_never_sends(page):
    assert page.evaluate("window.submitted") is False
    # The three required fields must be filled, or the browser blocks the submit before the page sees it.
    page.fill("#full-name", "Haaroon Afroz Ognawala")
    page.fill("#email", "haaroon.afroz@gmail.com")
    page.fill("#phone", "+49 000 0000")
    page.click("#submit")
    assert page.evaluate("window.submitted") is True
    assert page.url.startswith("file://"), "the form must not navigate away"
