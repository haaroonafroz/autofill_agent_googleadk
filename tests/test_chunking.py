from autofill_agent.cv_ingest import _section_label, split_markdown

SAMPLE_CV = """# Jane Doe

Software engineer based in London.

## Experience

### Acme Ltd - Backend Engineer

- Built payment APIs in Python.
- Reduced latency by 40 percent.

## Education

BSc Computer Science, University of Somewhere.
"""


def test_split_keeps_headers_and_sections():
    chunks = split_markdown(SAMPLE_CV, source="cv.pdf")
    assert chunks, "expected at least one chunk"
    assert all(c.metadata["source"] == "cv.pdf" for c in chunks)

    experience = [c for c in chunks if "Built payment APIs" in c.page_content][0]
    assert "## Experience" in experience.page_content
    assert _section_label(experience.metadata) == "Jane Doe > Experience > Acme Ltd - Backend Engineer"


def test_section_label_handles_missing_headers():
    assert _section_label({}) is None
    assert _section_label({"h2": "Skills"}) == "Skills"
