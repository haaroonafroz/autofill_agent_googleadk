import sqlite3

import re

from langchain_core.messages import HumanMessage

from autofill_agent import cv_ingest
from autofill_agent.cv_ingest import describe_hit, index_markdown, search_cv
from autofill_agent.cv_metadata import PRESENT_KEY, CVLabels, EntryLabel, date_key
from autofill_agent.storage import Storage


def test_date_key_reads_common_formats():
    assert date_key("07/2025") == "2025-07"
    assert date_key("7.2025") == "2025-07"
    assert date_key("Jul 2025") == "2025-07"
    assert date_key("September 2019") == "2019-09"
    assert date_key("2016") == "2016-01"
    assert date_key("Present") == PRESENT_KEY
    assert date_key("current") == PRESENT_KEY
    assert date_key(None) is None
    assert date_key("sometime") is None


def _store_with_entries(tmp_path):
    storage = Storage(tmp_path / "meta.db")
    rows = [
        # (content, kind, entity, role, start, end)
        ("Older job", "experience", "Old Co", "Analyst", "11/2022", "03/2023"),
        ("Current job", "experience", "Now Co", "Engineer", "07/2025", "Present"),
        ("Undated job", "experience", "Mystery Co", "Intern", None, None),
        ("Previous job", "experience", "Prev Co", "Developer", "03/2024", "06/2025"),
        ("Project B", "project", "Second project", None, None, None),
        ("Project A", "project", "First project", None, None, None),
        ("Degree", "education", "Uni", "M.Sc.", "04/2023", "07/2026"),
        ("Hobbies", "other", None, None, None, None),
    ]
    contents = [r[0] for r in rows]
    metadata = [
        {"kind": r[1], "entity": r[2], "role": r[3], "start": r[4], "end": r[5],
         "start_key": date_key(r[4]), "end_key": date_key(r[5])}
        for r in rows
    ]
    storage.replace_chunks("u1", contents=contents, sections=[None] * len(rows),
                           vectors=[[1.0, 0.0]] * len(rows), metadata=metadata)
    return storage


def test_experience_is_listed_newest_first_with_undated_last(tmp_path):
    storage = _store_with_entries(tmp_path)
    hits = storage.list_chunks("u1", "experience")
    assert [h.entity for h in hits] == ["Now Co", "Prev Co", "Old Co", "Mystery Co"]
    assert hits[0].role == "Engineer" and hits[0].start == "07/2025" and hits[0].end == "Present"


def test_projects_keep_cv_order_and_other_kinds_are_filtered(tmp_path):
    storage = _store_with_entries(tmp_path)
    assert [h.entity for h in storage.list_chunks("u1", "project")] == ["Second project", "First project"]
    assert [h.kind for h in storage.list_chunks("u1", "education")] == ["education"]
    assert storage.list_chunks("u1", "skills") == []


def test_listing_does_not_depend_on_the_query_vector(tmp_path, monkeypatch):
    storage = _store_with_entries(tmp_path)
    monkeypatch.setattr(cv_ingest, "get_embeddings", lambda: FakeEmbeddings())
    first = search_cv("anything", user_id="u1", storage=storage, kind="experience")
    second = search_cv("something else entirely", user_id="u1", storage=storage, kind="experience")
    assert [h.content for h in first] == [h.content for h in second]


def test_describe_hit_puts_the_metadata_line_before_the_text(tmp_path):
    storage = _store_with_entries(tmp_path)
    hit = storage.list_chunks("u1", "experience")[0]
    assert describe_hit(hit) == "[experience | Now Co | Engineer | 07/2025 - Present]\nCurrent job"
    plain = storage.list_chunks("u1", "other")[0]
    assert describe_hit(plain) == "[other]\nHobbies"


def test_old_database_is_migrated_and_keeps_its_rows(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE cv_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, chunk_index INTEGER NOT NULL,
            section TEXT, content TEXT NOT NULL, embedding BLOB NOT NULL);
        INSERT INTO cv_chunks (user_id, chunk_index, section, content, embedding)
        VALUES ('u1', 0, 'Skills', 'Python and SQL', x'0000803F00000000');
    """)
    conn.commit()
    conn.close()

    storage = Storage(path)  # migrates the table in place
    columns = {row[1] for row in sqlite3.connect(path).execute("PRAGMA table_info(cv_chunks)")}
    assert {"kind", "entity", "start_key", "end_key"} <= columns
    hits = storage.search_chunks("u1", [1.0, 0.0], k=1)
    assert hits[0].content == "Python and SQL" and hits[0].kind is None


CV_WITH_SPLIT_HEADINGS = """# Jane Doe

## Qdrant Solutions GmbH

- Built payment APIs in Python.

## AI Engineer - Working Student

07/2025 - 02/2026

## Hobbies

Climbing.
"""


def test_index_groups_split_headings_into_one_entry_with_its_dates(tmp_path):
    # The model labels the company heading and the dates heading as two entries with identical fields.
    same_job = {"kind": "experience", "entity": "Qdrant Solutions GmbH",
                "role": "AI Engineer - Working Student", "start": "07/2025", "end": "02/2026"}
    llm = FakeLLM([{"match": ["Qdrant"], **same_job}, {"match": ["07/2025"], **same_job}])
    embeddings = FakeEmbeddings()
    storage = Storage(tmp_path / "index.db")

    index_markdown(CV_WITH_SPLIT_HEADINGS, user_id="u1", source="cv.pdf", storage=storage,
                   llm=llm, embeddings=embeddings)

    experience = storage.list_chunks("u1", "experience")
    assert len(experience) == 1, "both headings of one job should make one entry"
    assert experience[0].entity == "Qdrant Solutions GmbH"
    assert experience[0].start == "07/2025" and experience[0].end == "02/2026"
    assert "Built payment APIs" in experience[0].content and "07/2025 - 02/2026" in experience[0].content
    assert any(t.startswith("[experience | Qdrant Solutions GmbH | AI Engineer - Working Student | 07/2025 - 02/2026]")
               for t in embeddings.seen)
    assert storage.list_chunks("u1", "other"), "unlabelled headings are kept as 'other'"


def test_values_not_in_the_cv_are_dropped(tmp_path):
    # The model invents a date and a company name for "Hobbies". Neither appears in the CV.
    llm = FakeLLM([
        {"match": ["Hobbies"], "kind": "experience", "entity": "Hobbies Ltd",
         "role": None, "start": "03/2026", "end": "09/2026"},
    ])
    storage = Storage(tmp_path / "grounded.db")
    index_markdown(CV_WITH_SPLIT_HEADINGS, user_id="u1", source="cv.pdf", storage=storage,
                   llm=llm, embeddings=FakeEmbeddings())

    hobbies = storage.list_chunks("u1", "experience")
    assert len(hobbies) == 1
    assert hobbies[0].entity is None and hobbies[0].start is None and hobbies[0].end is None


class FakeLLM:
    """
    Stands in for the structured-output call. Each spec names the section texts it matches
    (by substring); the fake finds the numbered sections in the prompt and returns the entries.
    """

    def __init__(self, specs):
        self.specs = specs

    def with_structured_output(self, schema, method=None):
        return self

    def invoke(self, messages):
        body = next(m.content for m in messages if isinstance(m, HumanMessage))
        numbered = {int(n): text for n, text in re.findall(r"\[S(\d+)\]\n(.*?)(?=\n\n\[S\d+\]\n|\Z)", body, re.S)}
        entries = []
        for spec in self.specs:
            sections = [n for n, text in numbered.items() if any(k in text for k in spec["match"])]
            entries.append(EntryLabel(
                kind=spec["kind"], entity=spec.get("entity"), role=spec.get("role"),
                start=spec.get("start"), end=spec.get("end"), sections=sections,
            ))
        claimed = {n for e in entries for n in e.sections}
        entries += [EntryLabel(kind="other", sections=[n]) for n in numbered if n not in claimed]
        return CVLabels(entries=entries)


class FakeEmbeddings:
    def __init__(self):
        self.seen = []

    def embed_documents(self, texts):
        self.seen.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]
