from pathlib import Path

from autofill_agent.profile import CandidateProfile, Education, Experience
from autofill_agent.storage import Storage, normalize_key


def make_storage(tmp_path: Path) -> Storage:
    return Storage(tmp_path / "test.db")


def test_profile_round_trip(tmp_path):
    storage = make_storage(tmp_path)
    profile = CandidateProfile(
        full_name="Jane Doe",
        email="jane@example.com",
        experience=[Experience(company="Acme", title="Engineer", highlights=["Built APIs"])],
        education=[Education(institution="Uni", degree="BSc")],
        skills=["Python"],
    )
    storage.save_profile("u1", profile, source="cv.pdf")

    loaded = storage.get_profile("u1")
    assert loaded == profile
    assert storage.get_profile("someone-else") is None


def test_profile_overwrite(tmp_path):
    storage = make_storage(tmp_path)
    storage.save_profile("u1", CandidateProfile(full_name="Old"), source=None)
    storage.save_profile("u1", CandidateProfile(full_name="New"), source=None)
    assert storage.get_profile("u1").full_name == "New"


def test_search_ranks_by_similarity_and_isolates_users(tmp_path):
    storage = make_storage(tmp_path)
    storage.replace_chunks(
        "u1",
        contents=["python backend work", "cooking recipes", "sql databases"],
        sections=["Experience", "Hobbies", "Skills"],
        vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7, 0.7, 0.0]],
    )
    storage.replace_chunks("u2", contents=["other person"], sections=[None], vectors=[[1.0, 0.0, 0.0]])

    hits = storage.search_chunks("u1", [1.0, 0.0, 0.0], k=2)
    assert [h.content for h in hits] == ["python backend work", "sql databases"]
    assert hits[0].score > hits[1].score
    assert all(h.content != "other person" for h in hits)


def test_replace_chunks_overwrites_previous_cv(tmp_path):
    storage = make_storage(tmp_path)
    storage.replace_chunks("u1", ["a", "b"], [None, None], [[1, 0], [0, 1]])
    storage.replace_chunks("u1", ["c"], [None], [[1, 0]])
    assert storage.chunk_count("u1") == 1


def test_search_with_no_chunks_returns_empty(tmp_path):
    assert make_storage(tmp_path).search_chunks("nobody", [1.0, 0.0]) == []


def test_answer_memory_matches_normalized_questions(tmp_path):
    storage = make_storage(tmp_path)
    storage.save_answer("u1", "What is your notice period?", "Two weeks")
    assert storage.get_answer("u1", "  what is your   NOTICE period? ") == "Two weeks"
    assert storage.get_answer("u2", "What is your notice period?") is None


def test_normalize_key():
    assert normalize_key("  Hello   World ") == "hello world"
