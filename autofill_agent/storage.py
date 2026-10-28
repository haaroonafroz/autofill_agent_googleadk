"""
Local SQLite storage for the autofill agent.

Tables:
- cv_profiles:   one structured profile (JSON) per user
- cv_chunks:     CV text chunks with their embedding vectors (float32 BLOBs)
- answer_memory: answers the user gave to questions the CV could not answer

Vector search is done in numpy over the user's chunks. A CV is a few dozen
chunks, so this is fast and avoids running a separate vector database.
"""

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import numpy as np

from .config import DB_PATH
from .cv_metadata import DATED_KINDS
from .profile import CandidateProfile

SCHEMA = """
CREATE TABLE IF NOT EXISTS cv_profiles (
    user_id      TEXT PRIMARY KEY,
    profile_json TEXT NOT NULL,
    source       TEXT,
    updated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cv_chunks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      TEXT NOT NULL,
    chunk_index  INTEGER NOT NULL,
    section      TEXT,
    content      TEXT NOT NULL,
    embedding    BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cv_chunks_user ON cv_chunks(user_id);

CREATE TABLE IF NOT EXISTS answer_memory (
    user_id      TEXT NOT NULL,
    question_key TEXT NOT NULL,
    question     TEXT NOT NULL,
    answer       TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (user_id, question_key)
);
"""


@dataclass
class ChunkHit:
    content: str
    section: Optional[str]
    score: float = 0.0
    kind: Optional[str] = None
    entity: Optional[str] = None
    role: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None


# Metadata columns added after the first release. Older databases get them by migration.
METADATA_COLUMNS = {
    "kind": "TEXT",
    "entity": "TEXT",
    "role": "TEXT",
    "start_text": "TEXT",
    "end_text": "TEXT",
    "start_key": "TEXT",  # 'YYYY-MM' used for sorting
    "end_key": "TEXT",
}
_CHUNK_COLUMNS = (
    "content, section, embedding, kind, entity, role, start_text, end_text"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_key(question: str) -> str:
    """Stable key for matching the same question asked in different forms."""
    return " ".join(question.lower().split())


def _hit(row: tuple, score: float) -> ChunkHit:
    # Column order follows _CHUNK_COLUMNS: content, section, embedding, kind, entity, role, start, end.
    return ChunkHit(
        content=row[0], section=row[1], score=score,
        kind=row[3], entity=row[4], role=row[5], start=row[6], end=row[7],
    )


def _migrate(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(cv_chunks)")}
    for name, sql_type in METADATA_COLUMNS.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE cv_chunks ADD COLUMN {name} {sql_type}")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cv_chunks_kind ON cv_chunks(user_id, kind)")


class Storage:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            _migrate(conn)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- CV profile -------------------------------------------------------

    def save_profile(self, user_id: str, profile: CandidateProfile, source: Optional[str]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cv_profiles (user_id, profile_json, source, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    source = excluded.source,
                    updated_at = excluded.updated_at
                """,
                (user_id, profile.model_dump_json(), source, _now()),
            )

    def get_profile(self, user_id: str) -> Optional[CandidateProfile]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT profile_json FROM cv_profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
        if row is None:
            return None
        return CandidateProfile.model_validate(json.loads(row[0]))

    # --- CV chunks --------------------------------------------------------

    def replace_chunks(
        self,
        user_id: str,
        contents: Sequence[str],
        sections: Sequence[Optional[str]],
        vectors: Sequence[Sequence[float]],
        metadata: Optional[Sequence[Optional[dict]]] = None,
    ) -> int:
        """
        Replaces all chunks for a user (re-ingesting a CV overwrites the old one).
        `metadata[i]` is a dict with keys kind, entity, role, start, end, start_key, end_key.
        """
        if not (len(contents) == len(sections) == len(vectors)):
            raise ValueError("contents, sections and vectors must have the same length")
        meta = list(metadata) if metadata is not None else [None] * len(contents)
        if len(meta) != len(contents):
            raise ValueError("metadata must have one entry per chunk")
        rows = []
        for i in range(len(contents)):
            m = meta[i] or {}
            rows.append((
                user_id, i, sections[i], contents[i],
                np.asarray(vectors[i], dtype=np.float32).tobytes(),
                m.get("kind"), m.get("entity"), m.get("role"),
                m.get("start"), m.get("end"), m.get("start_key"), m.get("end_key"),
            ))
        with self._connect() as conn:
            conn.execute("DELETE FROM cv_chunks WHERE user_id = ?", (user_id,))
            conn.executemany(
                """
                INSERT INTO cv_chunks (user_id, chunk_index, section, content, embedding,
                                       kind, entity, role, start_text, end_text, start_key, end_key)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def search_chunks(self, user_id: str, query_vector: Sequence[float], k: int = 4) -> List[ChunkHit]:
        """Most similar chunks first. Use this for open questions, not for listing every entry."""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_CHUNK_COLUMNS} FROM cv_chunks WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        if not rows:
            return []

        query = np.asarray(query_vector, dtype=np.float32)
        query = query / (np.linalg.norm(query) or 1.0)
        matrix = np.stack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1.0
        scores = (matrix / norms[:, None]) @ query

        top = np.argsort(-scores)[:k]
        return [_hit(rows[i], score=float(scores[i])) for i in top]

    def list_chunks(self, user_id: str, kind: str) -> List[ChunkHit]:
        """
        Every chunk of one kind, in a fixed order that does not depend on the query.
        Dated kinds (experience, education) are newest first, undated entries last.
        Other kinds keep the order they have in the CV.
        """
        if kind in DATED_KINDS:
            order = "COALESCE(end_key, '0000-00') DESC, COALESCE(start_key, '0000-00') DESC, chunk_index"
        else:
            order = "chunk_index"
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {_CHUNK_COLUMNS} FROM cv_chunks WHERE user_id = ? AND kind = ? ORDER BY {order}",
                (user_id, kind),
            ).fetchall()
        return [_hit(r, score=1.0) for r in rows]

    def cv_text(self, user_id: str) -> str:
        """All indexed CV text for a user, in CV order. Used to check values before filling them."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT content FROM cv_chunks WHERE user_id = ? ORDER BY chunk_index", (user_id,)
            ).fetchall()
        return "\n".join(r[0] for r in rows)

    def chunk_count(self, user_id: str) -> int:
        with self._connect() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM cv_chunks WHERE user_id = ?", (user_id,)
            ).fetchone()[0]

    # --- Answer memory ----------------------------------------------------

    def save_answer(self, user_id: str, question: str, answer: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO answer_memory (user_id, question_key, question, answer, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, question_key) DO UPDATE SET
                    answer = excluded.answer,
                    updated_at = excluded.updated_at
                """,
                (user_id, normalize_key(question), question, answer, _now()),
            )

    def list_answers(self, user_id: str) -> List[tuple]:
        """Returns (question, answer) pairs, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT question, answer FROM answer_memory WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        return [(q, a) for q, a in rows]

    def get_answer(self, user_id: str, question: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT answer FROM answer_memory WHERE user_id = ? AND question_key = ?",
                (user_id, normalize_key(question)),
            ).fetchone()
        return row[0] if row else None
