"""
CV ingestion: PDF -> Markdown (Docling) -> sections -> LLM metadata per section
-> header-aware chunks -> embeddings, plus a structured profile extracted by the LLM.

Every chunk carries its section's metadata (kind, company or school, role, dates).
The label is also written at the start of the text that is embedded, so semantic
search sees it too. Lists such as "all my jobs" are answered with `list_chunks`
in a fixed order, not by similarity.

Used by the CLI (`python -m autofill_agent.cli ingest ...`).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from .config import INDEX_MODEL, INDEX_REASONING_EFFORT
from .cv_metadata import EntryLabel, date_key, describe, label_sections
from .llm import get_chat_model, get_embeddings
from .profile import CandidateProfile, extract_profile
from .storage import ChunkHit, Storage

HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
ENTRY_MAX_CHARS = 2500  # entries up to this length stay whole, so a list shows each one once


@dataclass
class IngestResult:
    user_id: str
    source: str
    chunk_count: int
    profile: CandidateProfile


def pdf_to_markdown(pdf_path: Path) -> str:
    # Imported lazily: Docling is heavy and only needed for ingestion.
    from docling.document_converter import DocumentConverter

    result = DocumentConverter().convert(str(pdf_path))
    return result.document.export_to_markdown()


def split_sections(markdown: str) -> List[Document]:
    """One document per heading section, headings kept in the text."""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,  # keep headings in the text so chunks carry their section
    )
    return header_splitter.split_text(markdown)


def split_markdown(markdown: str, source: str) -> List[Document]:
    """Section-aware chunks without metadata. Kept for callers that only need text."""
    size_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = size_splitter.split_documents(split_sections(markdown))
    for chunk in chunks:
        chunk.metadata["source"] = source
    return chunks


def _section_label(metadata: dict) -> Optional[str]:
    parts = [metadata.get(key) for key in ("h1", "h2", "h3")]
    label = " > ".join(p for p in parts if p)
    return label or None


def _meta_dict(entry: EntryLabel) -> dict:
    return {
        "kind": entry.kind,
        "entity": entry.entity,
        "role": entry.role,
        "start": entry.start,
        "end": entry.end,
        "start_key": date_key(entry.start),
        "end_key": date_key(entry.end),
    }


def index_markdown(
    markdown: str,
    user_id: str,
    source: str,
    storage: Storage,
    llm=None,
    embeddings=None,
) -> int:
    """Labels each section with one LLM call, chunks it, embeds it and stores the chunks."""
    sections = split_sections(markdown)
    if not sections:
        raise ValueError(f"No text could be extracted from {source}")

    llm = llm or get_chat_model(INDEX_MODEL, INDEX_REASONING_EFFORT)
    labels = label_sections(llm, [s.page_content for s in sections], markdown)

    # Sections of one entry (e.g. a company heading and its title/dates heading) become one document,
    # so each job is one entry in the index. Unlabelled sections stay on their own.
    groups: Dict[object, dict] = {}
    for i, section in enumerate(sections):
        label = labels.get(i)
        if label is None or label.kind == "other":
            key = ("single", i)
            label = label or EntryLabel(kind="other", sections=[i])
        else:
            # The model may label two headings of one job separately; identical fields mean one entry.
            key = (label.kind, label.entity, label.role, label.start, label.end)
        group = groups.setdefault(key, {"label": label, "first": section, "texts": []})
        group["texts"].append(section.page_content)
    documents: List[Document] = []
    for group in groups.values():
        first = group["first"]
        label = group["label"]
        documents.append(Document(
            page_content="\n\n".join(group["texts"]),
            metadata={**first.metadata, "meta": _meta_dict(label), "label": describe(label)},
        ))

    # A job or project is kept as one chunk, so a list shows it once. Only very long entries are split.
    size_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks: List[Document] = []
    for doc in documents:
        if len(doc.page_content) <= ENTRY_MAX_CHARS:
            chunks.append(doc)
        else:
            chunks.extend(size_splitter.split_documents([doc]))
    for chunk in chunks:
        chunk.metadata["source"] = source

    embeddings = embeddings or get_embeddings()
    embed_texts = [
        f"[{c.metadata['label']}]\n{c.page_content}" if c.metadata.get("label") else c.page_content
        for c in chunks
    ]
    vectors = embeddings.embed_documents(embed_texts)
    storage.replace_chunks(
        user_id,
        contents=[c.page_content for c in chunks],
        sections=[_section_label(c.metadata) for c in chunks],
        vectors=vectors,
        metadata=[c.metadata.get("meta") for c in chunks],
    )
    return len(chunks)


def ingest_cv(pdf_path: Path, user_id: str, storage: Optional[Storage] = None) -> IngestResult:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"CV not found: {pdf_path}")

    storage = storage or Storage()
    source = str(pdf_path)

    markdown = pdf_to_markdown(pdf_path)
    chunk_count = index_markdown(markdown, user_id=user_id, source=source, storage=storage)

    profile = extract_profile(get_chat_model(), markdown)
    storage.save_profile(user_id, profile, source=source)

    return IngestResult(user_id=user_id, source=source, chunk_count=chunk_count, profile=profile)


def search_cv(
    query: str,
    user_id: str,
    k: int = 4,
    storage: Optional[Storage] = None,
    kind: Optional[str] = None,
) -> List[ChunkHit]:
    """
    With `kind` (e.g. "experience"), returns every entry of that kind in a fixed order
    and ignores the query. Without it, returns the k passages most similar to the query.
    """
    storage = storage or Storage()
    if kind:
        return storage.list_chunks(user_id, kind)
    query_vector = get_embeddings().embed_query(query)
    return storage.search_chunks(user_id, query_vector, k=k)


def describe_hit(hit: ChunkHit) -> str:
    """Text the agent reads for one search result: the metadata line, then the CV text."""
    if not hit.kind:
        return hit.content
    dates = " - ".join(x for x in (hit.start, hit.end) if x)
    line = " | ".join(x for x in (hit.kind, hit.entity, hit.role, dates) if x)
    return f"[{line}]\n{hit.content}"
