"""
Document loading and chunking module.

Reads Markdown files from a data directory, wraps them in lightweight
dataclasses, and splits them into overlapping chunks using LangChain's
RecursiveCharacterTextSplitter with Markdown-aware separators.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data models
# ──────────────────────────────────────────────
@dataclass
class Document:
    """Represents a single raw document loaded from disk."""

    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class DocumentChunk:
    """Represents one chunk of a document after splitting."""

    content: str
    metadata: dict = field(default_factory=dict)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def _extract_doc_id(filename: str) -> str:
    """Extract a numeric/alphanumeric document ID from a filename.

    Examples
    --------
    >>> _extract_doc_id("01_early_mars_missions.md")
    '01'
    >>> _extract_doc_id("chapter3_overview.md")
    'chapter3_overview'
    """
    match = re.match(r"^(\d+)", filename)
    if match:
        return match.group(1)
    # Fallback: use the stem (filename without extension)
    return Path(filename).stem


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────
def load_documents(data_dir: str) -> list[Document]:
    """Read all ``.md`` files from *data_dir* and return a list of Documents.

    Parameters
    ----------
    data_dir:
        Path to the directory containing Markdown documents.

    Returns
    -------
    list[Document]
        One :class:`Document` per file, sorted by filename.

    Raises
    ------
    FileNotFoundError
        If *data_dir* does not exist.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    md_files = sorted(data_path.glob("*.md"))
    if not md_files:
        logger.warning("No .md files found in %s", data_path)
        return []

    documents: list[Document] = []
    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8")
        doc_id = _extract_doc_id(md_file.name)
        metadata = {
            "source": md_file.name,
            "doc_id": doc_id,
        }
        documents.append(Document(content=content, metadata=metadata))
        logger.info(
            "Loaded document: %s (doc_id=%s, %d chars)",
            md_file.name, doc_id, len(content),
        )

    logger.info("Loaded %d document(s) from %s", len(documents), data_path)
    return documents


def chunk_documents(
    documents: list[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Split *documents* into overlapping chunks.

    Uses :class:`RecursiveCharacterTextSplitter` configured with
    Markdown-aware separators so that headers and paragraph boundaries
    are preferred split points.

    Parameters
    ----------
    documents:
        Documents to split.
    chunk_size:
        Maximum chunk length in characters.
    chunk_overlap:
        Overlap between consecutive chunks in characters.

    Returns
    -------
    list[DocumentChunk]
        Flat list of chunks across all documents.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    all_chunks: list[DocumentChunk] = []

    for doc in documents:
        texts = splitter.split_text(doc.content)
        for idx, text in enumerate(texts):
            chunk_metadata = {
                **doc.metadata,
                "chunk_index": idx,
            }
            all_chunks.append(DocumentChunk(content=text, metadata=chunk_metadata))

        logger.debug(
            "Document %s → %d chunk(s)",
            doc.metadata.get("source", "unknown"),
            len(texts),
        )

    logger.info(
        "Created %d chunk(s) from %d document(s)",
        len(all_chunks), len(documents),
    )
    return all_chunks
