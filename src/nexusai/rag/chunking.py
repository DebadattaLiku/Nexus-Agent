"""
Text Chunking (Phase 3)
=========================

Splits document text into overlapping windows small enough to embed
meaningfully and retrieve precisely. Chunking is word-based (splits on
whitespace) rather than character-based, so `chunk_size`/`chunk_overlap`
behave predictably across documents regardless of line length.

No sizes are hard-coded here — every call takes explicit `chunk_size` and
`chunk_overlap` values (see `rag/config.py` for where the defaults live).
"""

from __future__ import annotations

from dataclasses import dataclass

from nexusai.rag.ingestion import RawDocument


@dataclass(frozen=True)
class Chunk:
    """One chunk of a document, ready to embed and index."""

    chunk_id: str
    filename: str
    text: str
    chunk_index: int  # position of this chunk within its source document


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split `text` into a list of overlapping word-windows.

    Args:
        text: Text to split.
        chunk_size: Target number of whitespace-split words per chunk. Must
            be a positive integer.
        chunk_overlap: Number of words shared between consecutive chunks.
            Must be >= 0 and strictly less than `chunk_size` (otherwise the
            window never advances and chunking would loop forever).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must not be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    stride = chunk_size - chunk_overlap
    chunks: list[str] = []
    start = 0
    while start < len(words):
        window = words[start : start + chunk_size]
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
        start += stride

    return chunks


def chunk_document(document: RawDocument, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Chunk one `RawDocument` into a list of `Chunk` objects with stable IDs."""
    pieces = chunk_text(document.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [
        Chunk(
            chunk_id=f"{document.filename}::{i}",
            filename=document.filename,
            text=piece,
            chunk_index=i,
        )
        for i, piece in enumerate(pieces)
    ]


def chunk_documents(
    documents: list[RawDocument], chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """Chunk a list of documents, concatenating every document's chunks in order."""
    all_chunks: list[Chunk] = []
    for document in documents:
        all_chunks.extend(chunk_document(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    return all_chunks
