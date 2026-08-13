"""
Document Ingestion (Phase 3)
==============================

Turns files on disk into `RawDocument` objects the chunking stage can
consume. Only plain `.txt` loading is implemented today (matching what
Phase 1/2 already store in `data/documents/`), but the loader is looked up
by file extension through a small registry, so adding a PDF/Word loader
later means registering a new function here — nothing else in the RAG
pipeline needs to change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger("nexusai.rag.ingestion")


@dataclass(frozen=True)
class RawDocument:
    """One ingested document, before chunking."""

    filename: str
    text: str


def _load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# Extension -> loader function. Register new loaders here (e.g. ".pdf") to
# extend ingestion without touching `load_documents()` or any downstream
# chunking/embedding/index code.
LOADERS: dict[str, Callable[[Path], str]] = {
    ".txt": _load_txt,
}


def iter_document_paths(documents_dir: Path) -> list[Path]:
    """All ingestible files directly inside `documents_dir`, sorted for determinism."""
    if not documents_dir.exists():
        logger.warning("Documents directory does not exist: %s", documents_dir)
        return []

    paths = [
        path
        for path in documents_dir.iterdir()
        if path.is_file() and path.suffix.lower() in LOADERS
    ]
    return sorted(paths, key=lambda p: p.name)


def load_document(path: Path) -> RawDocument:
    """Load a single file into a `RawDocument`, using the loader for its extension."""
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"No loader registered for file type: {path.suffix!r}")
    return RawDocument(filename=path.name, text=loader(path))


def load_documents(documents_dir: Path) -> list[RawDocument]:
    """Load every ingestible document directly inside `documents_dir`."""
    documents: list[RawDocument] = []
    for path in iter_document_paths(documents_dir):
        try:
            documents.append(load_document(path))
        except OSError as exc:
            logger.warning("Could not read %s: %s", path.name, exc)
    logger.info("Ingested %d document(s) from %s", len(documents), documents_dir)
    return documents
