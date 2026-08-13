"""
Vector Store (Phase 3)
========================

Thin, focused wrapper around a FAISS index that:

- adds embedded document chunks,
- retrieves the top-k most similar chunks for a query vector,
- saves/loads the index (+ its chunk metadata) to/from disk.

Vectors are expected to already be L2-normalized (both `EmbeddingProvider`
implementations do this), so a `faiss.IndexFlatIP` (inner product) gives
exact cosine-similarity search — no approximate-index tuning needed at this
data scale, and it keeps the implementation simple and fully deterministic.

Metadata (filename, chunk id, chunk text) is kept in a parallel Python list,
indexed by FAISS vector position, and persisted alongside the index as JSON
so `source filename` / `chunk id` / `chunk text` survive a save/load cycle.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np

from nexusai.rag.chunking import Chunk

logger = logging.getLogger("nexusai.rag.vector_store")


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk returned from a similarity search, with its score."""

    chunk_id: str
    filename: str
    text: str
    score: float


class VectorStore:
    """A FAISS `IndexFlatIP` plus the chunk metadata needed to interpret hits."""

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError("dim must be a positive integer")
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._metadata: list[Chunk] = []

    @property
    def size(self) -> int:
        return len(self._metadata)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Add embedded chunks to the index. `vectors` must be (len(chunks), dim)."""
        if len(chunks) == 0:
            return
        if vectors.shape != (len(chunks), self.dim):
            raise ValueError(
                f"vectors shape {vectors.shape} does not match "
                f"(len(chunks)={len(chunks)}, dim={self.dim})"
            )
        self._index.add(np.ascontiguousarray(vectors, dtype="float32"))
        self._metadata.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        """Return the top-k chunks most similar to `query_vector`."""
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if self.size == 0:
            return []

        query = np.ascontiguousarray(query_vector, dtype="float32").reshape(1, -1)
        k = min(top_k, self.size)
        scores, indices = self._index.search(query, k)

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self._metadata[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    filename=chunk.filename,
                    text=chunk.text,
                    score=float(score),
                )
            )
        return results

    def save(self, index_path: Path, metadata_path: Path) -> None:
        """Persist the FAISS index and its chunk metadata to disk."""
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(index_path))
        payload = {
            "dim": self.dim,
            "chunks": [asdict(chunk) for chunk in self._metadata],
        }
        metadata_path.write_text(json.dumps(payload), encoding="utf-8")
        logger.info(
            "Saved vector store (%d chunk(s)) to %s / %s", self.size, index_path, metadata_path
        )

    @classmethod
    def load(cls, index_path: Path, metadata_path: Path) -> "VectorStore":
        """Load a previously saved FAISS index and its chunk metadata."""
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"No saved vector store at {index_path} / {metadata_path}"
            )

        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        store = cls(dim=payload["dim"])
        store._index = faiss.read_index(str(index_path))
        store._metadata = [Chunk(**chunk) for chunk in payload["chunks"]]

        if store._index.ntotal != len(store._metadata):
            raise ValueError(
                "Corrupt vector store: FAISS index size "
                f"({store._index.ntotal}) does not match metadata size "
                f"({len(store._metadata)})"
            )
        return store

    @staticmethod
    def exists_on_disk(index_path: Path, metadata_path: Path) -> bool:
        return index_path.exists() and metadata_path.exists()
