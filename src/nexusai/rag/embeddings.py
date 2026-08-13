"""
Embeddings (Phase 3)
======================

Two `EmbeddingProvider` implementations, selected by
`RAGConfig.embedding_provider` so the concrete provider can be swapped
without touching `vector_store.py`, the MCP tool, or the agent:

- `FastEmbedProvider` (default, "fastembed") — real, local, free embeddings
  via the `fastembed` package, which runs Hugging Face sentence-embedding
  models (default: `BAAI/bge-small-en-v1.5`) locally through ONNX Runtime.
  No API key, no per-call cost, no torch dependency. The model weights are
  downloaded from Hugging Face once and cached under
  `RAGConfig.embedding_cache_dir`; every call after that is fully offline.

- `HashingEmbeddingProvider` ("hashing") — a deterministic, dependency-free
  bag-of-words feature-hashing embedding (the same technique behind
  scikit-learn's `HashingVectorizer`). It never downloads anything and never
  touches the network, which is what makes it the right choice for the test
  suite (deterministic, offline, reproducible) and a reasonable degraded-mode
  fallback if a model can't be downloaded (e.g. an offline deployment).

Both return unit-normalized vectors, so FAISS inner-product search
(`vector_store.py`) is equivalent to cosine similarity for either one.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Protocol

import numpy as np

from nexusai.rag.config import RAGConfig

logger = logging.getLogger("nexusai.rag.embeddings")

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class EmbeddingProvider(Protocol):
    """Interface every embedding provider must implement."""

    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts, returning an (n, dim) float32 array."""
        ...


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class HashingEmbeddingProvider:
    """
    Deterministic, offline, dependency-free bag-of-words embedding.

    Each token is hashed (SHA-256, so it's stable across processes and Python
    versions, unlike Python's salted `hash()`) into one of `dim` buckets; the
    resulting vector is a signed bag-of-words count vector, L2-normalized.
    Same text always produces exactly the same vector, and no network or
    model file is ever touched.
    """

    def __init__(self, dim: int = 384) -> None:
        if dim <= 0:
            raise ValueError("dim must be a positive integer")
        self.dim = dim

    def _embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype="float32")
        tokens = _TOKEN_RE.findall(text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        return vector

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        vectors = np.stack([self._embed_one(t) for t in texts]).astype("float32")
        return _normalize(vectors)


class FastEmbedProvider:
    """Real local/free embeddings backed by the `fastembed` package."""

    def __init__(self, model_name: str, dim: int, cache_dir: str | None = None) -> None:
        from fastembed import TextEmbedding  # imported lazily: heavy + optional

        self.dim = dim
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        logger.info("Loaded fastembed model %r", model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        vectors = np.array(list(self._model.embed(texts)), dtype="float32")
        return _normalize(vectors)


def get_embedding_provider(config: RAGConfig) -> EmbeddingProvider:
    """Build the embedding provider selected by `config.embedding_provider`."""
    name = config.embedding_provider.strip().lower()

    if name == "hashing":
        return HashingEmbeddingProvider(dim=config.embedding_dim)

    if name == "fastembed":
        config.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
        return FastEmbedProvider(
            model_name=config.embedding_model,
            dim=config.embedding_dim,
            cache_dir=str(config.embedding_cache_dir),
        )

    raise ValueError(
        f"Unknown embedding_provider: {name!r}. Supported values are 'fastembed' or 'hashing'."
    )
