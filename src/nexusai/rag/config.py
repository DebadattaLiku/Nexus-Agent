"""
RAG Configuration (Phase 3)
============================

Every tunable knob for the retrieval pipeline lives here, in one place,
instead of being hard-coded across ingestion/chunking/embeddings/vector
store code. `RAGConfig` holds plain defaults; `get_default_config()` is the
single place that overlays environment-variable overrides on top of those
defaults (same `.env` pattern used by `llm/provider.py`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class RAGConfig:
    """Explicit, immutable configuration for the whole RAG pipeline."""

    # -- ingestion --
    documents_dir: Path = REPO_ROOT / "data" / "documents"

    # -- chunking --
    chunk_size: int = 200          # target chunk size, in whitespace-split words
    chunk_overlap: int = 40        # words shared between consecutive chunks

    # -- embeddings --
    # "fastembed" = real local/free HF-model-backed embeddings (downloads a
    #   small ONNX model from Hugging Face on first use, then caches it on
    #   disk — no API key, no paid API, no network needed after the first
    #   run).
    # "hashing"   = deterministic, dependency-free feature-hashing embeddings.
    #   Used as an offline fallback and in tests, so the test suite never
    #   needs network access to a model hub.
    embedding_provider: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embedding_cache_dir: Path = REPO_ROOT / ".cache" / "fastembed"

    # -- retrieval --
    top_k: int = 4
    max_top_k: int = 20

    # -- vector index persistence --
    index_dir: Path = REPO_ROOT / "data" / "index"
    index_filename: str = "documents.faiss"
    metadata_filename: str = "documents.meta.json"

    @property
    def index_path(self) -> Path:
        return self.index_dir / self.index_filename

    @property
    def metadata_path(self) -> Path:
        return self.index_dir / self.metadata_filename


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def get_default_config() -> RAGConfig:
    """Build the default RAG configuration, overlaying `RAG_*` env vars."""
    base = RAGConfig()
    return replace(
        base,
        chunk_size=_int_env("RAG_CHUNK_SIZE", base.chunk_size),
        chunk_overlap=_int_env("RAG_CHUNK_OVERLAP", base.chunk_overlap),
        embedding_provider=os.environ.get("RAG_EMBEDDING_PROVIDER", base.embedding_provider),
        embedding_model=os.environ.get("RAG_EMBEDDING_MODEL", base.embedding_model),
        embedding_dim=_int_env("RAG_EMBEDDING_DIM", base.embedding_dim),
        top_k=_int_env("RAG_TOP_K", base.top_k),
    )
