"""
RAG Pipeline (Phase 3)
========================

Wires ingestion -> chunking -> embeddings -> FAISS vector store into one
`RAGPipeline` object with a single `retrieve(query, top_k)` method. This is
the only piece of RAG code the MCP document server talks to.

Performance / caching strategy
-------------------------------
Building the index (ingest + chunk + embed all documents) is the expensive
step, so `RAGPipeline.load_or_build()` persists the FAISS index + chunk
metadata to disk (`RAGConfig.index_path` / `metadata_path`) and only rebuilds
when needed:

- no saved index yet -> build once, save.
- a saved index exists but its fingerprint (embedding provider/model/dim,
  chunk size/overlap, and the ingested document set) no longer matches the
  current config or `documents_dir` contents -> rebuild, save.
- otherwise -> load the saved index straight off disk; no re-embedding.

This means the MCP server does not regenerate embeddings on every restart —
only when the documents or chunking/embedding configuration actually change.
"""

from __future__ import annotations

import hashlib
import json
import logging

from nexusai.rag.chunking import chunk_documents
from nexusai.rag.config import RAGConfig, get_default_config
from nexusai.rag.embeddings import EmbeddingProvider, get_embedding_provider
from nexusai.rag.ingestion import load_documents
from nexusai.rag.vector_store import RetrievedChunk, VectorStore

logger = logging.getLogger("nexusai.rag.pipeline")


def _fingerprint(config: RAGConfig) -> str:
    """
    A short hash identifying "what the index was built from": the chunking
    config, the embedding provider/model/dim, and the ingested documents'
    filenames + contents. If any of that changes, a saved index is stale.
    """
    documents = load_documents(config.documents_dir)
    payload = {
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "embedding_provider": config.embedding_provider,
        "embedding_model": config.embedding_model,
        "embedding_dim": config.embedding_dim,
        "documents": sorted((doc.filename, doc.text) for doc in documents),
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class RAGPipeline:
    """Ingest -> chunk -> embed -> retrieve, backed by a FAISS `VectorStore`."""

    def __init__(
        self,
        config: RAGConfig,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        fingerprint: str,
    ) -> None:
        self.config = config
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.fingerprint = fingerprint

    # -- construction --------------------------------------------------

    @classmethod
    def build(
        cls, config: RAGConfig | None = None, embedding_provider: EmbeddingProvider | None = None
    ) -> "RAGPipeline":
        """Ingest every document, chunk it, embed it, and build a fresh index."""
        config = config or get_default_config()
        provider = embedding_provider or get_embedding_provider(config)

        documents = load_documents(config.documents_dir)
        chunks = chunk_documents(documents, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)

        store = VectorStore(dim=provider.dim)
        if chunks:
            vectors = provider.embed([chunk.text for chunk in chunks])
            store.add(chunks, vectors)

        logger.info(
            "Built RAG index: %d document(s) -> %d chunk(s)", len(documents), len(chunks)
        )
        return cls(config=config, embedding_provider=provider, vector_store=store, fingerprint=_fingerprint(config))

    def save(self) -> None:
        self.vector_store.save(self.config.index_path, self.config.metadata_path)
        fingerprint_path = self.config.index_dir / "fingerprint.txt"
        fingerprint_path.write_text(self.fingerprint, encoding="utf-8")

    @classmethod
    def load_or_build(
        cls, config: RAGConfig | None = None, embedding_provider: EmbeddingProvider | None = None
    ) -> "RAGPipeline":
        """
        Load a previously saved index if it's still fresh for `config`;
        otherwise build (and persist) a new one. See module docstring for
        the freshness/caching rule.
        """
        config = config or get_default_config()
        current_fingerprint = _fingerprint(config)
        fingerprint_path = config.index_dir / "fingerprint.txt"

        saved_index = VectorStore.exists_on_disk(config.index_path, config.metadata_path)
        saved_fingerprint = fingerprint_path.read_text(encoding="utf-8").strip() if fingerprint_path.exists() else None

        if saved_index and saved_fingerprint == current_fingerprint:
            logger.info("Loading cached RAG index from %s", config.index_dir)
            store = VectorStore.load(config.index_path, config.metadata_path)
            provider = embedding_provider or get_embedding_provider(config)
            return cls(config=config, embedding_provider=provider, vector_store=store, fingerprint=current_fingerprint)

        logger.info("No fresh cached RAG index found; building one now.")
        pipeline = cls.build(config=config, embedding_provider=embedding_provider)
        pipeline.save()
        return pipeline

    # -- retrieval --------------------------------------------------------

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return the top-k chunks most semantically similar to `query`."""
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        top_k = min(top_k, self.config.max_top_k)
        query_vector = self.embedding_provider.embed([query])[0]
        return self.vector_store.search(query_vector, top_k=top_k)
