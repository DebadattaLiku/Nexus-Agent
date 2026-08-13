"""Tests for nexusai.rag.vector_store — deterministic, no network."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nexusai.rag.chunking import Chunk
from nexusai.rag.embeddings import HashingEmbeddingProvider
from nexusai.rag.vector_store import VectorStore

DIM = 32


def _chunk(chunk_id: str, filename: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, filename=filename, text=text, chunk_index=0)


def test_add_and_search_returns_most_similar_chunk_first():
    provider = HashingEmbeddingProvider(dim=DIM)
    chunks = [
        _chunk("a::0", "a.txt", "the sun is a star at the center of the solar system"),
        _chunk("b::0", "b.txt", "quarterly earnings beat analyst forecasts this quarter"),
        _chunk("c::0", "c.txt", "stars and the sun are studied in astronomy"),
    ]
    vectors = provider.embed([c.text for c in chunks])

    store = VectorStore(dim=DIM)
    store.add(chunks, vectors)

    query_vector = provider.embed(["tell me about the sun and stars"])[0]
    results = store.search(query_vector, top_k=2)

    assert len(results) == 2
    result_ids = [r.chunk_id for r in results]
    assert "b::0" not in result_ids  # least related chunk should not be top-2


def test_search_preserves_metadata():
    provider = HashingEmbeddingProvider(dim=DIM)
    chunks = [_chunk("doc.txt::0", "doc.txt", "some example chunk text")]
    vectors = provider.embed([c.text for c in chunks])

    store = VectorStore(dim=DIM)
    store.add(chunks, vectors)

    results = store.search(vectors[0], top_k=1)

    assert results[0].chunk_id == "doc.txt::0"
    assert results[0].filename == "doc.txt"
    assert results[0].text == "some example chunk text"


def test_search_on_empty_store_returns_empty_list():
    store = VectorStore(dim=DIM)
    query = np.zeros(DIM, dtype="float32")
    assert store.search(query, top_k=5) == []


def test_search_top_k_larger_than_store_size_is_clamped():
    provider = HashingEmbeddingProvider(dim=DIM)
    chunks = [_chunk("a::0", "a.txt", "one"), _chunk("b::0", "b.txt", "two")]
    vectors = provider.embed([c.text for c in chunks])

    store = VectorStore(dim=DIM)
    store.add(chunks, vectors)

    results = store.search(vectors[0], top_k=100)
    assert len(results) == 2


def test_search_rejects_non_positive_top_k():
    store = VectorStore(dim=DIM)
    with pytest.raises(ValueError):
        store.search(np.zeros(DIM, dtype="float32"), top_k=0)


def test_add_rejects_mismatched_vector_count():
    store = VectorStore(dim=DIM)
    chunks = [_chunk("a::0", "a.txt", "one"), _chunk("b::0", "b.txt", "two")]
    bad_vectors = np.zeros((1, DIM), dtype="float32")  # only 1 vector for 2 chunks
    with pytest.raises(ValueError):
        store.add(chunks, bad_vectors)


def test_save_and_load_round_trip_preserves_results(tmp_path):
    provider = HashingEmbeddingProvider(dim=DIM)
    chunks = [
        _chunk("a::0", "a.txt", "first chunk of text"),
        _chunk("b::0", "b.txt", "second chunk of text"),
    ]
    vectors = provider.embed([c.text for c in chunks])

    store = VectorStore(dim=DIM)
    store.add(chunks, vectors)

    index_path = tmp_path / "index.faiss"
    metadata_path = tmp_path / "index.meta.json"
    store.save(index_path, metadata_path)

    assert index_path.exists()
    assert metadata_path.exists()

    loaded = VectorStore.load(index_path, metadata_path)
    assert loaded.size == store.size

    results = loaded.search(vectors[0], top_k=1)
    assert results[0].chunk_id == "a::0"
    assert results[0].text == "first chunk of text"


def test_load_missing_files_raises():
    with pytest.raises(FileNotFoundError):
        VectorStore.load(
            index_path=Path("/tmp/does_not_exist.faiss"),
            metadata_path=Path("/tmp/does_not_exist.meta.json"),
        )


def test_exists_on_disk_false_when_not_saved(tmp_path):
    assert VectorStore.exists_on_disk(tmp_path / "a.faiss", tmp_path / "a.meta.json") is False
