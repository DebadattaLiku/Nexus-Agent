"""
Tests for nexusai.rag.pipeline.

Everything here uses embedding_provider="hashing" (deterministic, offline)
and tmp_path-scoped documents_dir/index_dir, so no test ever touches the
network, a model hub, or the real repo's data/ directory.
"""

from __future__ import annotations

import pytest

from nexusai.rag.config import RAGConfig
from nexusai.rag.pipeline import RAGPipeline, _fingerprint


def _make_config(tmp_path, **overrides) -> RAGConfig:
    defaults = dict(
        documents_dir=tmp_path / "documents",
        index_dir=tmp_path / "index",
        chunk_size=20,
        chunk_overlap=4,
        embedding_provider="hashing",
        embedding_dim=32,
        top_k=3,
    )
    defaults.update(overrides)
    return RAGConfig(**defaults)


def _write_docs(documents_dir):
    documents_dir.mkdir(parents=True, exist_ok=True)
    (documents_dir / "sun.txt").write_text(
        "The sun is the star at the center of the solar system. "
        "It provides the energy that sustains life on Earth.",
        encoding="utf-8",
    )
    (documents_dir / "finance.txt").write_text(
        "Quarterly revenue exceeded analyst expectations this period, "
        "driven by strong demand in the enterprise segment.",
        encoding="utf-8",
    )


def test_build_ingests_chunks_and_embeds_all_documents(tmp_path):
    config = _make_config(tmp_path)
    _write_docs(config.documents_dir)

    pipeline = RAGPipeline.build(config)

    assert pipeline.vector_store.size >= 2  # at least one chunk per document


def test_build_with_no_documents_produces_empty_index(tmp_path):
    config = _make_config(tmp_path)
    config.documents_dir.mkdir(parents=True, exist_ok=True)

    pipeline = RAGPipeline.build(config)

    assert pipeline.vector_store.size == 0


def test_retrieve_returns_relevant_chunks_with_metadata(tmp_path):
    config = _make_config(tmp_path)
    _write_docs(config.documents_dir)
    pipeline = RAGPipeline.build(config)

    results = pipeline.retrieve("tell me about the sun and solar energy", top_k=2)

    assert len(results) > 0
    assert any(r.filename == "sun.txt" for r in results)
    for r in results:
        assert r.text
        assert r.chunk_id


def test_retrieve_rejects_empty_query(tmp_path):
    config = _make_config(tmp_path)
    _write_docs(config.documents_dir)
    pipeline = RAGPipeline.build(config)

    with pytest.raises(ValueError):
        pipeline.retrieve("   ", top_k=2)


def test_retrieve_rejects_non_positive_top_k(tmp_path):
    config = _make_config(tmp_path)
    _write_docs(config.documents_dir)
    pipeline = RAGPipeline.build(config)

    with pytest.raises(ValueError):
        pipeline.retrieve("the sun", top_k=0)


def test_retrieve_top_k_is_capped_at_max_top_k(tmp_path):
    config = _make_config(tmp_path, max_top_k=1)
    _write_docs(config.documents_dir)
    pipeline = RAGPipeline.build(config)

    results = pipeline.retrieve("the sun", top_k=50)
    assert len(results) <= 1


def test_save_then_load_or_build_reuses_cached_index_without_rebuilding(tmp_path):
    from unittest.mock import patch

    config = _make_config(tmp_path)
    _write_docs(config.documents_dir)

    built = RAGPipeline.build(config)
    built.save()

    with patch.object(RAGPipeline, "build", wraps=RAGPipeline.build) as build_spy:
        loaded = RAGPipeline.load_or_build(config)
        build_spy.assert_not_called()  # cached index reused, build() never called again

    assert loaded.vector_store.size == built.vector_store.size


def test_load_or_build_rebuilds_when_documents_change(tmp_path):
    config = _make_config(tmp_path)
    _write_docs(config.documents_dir)

    first = RAGPipeline.load_or_build(config)
    first_size = first.vector_store.size

    (config.documents_dir / "extra.txt").write_text(
        "An entirely new document added after the index was built.",
        encoding="utf-8",
    )

    second = RAGPipeline.load_or_build(config)

    assert second.vector_store.size > first_size


def test_load_or_build_rebuilds_when_chunk_config_changes(tmp_path):
    config = _make_config(tmp_path, chunk_size=20, chunk_overlap=4)
    _write_docs(config.documents_dir)

    first = RAGPipeline.load_or_build(config)

    different_chunking = _make_config(
        tmp_path, chunk_size=6, chunk_overlap=1, index_dir=config.index_dir
    )
    second = RAGPipeline.load_or_build(different_chunking)

    # smaller chunk_size on the same documents should produce more chunks
    assert second.vector_store.size > first.vector_store.size


def test_fingerprint_changes_when_documents_change(tmp_path):
    config = _make_config(tmp_path)
    _write_docs(config.documents_dir)

    fp1 = _fingerprint(config)
    (config.documents_dir / "sun.txt").write_text("changed content", encoding="utf-8")
    fp2 = _fingerprint(config)

    assert fp1 != fp2


def test_fingerprint_stable_for_unchanged_config_and_documents(tmp_path):
    config = _make_config(tmp_path)
    _write_docs(config.documents_dir)

    assert _fingerprint(config) == _fingerprint(config)
