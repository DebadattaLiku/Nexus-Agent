"""
Tests for nexusai.rag.embeddings.

Only HashingEmbeddingProvider is exercised here: it's deterministic and
never touches the network or a model hub, which is exactly what the test
suite requires (no internet access, no paid APIs). FastEmbedProvider is
covered by its factory wiring in get_embedding_provider() with an invalid
name, and is otherwise a thin, mostly-untestable-offline wrapper around the
`fastembed` package.
"""

from __future__ import annotations

import numpy as np
import pytest

from nexusai.rag.config import RAGConfig
from nexusai.rag.embeddings import HashingEmbeddingProvider, get_embedding_provider


def test_embed_returns_correct_shape():
    provider = HashingEmbeddingProvider(dim=64)
    vectors = provider.embed(["hello world", "another text", "a third one"])
    assert vectors.shape == (3, 64)
    assert vectors.dtype == np.float32


def test_embed_is_deterministic_across_calls():
    provider = HashingEmbeddingProvider(dim=64)
    v1 = provider.embed(["the quick brown fox"])
    v2 = provider.embed(["the quick brown fox"])
    assert np.allclose(v1, v2)


def test_embed_vectors_are_unit_normalized():
    provider = HashingEmbeddingProvider(dim=32)
    vectors = provider.embed(["some text with several distinct words here"])
    norm = np.linalg.norm(vectors[0])
    assert norm == pytest.approx(1.0, abs=1e-5)


def test_embed_empty_list_returns_empty_array():
    provider = HashingEmbeddingProvider(dim=16)
    vectors = provider.embed([])
    assert vectors.shape == (0, 16)


def test_similar_texts_are_more_similar_than_unrelated_texts():
    provider = HashingEmbeddingProvider(dim=256)
    a = provider.embed(["cats and dogs are common household pets"])[0]
    b = provider.embed(["dogs and cats make popular household pets"])[0]
    c = provider.embed(["quarterly revenue exceeded analyst expectations"])[0]

    sim_ab = float(np.dot(a, b))
    sim_ac = float(np.dot(a, c))

    assert sim_ab > sim_ac


def test_dim_must_be_positive():
    with pytest.raises(ValueError):
        HashingEmbeddingProvider(dim=0)


def test_get_embedding_provider_builds_hashing_provider():
    config = RAGConfig(embedding_provider="hashing", embedding_dim=48)
    provider = get_embedding_provider(config)
    assert isinstance(provider, HashingEmbeddingProvider)
    assert provider.dim == 48


def test_get_embedding_provider_rejects_unknown_name():
    config = RAGConfig(embedding_provider="not-a-real-provider")
    with pytest.raises(ValueError):
        get_embedding_provider(config)
