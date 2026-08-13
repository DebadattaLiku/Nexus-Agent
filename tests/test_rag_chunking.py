"""Tests for nexusai.rag.chunking — no network, no external services."""

from __future__ import annotations

import pytest

from nexusai.rag.chunking import Chunk, chunk_document, chunk_documents, chunk_text
from nexusai.rag.ingestion import RawDocument


def test_chunk_text_splits_by_word_count():
    text = " ".join(f"word{i}" for i in range(10))

    chunks = chunk_text(text, chunk_size=4, chunk_overlap=0)

    assert chunks == [
        "word0 word1 word2 word3",
        "word4 word5 word6 word7",
        "word8 word9",
    ]


def test_chunk_text_applies_overlap():
    text = " ".join(f"w{i}" for i in range(6))

    chunks = chunk_text(text, chunk_size=4, chunk_overlap=2)

    # windows: [0:4], [2:6]
    assert chunks == ["w0 w1 w2 w3", "w2 w3 w4 w5"]


def test_chunk_text_short_document_is_single_chunk():
    chunks = chunk_text("just a few words", chunk_size=200, chunk_overlap=40)
    assert chunks == ["just a few words"]


def test_chunk_text_empty_string_returns_no_chunks():
    assert chunk_text("", chunk_size=100, chunk_overlap=10) == []
    assert chunk_text("   ", chunk_size=100, chunk_overlap=10) == []


def test_chunk_text_rejects_non_positive_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=0, chunk_overlap=0)


def test_chunk_text_rejects_negative_overlap():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=10, chunk_overlap=-1)


def test_chunk_text_rejects_overlap_gte_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=10, chunk_overlap=10)


def test_chunk_document_produces_stable_ids_and_indices():
    document = RawDocument(filename="doc.txt", text=" ".join(f"w{i}" for i in range(10)))

    chunks = chunk_document(document, chunk_size=4, chunk_overlap=0)

    assert all(isinstance(c, Chunk) for c in chunks)
    assert [c.chunk_id for c in chunks] == ["doc.txt::0", "doc.txt::1", "doc.txt::2"]
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert all(c.filename == "doc.txt" for c in chunks)


def test_chunk_documents_concatenates_all_documents_chunks():
    docs = [
        RawDocument(filename="a.txt", text="alpha beta"),
        RawDocument(filename="b.txt", text="gamma delta"),
    ]

    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=0)

    assert [c.filename for c in chunks] == ["a.txt", "b.txt"]
    assert [c.chunk_id for c in chunks] == ["a.txt::0", "b.txt::0"]
