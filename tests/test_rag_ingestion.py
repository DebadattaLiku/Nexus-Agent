"""Tests for nexusai.rag.ingestion — no network, no external services."""

from __future__ import annotations

from nexusai.rag.ingestion import iter_document_paths, load_document, load_documents


def test_load_documents_reads_all_txt_files(tmp_path):
    (tmp_path / "a.txt").write_text("Hello world.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Second document.", encoding="utf-8")

    documents = load_documents(tmp_path)

    filenames = {d.filename for d in documents}
    assert filenames == {"a.txt", "b.txt"}


def test_load_documents_is_sorted_for_determinism(tmp_path):
    (tmp_path / "zeta.txt").write_text("z", encoding="utf-8")
    (tmp_path / "alpha.txt").write_text("a", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert [d.filename for d in documents] == ["alpha.txt", "zeta.txt"]


def test_load_documents_ignores_non_registered_extensions(tmp_path):
    (tmp_path / "keep.txt").write_text("keep me", encoding="utf-8")
    (tmp_path / "skip.bin").write_bytes(b"\x00\x01")

    documents = load_documents(tmp_path)

    assert [d.filename for d in documents] == ["keep.txt"]


def test_load_documents_on_missing_directory_returns_empty(tmp_path):
    missing = tmp_path / "does_not_exist"
    assert load_documents(missing) == []


def test_load_document_preserves_content(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text("exact content here", encoding="utf-8")

    doc = load_document(path)

    assert doc.filename == "doc.txt"
    assert doc.text == "exact content here"


def test_load_document_unregistered_extension_raises(tmp_path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4")

    try:
        load_document(path)
        assert False, "expected ValueError for unregistered extension"
    except ValueError:
        pass


def test_iter_document_paths_sorted_and_filtered(tmp_path):
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "ignore.md").write_text("ignored", encoding="utf-8")

    paths = iter_document_paths(tmp_path)

    assert [p.name for p in paths] == ["a.txt", "b.txt"]
