"""
Tests for nexusai.rag.evaluation.

Metric functions are tested directly with plain lists (no I/O, no
randomness). The end-to-end `evaluate_retrieval()` test uses
embedding_provider="hashing" (deterministic, offline, no model download,
same convention as tests/test_rag_pipeline.py) with a tmp_path-scoped
documents_dir/index_dir, so it never touches the network, a model hub, or
the real repo's data/ directory.
"""

from __future__ import annotations

import json

import pytest

from nexusai.rag.config import RAGConfig
from nexusai.rag.evaluation import (
    EvalExample,
    evaluate_retrieval,
    hit_at_k,
    load_dataset,
    reciprocal_rank,
    recall_at_k,
)
from nexusai.rag.pipeline import RAGPipeline

# ---------------------------------------------------------------------------
# Metric unit tests
# ---------------------------------------------------------------------------


def test_hit_at_k_true_when_expected_source_in_top_k():
    retrieved = ["a.txt", "b.txt", "c.txt"]
    assert hit_at_k(retrieved, ["b.txt"], k=3) == 1.0


def test_hit_at_k_false_when_expected_source_outside_top_k():
    retrieved = ["a.txt", "b.txt", "c.txt"]
    assert hit_at_k(retrieved, ["c.txt"], k=2) == 0.0


def test_hit_at_k_false_when_expected_source_never_retrieved():
    retrieved = ["a.txt", "b.txt"]
    assert hit_at_k(retrieved, ["z.txt"], k=2) == 0.0


def test_hit_at_k_rejects_non_positive_k():
    with pytest.raises(ValueError):
        hit_at_k(["a.txt"], ["a.txt"], k=0)


def test_recall_at_k_full_recall_when_all_expected_present():
    retrieved = ["a.txt", "b.txt", "c.txt"]
    assert recall_at_k(retrieved, ["a.txt", "b.txt"], k=3) == 1.0


def test_recall_at_k_partial_recall_when_some_expected_missing():
    retrieved = ["a.txt", "x.txt", "y.txt"]
    # only 1 of 2 expected sources shows up in the top-k
    assert recall_at_k(retrieved, ["a.txt", "b.txt"], k=3) == pytest.approx(0.5)


def test_recall_at_k_zero_when_none_present():
    retrieved = ["x.txt", "y.txt"]
    assert recall_at_k(retrieved, ["a.txt"], k=2) == 0.0


def test_recall_at_k_respects_k_cutoff():
    retrieved = ["x.txt", "y.txt", "a.txt"]  # a.txt only shows up at rank 3
    assert recall_at_k(retrieved, ["a.txt"], k=2) == 0.0
    assert recall_at_k(retrieved, ["a.txt"], k=3) == 1.0


def test_recall_at_k_equals_hit_at_k_for_single_expected_source():
    retrieved = ["a.txt", "b.txt", "c.txt"]
    for expected in (["a.txt"], ["z.txt"], ["c.txt"]):
        assert recall_at_k(retrieved, expected, k=3) == hit_at_k(retrieved, expected, k=3)


def test_reciprocal_rank_is_one_over_rank_of_first_relevant_hit():
    retrieved = ["x.txt", "a.txt", "y.txt"]
    assert reciprocal_rank(retrieved, ["a.txt"], k=3) == pytest.approx(1 / 2)


def test_reciprocal_rank_uses_first_relevant_hit_when_multiple_expected():
    retrieved = ["x.txt", "y.txt", "a.txt", "b.txt"]
    # "b.txt" is also relevant but "a.txt" (rank 3) comes first
    assert reciprocal_rank(retrieved, ["a.txt", "b.txt"], k=4) == pytest.approx(1 / 3)


def test_reciprocal_rank_one_when_first_result_is_relevant():
    retrieved = ["a.txt", "x.txt"]
    assert reciprocal_rank(retrieved, ["a.txt"], k=2) == 1.0


def test_reciprocal_rank_zero_when_no_relevant_result_within_k():
    retrieved = ["x.txt", "y.txt", "a.txt"]
    assert reciprocal_rank(retrieved, ["a.txt"], k=2) == 0.0


def test_reciprocal_rank_rejects_non_positive_k():
    with pytest.raises(ValueError):
        reciprocal_rank(["a.txt"], ["a.txt"], k=0)


# ---------------------------------------------------------------------------
# EvalExample validation
# ---------------------------------------------------------------------------


def test_eval_example_rejects_empty_query():
    with pytest.raises(ValueError):
        EvalExample(query="   ", expected_sources=("a.txt",))


def test_eval_example_rejects_empty_expected_sources():
    with pytest.raises(ValueError):
        EvalExample(query="some query", expected_sources=())


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def test_load_dataset_parses_examples(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "examples": [
                    {"query": "what is the sun", "expected_sources": ["sun.txt"]},
                    {"query": "quarterly revenue", "expected_sources": ["finance.txt"]},
                ]
            }
        ),
        encoding="utf-8",
    )

    examples = load_dataset(dataset_path)

    assert examples == [
        EvalExample(query="what is the sun", expected_sources=("sun.txt",)),
        EvalExample(query="quarterly revenue", expected_sources=("finance.txt",)),
    ]


def test_default_dataset_file_loads_and_is_non_empty():
    examples = load_dataset()  # default path: data/eval/rag_eval_dataset.json
    assert len(examples) > 0
    for example in examples:
        assert example.query
        assert example.expected_sources


# ---------------------------------------------------------------------------
# End-to-end: evaluate_retrieval() against a real (hashing-backed) pipeline
# ---------------------------------------------------------------------------


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


def test_evaluate_retrieval_scores_each_example(tmp_path):
    config = _make_config(tmp_path)
    _write_docs(config.documents_dir)
    pipeline = RAGPipeline.build(config)

    examples = [
        EvalExample(query="tell me about the sun and solar energy", expected_sources=("sun.txt",)),
        EvalExample(query="quarterly revenue and enterprise demand", expected_sources=("finance.txt",)),
    ]

    report = evaluate_retrieval(pipeline, examples, k=2)

    assert report.k == 2
    assert len(report.results) == 2
    for result in report.results:
        assert result.retrieved_filenames  # something was retrieved
        assert result.hit in (0.0, 1.0)
        assert 0.0 <= result.recall <= 1.0
        assert 0.0 <= result.reciprocal_rank <= 1.0


def test_evaluate_retrieval_is_deterministic_across_runs(tmp_path):
    config = _make_config(tmp_path)
    _write_docs(config.documents_dir)
    pipeline = RAGPipeline.build(config)

    examples = [
        EvalExample(query="tell me about the sun and solar energy", expected_sources=("sun.txt",)),
        EvalExample(query="quarterly revenue and enterprise demand", expected_sources=("finance.txt",)),
    ]

    report_a = evaluate_retrieval(pipeline, examples, k=2)
    report_b = evaluate_retrieval(pipeline, examples, k=2)

    assert report_a.mean_hit_at_k == report_b.mean_hit_at_k
    assert report_a.mean_recall_at_k == report_b.mean_recall_at_k
    assert report_a.mrr == report_b.mrr
    assert [r.retrieved_filenames for r in report_a.results] == [
        r.retrieved_filenames for r in report_b.results
    ]


def test_evaluate_retrieval_perfect_score_when_queries_clearly_match_their_document(tmp_path):
    config = _make_config(tmp_path)
    _write_docs(config.documents_dir)
    pipeline = RAGPipeline.build(config)

    examples = [
        EvalExample(query="tell me about the sun and solar energy", expected_sources=("sun.txt",)),
        EvalExample(query="quarterly revenue and enterprise demand", expected_sources=("finance.txt",)),
    ]

    report = evaluate_retrieval(pipeline, examples, k=2)

    assert report.mean_hit_at_k == 1.0
    assert report.mean_recall_at_k == 1.0
    assert report.mrr == 1.0


def test_evaluate_retrieval_rejects_non_positive_k(tmp_path):
    config = _make_config(tmp_path)
    _write_docs(config.documents_dir)
    pipeline = RAGPipeline.build(config)

    with pytest.raises(ValueError):
        evaluate_retrieval(pipeline, [EvalExample(query="q", expected_sources=("sun.txt",))], k=0)


def test_evaluate_retrieval_scores_zero_for_unmatched_expected_source(tmp_path):
    config = _make_config(tmp_path)
    _write_docs(config.documents_dir)
    pipeline = RAGPipeline.build(config)

    examples = [EvalExample(query="tell me about the sun", expected_sources=("nonexistent.txt",))]
    report = evaluate_retrieval(pipeline, examples, k=2)

    assert report.mean_hit_at_k == 0.0
    assert report.mean_recall_at_k == 0.0
    assert report.mrr == 0.0


def test_format_report_includes_metrics_and_queries(tmp_path):
    config = _make_config(tmp_path)
    _write_docs(config.documents_dir)
    pipeline = RAGPipeline.build(config)

    examples = [EvalExample(query="tell me about the sun", expected_sources=("sun.txt",))]
    report = evaluate_retrieval(pipeline, examples, k=2)

    text = report.format_report()
    assert "Hit@2" in text
    assert "Recall@2" in text
    assert "MRR" in text
    assert "tell me about the sun" in text
