"""
RAG Retrieval Evaluation (Phase 6 - eval add-on)
==================================================

A small, self-contained module for measuring *retrieval* quality of the
existing `RAGPipeline` (see `nexusai/rag/pipeline.py`) against a small,
deterministic evaluation dataset.

This module does NOT touch ingestion, chunking, embeddings, the vector
store, the RAG pipeline, the MCP servers, the agents, or Groq. It only
*calls* `RAGPipeline.retrieve()` (read-only) and scores the results.

Dataset format
--------------
A JSON file with the shape:

    {
      "examples": [
        {"query": "...", "expected_sources": ["some_doc.txt"]},
        ...
      ]
    }

`expected_sources` is a list of filenames (as they appear in
`RetrievedChunk.filename`) that count as a correct/relevant retrieval for
that query. Most examples have exactly one, but the metrics below support
more than one relevant document per query.

The default dataset lives at `data/eval/rag_eval_dataset.json`.

Metrics
-------
For a query with retrieved filenames (ranked, top-K) and a set of expected
(relevant) source filenames:

- Hit@K:    1.0 if at least one expected source appears in the top-K
            retrieved filenames, else 0.0. Averaged over all queries.
- Recall@K: |top-K retrieved filenames ∩ expected sources| / |expected
            sources|. Averaged over all queries. Equivalent to Hit@K when
            each query has exactly one expected source.
- MRR:      1 / rank of the first retrieved chunk whose filename is an
            expected source (rank counted within the top-K list, starting
            at 1); 0.0 if none of the top-K are relevant. Averaged over all
            queries (this is "MRR@K").

Everything here is pure/deterministic given deterministic retrieval (e.g.
`embedding_provider="hashing"`, used by the test suite and available as a
network-free option for the CLI too).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from nexusai.rag.config import RAGConfig, get_default_config
from nexusai.rag.pipeline import RAGPipeline

logger = logging.getLogger("nexusai.rag.evaluation")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "eval" / "rag_eval_dataset.json"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalExample:
    """One evaluation case: a query and the source document(s) that should
    be retrieved for it."""

    query: str
    expected_sources: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.query or not self.query.strip():
            raise ValueError("EvalExample.query must not be empty")
        if not self.expected_sources:
            raise ValueError("EvalExample.expected_sources must not be empty")


def load_dataset(path: Path | None = None) -> list[EvalExample]:
    """Load evaluation examples from a JSON file (see module docstring for
    the expected shape)."""
    path = path or DEFAULT_DATASET_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    examples = payload.get("examples", [])
    return [
        EvalExample(query=item["query"], expected_sources=tuple(item["expected_sources"]))
        for item in examples
    ]


# ---------------------------------------------------------------------------
# Metrics (pure functions over a ranked list of retrieved filenames)
# ---------------------------------------------------------------------------


def hit_at_k(retrieved_filenames: Sequence[str], expected_sources: Iterable[str], k: int) -> float:
    """1.0 if any expected source is present in the top-k retrieved
    filenames, else 0.0."""
    if k <= 0:
        raise ValueError("k must be a positive integer")
    top_k = set(retrieved_filenames[:k])
    return 1.0 if top_k & set(expected_sources) else 0.0


def recall_at_k(retrieved_filenames: Sequence[str], expected_sources: Iterable[str], k: int) -> float:
    """Fraction of expected sources present in the top-k retrieved
    filenames: |top-k ∩ expected| / |expected|."""
    if k <= 0:
        raise ValueError("k must be a positive integer")
    expected = set(expected_sources)
    if not expected:
        raise ValueError("expected_sources must not be empty")
    top_k = set(retrieved_filenames[:k])
    return len(top_k & expected) / len(expected)


def reciprocal_rank(retrieved_filenames: Sequence[str], expected_sources: Iterable[str], k: int) -> float:
    """1 / rank of the first retrieved (top-k) filename that is an expected
    source (rank starts at 1); 0.0 if none of the top-k are relevant."""
    if k <= 0:
        raise ValueError("k must be a positive integer")
    expected = set(expected_sources)
    for rank, filename in enumerate(retrieved_filenames[:k], start=1):
        if filename in expected:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExampleResult:
    """Per-query metric values, plus what was actually retrieved (useful
    for debugging a low score)."""

    query: str
    expected_sources: tuple[str, ...]
    retrieved_filenames: tuple[str, ...]
    hit: float
    recall: float
    reciprocal_rank: float


@dataclass(frozen=True)
class EvalReport:
    """Aggregate evaluation results across the whole dataset."""

    k: int
    results: tuple[ExampleResult, ...]

    @property
    def mean_hit_at_k(self) -> float:
        return _mean(r.hit for r in self.results)

    @property
    def mean_recall_at_k(self) -> float:
        return _mean(r.recall for r in self.results)

    @property
    def mrr(self) -> float:
        return _mean(r.reciprocal_rank for r in self.results)

    def format_report(self) -> str:
        lines = [f"RAG retrieval evaluation (k={self.k}, n={len(self.results)} queries)", ""]
        for r in self.results:
            status = "HIT " if r.hit else "MISS"
            lines.append(
                f"[{status}] {r.query!r} -> expected={list(r.expected_sources)} "
                f"retrieved={list(r.retrieved_filenames)}"
            )
        lines += [
            "",
            f"Hit@{self.k}:    {self.mean_hit_at_k:.3f}",
            f"Recall@{self.k}: {self.mean_recall_at_k:.3f}",
            f"MRR:       {self.mrr:.3f}",
        ]
        return "\n".join(lines)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def evaluate_retrieval(
    pipeline: RAGPipeline, examples: Sequence[EvalExample], k: int
) -> EvalReport:
    """Run `pipeline.retrieve()` for every example and score the results."""
    if k <= 0:
        raise ValueError("k must be a positive integer")

    results: list[ExampleResult] = []
    for example in examples:
        retrieved = pipeline.retrieve(example.query, top_k=k)
        retrieved_filenames = tuple(chunk.filename for chunk in retrieved)
        results.append(
            ExampleResult(
                query=example.query,
                expected_sources=example.expected_sources,
                retrieved_filenames=retrieved_filenames,
                hit=hit_at_k(retrieved_filenames, example.expected_sources, k),
                recall=recall_at_k(retrieved_filenames, example.expected_sources, k),
                reciprocal_rank=reciprocal_rank(retrieved_filenames, example.expected_sources, k),
            )
        )
    return EvalReport(k=k, results=tuple(results))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval quality (Hit@K, Recall@K, MRR) against a small deterministic dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Path to the evaluation dataset JSON file (default: {DEFAULT_DATASET_PATH}).",
    )
    parser.add_argument("--k", type=int, default=3, help="Top-K cutoff used for all metrics (default: 3).")
    parser.add_argument(
        "--embedding-provider",
        choices=["hashing", "fastembed"],
        default=None,
        help="Override RAGConfig.embedding_provider. 'hashing' is deterministic and fully offline. "
        "Defaults to whatever the existing RAG config specifies.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    args = _build_arg_parser().parse_args(argv)

    config: RAGConfig = get_default_config()
    if args.embedding_provider is not None:
        from dataclasses import replace

        config = replace(config, embedding_provider=args.embedding_provider)

    pipeline = RAGPipeline.load_or_build(config)
    examples = load_dataset(args.dataset)
    report = evaluate_retrieval(pipeline, examples, k=args.k)

    print(report.format_report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
