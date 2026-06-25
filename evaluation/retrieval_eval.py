"""Retrieval performance evaluation for the RAG pipeline.

Measures how well the retriever surfaces relevant documents:
- **Precision@k** — fraction of retrieved docs that are relevant
- **Recall@k** — fraction of relevant docs that were retrieved
- **MRR** — mean reciprocal rank of the first relevant result
"""

from __future__ import annotations

from typing import Any


def precision_at_k(
    retrieved_doc_ids: list[str],
    relevant_doc_ids: list[str],
) -> float:
    """Fraction of retrieved documents that are relevant.

    Args:
        retrieved_doc_ids: Ordered list of document IDs returned by retriever.
        relevant_doc_ids:  Ground-truth set of relevant document IDs.

    Returns:
        Precision in [0.0, 1.0].
    """
    if not retrieved_doc_ids:
        return 0.0

    relevant_set = set(relevant_doc_ids)
    hits = sum(1 for doc_id in retrieved_doc_ids if doc_id in relevant_set)
    return hits / len(retrieved_doc_ids)


def recall_at_k(
    retrieved_doc_ids: list[str],
    relevant_doc_ids: list[str],
) -> float:
    """Fraction of relevant documents that appear in the retrieved set.

    Args:
        retrieved_doc_ids: Ordered list of document IDs returned by retriever.
        relevant_doc_ids:  Ground-truth set of relevant document IDs.

    Returns:
        Recall in [0.0, 1.0].
    """
    if not relevant_doc_ids:
        return 1.0  # vacuously true — nothing to recall

    relevant_set = set(relevant_doc_ids)
    retrieved_set = set(retrieved_doc_ids)
    hits = len(relevant_set & retrieved_set)
    return hits / len(relevant_set)


def mean_reciprocal_rank(
    retrieved_doc_ids: list[str],
    relevant_doc_ids: list[str],
) -> float:
    """Reciprocal of the rank of the first relevant document.

    Args:
        retrieved_doc_ids: Ordered list of document IDs returned by retriever.
        relevant_doc_ids:  Ground-truth set of relevant document IDs.

    Returns:
        MRR value in [0.0, 1.0].  0.0 if no relevant document is found.
    """
    relevant_set = set(relevant_doc_ids)
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(
    retrieved_chunks: list[dict[str, Any]],
    relevant_doc_ids: list[str],
) -> dict[str, float]:
    """Compute all retrieval metrics for a single query.

    Args:
        retrieved_chunks: List of chunk dicts, each with at least a
                          ``"doc_id"`` key (and optionally ``"source"``,
                          ``"score"``, ``"content"``, ``"chunk_index"``).
        relevant_doc_ids: Ground-truth relevant document IDs from the
                          QA dataset.

    Returns:
        Dictionary with keys ``precision_at_k``, ``recall_at_k``, ``mrr``.
    """
    # Extract ordered doc_ids from chunks (preserve retrieval order)
    retrieved_ids: list[str] = [
        chunk["doc_id"]
        for chunk in retrieved_chunks
        if "doc_id" in chunk
    ]

    return {
        "precision_at_k": round(
            precision_at_k(retrieved_ids, relevant_doc_ids), 4
        ),
        "recall_at_k": round(
            recall_at_k(retrieved_ids, relevant_doc_ids), 4
        ),
        "mrr": round(
            mean_reciprocal_rank(retrieved_ids, relevant_doc_ids), 4
        ),
    }
