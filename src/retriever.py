"""
Retriever module.

Sits on top of :class:`VectorStore` and applies a minimum-similarity
threshold, returning a :class:`RetrievalOutput` that also carries an
aggregate confidence score (mean of the kept similarity scores).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from . import config
from .vector_store import RetrievalResult, VectorStore

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────
@dataclass
class RetrievalOutput:
    """Bundle returned by :meth:`Retriever.retrieve`.

    Attributes
    ----------
    chunks:
        The retrieval hits that passed the similarity threshold.
    confidence_score:
        Mean similarity score across the returned chunks (0.0 if none).
    """

    chunks: list[RetrievalResult] = field(default_factory=list)
    confidence_score: float = 0.0


# ──────────────────────────────────────────────
# Retriever
# ──────────────────────────────────────────────
class Retriever:
    """High-level retrieval interface with similarity-threshold filtering.

    Parameters
    ----------
    vector_store:
        The underlying :class:`VectorStore` to query.
    top_k:
        Number of chunks to request from the vector store.
    min_threshold:
        Minimum similarity score a result must have to be kept.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = config.TOP_K,
        min_threshold: float = config.MIN_SIMILARITY_THRESHOLD,
    ) -> None:
        self._vector_store = vector_store
        self._top_k = top_k
        self._min_threshold = min_threshold
        logger.info(
            "Retriever initialised (top_k=%d, min_threshold=%.2f)",
            top_k,
            min_threshold,
        )

    def retrieve(self, query: str) -> RetrievalOutput:
        """Retrieve relevant chunks for *query*.

        Parameters
        ----------
        query:
            The user question in natural language.

        Returns
        -------
        RetrievalOutput
            Filtered chunks and an aggregate confidence score.
        """
        raw_results: list[RetrievalResult] = self._vector_store.query(
            query_text=query,
            top_k=self._top_k,
        )

        # Filter by minimum similarity threshold
        filtered: list[RetrievalResult] = [
            r for r in raw_results if r.similarity_score >= self._min_threshold
        ]

        if not filtered:
            logger.warning(
                "No results above threshold %.2f for query: %s",
                self._min_threshold,
                query[:80],
            )
            return RetrievalOutput(chunks=[], confidence_score=0.0)

        confidence = sum(r.similarity_score for r in filtered) / len(filtered)

        logger.info(
            "Retrieved %d chunk(s) (confidence=%.4f) for query: %s",
            len(filtered),
            confidence,
            query[:80],
        )

        return RetrievalOutput(
            chunks=filtered,
            confidence_score=round(confidence, 4),
        )
