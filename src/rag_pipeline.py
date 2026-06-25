"""
End-to-end RAG pipeline.

Orchestrates document ingestion (load -> chunk -> embed -> index) and
query execution (retrieve -> generate), returning a structured
:class:`RAGResult`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

from . import config
from .document_loader import chunk_documents, load_documents
from .embeddings import EmbeddingModel
from .generator import Generator
from .providers import get_provider
from .retriever import Retriever
from .vector_store import VectorStore

# Ensure .env is loaded early
load_dotenv()

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────
@dataclass
class RAGResult:
    """Structured result returned by :meth:`RAGPipeline.query`.

    Attributes
    ----------
    answer:
        The generated answer text.
    sources:
        List of dicts with ``doc_name`` and ``chunk_index`` keys.
    confidence_scores:
        Per-chunk similarity scores.
    retrieved_chunks:
        List of dicts with ``content``, ``source``, and ``score`` keys.
    latency_ms:
        Total wall-clock time for the query (retrieval + generation).
    provider:
        Name of the LLM provider that generated the answer.
    model:
        Name of the model used.
    """

    answer: str = ""
    sources: list[dict[str, object]] = field(default_factory=list)
    confidence_scores: list[float] = field(default_factory=list)
    retrieved_chunks: list[dict[str, object]] = field(
        default_factory=list,
    )
    latency_ms: float = 0.0
    provider: str = ""
    model: str = ""


# ──────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────
class RAGPipeline:
    """Full Retrieval-Augmented Generation pipeline.

    Initialises all sub-components (embedding model, vector store,
    retriever, generator) on construction and exposes ``ingest`` /
    ``query`` as the primary interface.

    Parameters
    ----------
    dataset_id:
        Identifier of the dataset to use (folder name under ``data/``).
        Defaults to :pydata:`config.DEFAULT_DATASET`.
    provider_name:
        LLM provider to use for generation.
        Defaults to :pydata:`config.DEFAULT_PROVIDER`.
    fallback:
        Enable automatic provider fallback on failure.
    """

    def __init__(
        self,
        dataset_id: str = config.DEFAULT_DATASET,
        provider_name: str = config.DEFAULT_PROVIDER,
        fallback: bool = False,
    ) -> None:
        logger.info("Initialising RAG pipeline...")

        self._dataset_config = config.load_dataset_config(
            dataset_id,
        )

        # LLM provider
        self._llm_provider = get_provider(provider_name)

        self._embedding_model = EmbeddingModel()
        self._vector_store = VectorStore(
            persist_dir=config.CHROMA_PERSIST_DIR,
            collection_name=self._dataset_config.collection_name,
            embedding_model=self._embedding_model,
        )
        self._retriever = Retriever(
            vector_store=self._vector_store,
            top_k=config.TOP_K,
            min_threshold=config.MIN_SIMILARITY_THRESHOLD,
        )
        self._generator = Generator(
            provider=self._llm_provider,
            persona=self._dataset_config.persona,
            fallback=fallback,
        )

        logger.info(
            "RAG pipeline ready (dataset=%s, provider=%s, "
            "model=%s)",
            self._dataset_config.dataset_id,
            self._llm_provider.provider_name,
            self._llm_provider.model_name,
        )

    @property
    def vector_store(self) -> VectorStore:
        """Public access to the vector store for stats/management."""
        return self._vector_store

    @property
    def embedding_model(self) -> EmbeddingModel:
        """Public access to the embedding model for reuse."""
        return self._embedding_model

    @property
    def dataset_config(self) -> config.DatasetConfig:
        """Public access to the active dataset configuration."""
        return self._dataset_config

    @property
    def llm_provider(self):
        """The active LLM provider."""
        return self._llm_provider

    # ── ingest ───────────────────────────────
    def ingest(self, data_dir: str | None = None) -> int:
        """Load documents, chunk them, and build the vector index.

        Parameters
        ----------
        data_dir:
            Path to the document directory.  Falls back to the
            dataset's configured ``data_dir``.

        Returns
        -------
        int
            Total number of chunks indexed.
        """
        data_dir = data_dir or self._dataset_config.data_dir
        logger.info("Ingesting documents from %s", data_dir)

        documents = load_documents(data_dir)
        if not documents:
            logger.warning(
                "No documents found - index will be empty",
            )
            return 0

        chunks = chunk_documents(
            documents,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
        self._vector_store.build_index(chunks)

        count = self._vector_store.collection_count()
        logger.info(
            "Ingestion complete: %d chunk(s) indexed", count,
        )
        return count

    # ── query ────────────────────────────────
    def query(self, question: str) -> RAGResult:
        """Run the full RAG pipeline for a single question.

        Parameters
        ----------
        question:
            The user's natural-language question.

        Returns
        -------
        RAGResult
            Structured result with answer, sources, scores,
            latency, and provider info.
        """
        t_start = time.perf_counter()

        # Retrieve
        retrieval_output = self._retriever.retrieve(question)

        # Generate
        gen_result = self._generator.generate(
            query=question,
            context_chunks=retrieval_output.chunks,
        )

        t_end = time.perf_counter()
        total_latency_ms = round((t_end - t_start) * 1000, 2)

        # Assemble structured output
        sources: list[dict[str, object]] = []
        confidence_scores: list[float] = []
        retrieved_chunks: list[dict[str, object]] = []

        for chunk in retrieval_output.chunks:
            sources.append({
                "doc_name": chunk.metadata.get(
                    "source", "unknown",
                ),
                "chunk_index": chunk.metadata.get(
                    "chunk_index", -1,
                ),
            })
            confidence_scores.append(chunk.similarity_score)
            retrieved_chunks.append({
                "content": chunk.content,
                "source": chunk.metadata.get(
                    "source", "unknown",
                ),
                "doc_id": chunk.metadata.get(
                    "doc_id", "unknown",
                ),
                "chunk_index": chunk.metadata.get(
                    "chunk_index", -1,
                ),
                "score": chunk.similarity_score,
            })

        return RAGResult(
            answer=gen_result.answer,
            sources=sources,
            confidence_scores=confidence_scores,
            retrieved_chunks=retrieved_chunks,
            latency_ms=total_latency_ms,
            provider=gen_result.provider,
            model=gen_result.model,
        )

    # ── helpers ──────────────────────────────
    def is_indexed(self) -> bool:
        """Return ``True`` if the vector store has data."""
        return self._vector_store.collection_count() > 0
