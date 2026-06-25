"""
Vector store module.

Manages a ChromaDB persistent collection for storing and querying
document-chunk embeddings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import chromadb

from . import config
from .document_loader import DocumentChunk
from .embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────
@dataclass
class RetrievalResult:
    """A single retrieval hit from the vector store."""

    content: str
    metadata: dict = field(default_factory=dict)
    similarity_score: float = 0.0


# ──────────────────────────────────────────────
# VectorStore
# ──────────────────────────────────────────────
class VectorStore:
    """ChromaDB-backed vector store for document chunks.

    Parameters
    ----------
    persist_dir:
        Filesystem path where ChromaDB stores its data.
    collection_name:
        Name of the Chroma collection.
    embedding_model:
        An :class:`EmbeddingModel` instance used to produce query vectors.
    """

    def __init__(
        self,
        persist_dir: str = config.CHROMA_PERSIST_DIR,
        collection_name: str = "rag_default",
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._embedding_model = embedding_model

        logger.info(
            "Initialising ChromaDB (persist_dir=%s, collection=%s)",
            persist_dir,
            collection_name,
        )
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Collection '%s' ready (%d existing document(s))",
            collection_name,
            self._collection.count(),
        )

    # ── build index ──────────────────────────
    def build_index(self, chunks: list[DocumentChunk]) -> None:
        """Clear the collection and rebuild the index from *chunks*.

        Parameters
        ----------
        chunks:
            Document chunks to index.
        """
        if not chunks:
            logger.warning("build_index called with an empty chunk list")
            return

        if self._embedding_model is None:
            raise RuntimeError(
                "VectorStore requires an EmbeddingModel to build "
                "an index.  Pass one via the constructor."
            )

        # Clear existing data
        self.reset()
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Prepare data for ChromaDB
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        embeddings: list[list[float]] = []

        texts = [chunk.content for chunk in chunks]
        logger.info("Embedding %d chunk(s)…", len(texts))
        all_embeddings = self._embedding_model.embed_texts(texts)

        for idx, chunk in enumerate(chunks):
            doc_id = chunk.metadata.get('doc_id', 'unknown')
            c_idx = chunk.metadata.get('chunk_index', idx)
            chunk_id = f"{doc_id}_{c_idx}"
            ids.append(chunk_id)
            documents.append(chunk.content)
            metadatas.append(chunk.metadata)
            embeddings.append(all_embeddings[idx])

        # ChromaDB has a batch limit; add in slices of 5 000
        batch_size = 5000
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self._collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                embeddings=embeddings[start:end],
            )

        logger.info(
            "Indexed %d chunk(s) into collection '%s'",
            len(ids), self._collection_name,
        )

    # ── query ────────────────────────────────
    def query(
        self, query_text: str, top_k: int = config.TOP_K,
    ) -> list[RetrievalResult]:
        """Query the collection and return the top-k most similar chunks.

        Parameters
        ----------
        query_text:
            Natural-language query string.
        top_k:
            Number of results to return.

        Returns
        -------
        list[RetrievalResult]
            Results sorted by descending similarity score.
        """
        if self._embedding_model is None:
            raise RuntimeError(
                "VectorStore requires an EmbeddingModel to run "
                "queries.  Pass one via the constructor."
            )

        if self._collection.count() == 0:
            logger.warning("Query attempted on an empty collection")
            return []

        query_embedding = self._embedding_model.embed_text(query_text)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        retrieval_results: list[RetrievalResult] = []
        if results and results["documents"]:
            for doc, meta, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
                strict=True,
            ):
                # ChromaDB cosine distance → similarity: sim = 1 - distance
                similarity = 1.0 - distance
                retrieval_results.append(
                    RetrievalResult(
                        content=doc,
                        metadata=meta,
                        similarity_score=round(similarity, 4),
                    )
                )

        logger.debug("Query returned %d result(s)", len(retrieval_results))
        return retrieval_results

    # ── reset ────────────────────────────────
    def reset(self) -> None:
        """Delete the entire collection from ChromaDB."""
        try:
            self._client.delete_collection(name=self._collection_name)
            logger.info("Collection '%s' deleted", self._collection_name)
        except Exception:
            logger.debug(
                "Collection '%s' did not exist; nothing to delete",
                self._collection_name,
            )

    # ── count ────────────────────────────────
    def collection_count(self) -> int:
        """Return the number of documents stored in the collection."""
        return self._collection.count()
