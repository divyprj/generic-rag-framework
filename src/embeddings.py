"""
Embedding module.

Wraps the ``sentence-transformers`` library to produce normalised
dense vectors using the BGE-small-en-v1.5 model (384-dim).
"""

from __future__ import annotations

import logging
import warnings

# Suppress Hugging Face hub warning about unauthenticated requests
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer  # noqa: E402

from . import config  # noqa: E402

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Thin wrapper around a SentenceTransformer model.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.  Defaults to
        :pydata:`config.EMBEDDING_MODEL` (``BAAI/bge-small-en-v1.5``).
    """

    def __init__(self, model_name: str = config.EMBEDDING_MODEL) -> None:
        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimension = config.EMBEDDING_DIM
        logger.info(
            "Embedding model loaded (dim=%d)",
            self.dimension,
        )

    # ── single text ──────────────────────────
    def embed_text(self, text: str) -> list[float]:
        """Embed a single piece of text and return a normalised vector.

        Parameters
        ----------
        text:
            The input string to embed.

        Returns
        -------
        list[float]
            A normalised embedding vector of length ``self.dimension``.
        """
        embedding = self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()

    # ── batch ────────────────────────────────
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts and return normalised vectors.

        Parameters
        ----------
        texts:
            List of input strings.

        Returns
        -------
        list[list[float]]
            One normalised embedding vector per input string.
        """
        if not texts:
            return []

        logger.debug("Embedding %d text(s)", len(texts))
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )
        return embeddings.tolist()
