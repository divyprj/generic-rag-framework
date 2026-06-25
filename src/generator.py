"""
Generator module.

Builds prompts from retrieved context chunks and delegates text
generation to the configured :class:`LLMProvider`.  The generator
is provider-agnostic — it never imports a provider SDK directly.

Optional fallback: when enabled, if the primary provider fails
with a retriable error, the next provider in the fallback chain
is tried.  The retrieved context is preserved across retries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from . import config

if TYPE_CHECKING:
    from .vector_store import RetrievalResult

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Prompt template
# ──────────────────────────────────────────────
_PROMPT_TEMPLATE = """\
{persona}

Instructions:
- Answer only using the supplied context below.
- If the context does not contain enough information to answer, \
say "I cannot fully answer this from the provided documents."
- Never fabricate or assume facts not present in the context.
- Cite the source document names used in your answer.
- Keep answers concise but complete.
- Use bullet points when listing multiple items.

Context:
{context_block}

Question: {question}

Answer:
"""


# ──────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────
@dataclass
class GenerationResult:
    """Output from :meth:`Generator.generate`.

    Attributes
    ----------
    answer:
        The generated answer text.
    sources:
        List of source document names referenced.
    latency_ms:
        Wall-clock time for the LLM call in milliseconds.
    provider:
        Name of the provider that generated the answer.
    model:
        Name of the model used.
    """

    answer: str = ""
    sources: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    provider: str = ""
    model: str = ""


# ──────────────────────────────────────────────
# Generator
# ──────────────────────────────────────────────
class Generator:
    """Provider-agnostic answer generator with citation extraction.

    Parameters
    ----------
    provider:
        An :class:`LLMProvider` instance for text generation.
    persona:
        System persona injected into the prompt template.
    fallback:
        If ``True``, try the next provider in the fallback chain
        when the primary fails.
    """

    def __init__(
        self,
        provider,
        persona: str = "",
        fallback: bool = False,
    ) -> None:
        from .providers.base import LLMProvider  # type hint

        self._provider: LLMProvider = provider
        self._persona = (
            persona
            or "You are a knowledgeable assistant."
        )
        self._fallback = fallback
        logger.info(
            "Generator ready (provider=%s, model=%s)",
            provider.provider_name,
            provider.model_name,
        )

    @property
    def provider(self):
        """The active LLM provider."""
        return self._provider

    # ── public API ───────────────────────────
    def generate(
        self,
        query: str,
        context_chunks: list[RetrievalResult],
    ) -> GenerationResult:
        """Generate an answer for *query* grounded in *context_chunks*.

        Parameters
        ----------
        query:
            The user's question.
        context_chunks:
            Retrieved chunks that form the answer context.

        Returns
        -------
        GenerationResult
            The answer, cited sources, and latency.
        """
        if not context_chunks:
            return GenerationResult(
                answer=(
                    "The provided documents do not contain"
                    " enough information to fully answer"
                    " this question."
                ),
                sources=[],
                latency_ms=0.0,
                provider=self._provider.provider_name,
                model=self._provider.model_name,
            )

        # Build the prompt
        context_block = self._build_context_block(
            context_chunks,
        )
        prompt = _PROMPT_TEMPLATE.format(
            persona=self._persona,
            context_block=context_block,
            question=query,
        )

        # Generate with optional fallback
        answer_text, used_provider = self._generate_with_fallback(
            prompt,
        )

        # Extract sources
        sources = self._extract_sources(context_chunks)

        return GenerationResult(
            answer=answer_text,
            sources=sources,
            latency_ms=0.0,
            provider=used_provider.provider_name,
            model=used_provider.model_name,
        )

    # ── internals ────────────────────────────
    def _generate_with_fallback(self, prompt: str):
        """Try the primary provider, fall back if enabled."""
        try:
            text = self._provider.generate(prompt)
            return text, self._provider
        except RuntimeError as exc:
            if not self._fallback:
                raise

            logger.warning(
                "Primary provider %s failed: %s — "
                "trying fallback chain",
                self._provider.provider_name,
                exc,
            )
            return self._try_fallback_chain(prompt, exc)

    def _try_fallback_chain(
        self,
        prompt: str,
        original_error: Exception,
    ):
        """Walk the fallback chain and try each provider."""
        from .providers import get_provider, is_provider_configured

        primary = self._provider.provider_name.lower()
        skipped: list[str] = []
        for name in config.FALLBACK_PROVIDERS:
            if name == primary:
                continue
            configured, detail = is_provider_configured(name)
            if not configured:
                skipped.append(f"{name} ({detail})")
                logger.info(
                    "Skipping fallback provider %s: %s",
                    name, detail,
                )
                continue
            try:
                logger.info(
                    "Attempting fallback provider: %s", name,
                )
                fallback = get_provider(name)
                text = fallback.generate(prompt)
                logger.info(
                    "Fallback succeeded with %s", name,
                )
                return text, fallback
            except Exception as exc:
                logger.warning(
                    "Fallback %s also failed: %s",
                    name, exc,
                )
                continue

        raise RuntimeError(
            f"All providers failed. "
            f"Original error: {original_error}"
            + (
                f". Skipped unconfigured providers: "
                f"{', '.join(skipped)}"
                if skipped else ""
            )
        )

    @staticmethod
    def _build_context_block(
        chunks: list[RetrievalResult],
    ) -> str:
        """Format retrieved chunks into the prompt context."""
        sections: list[str] = []
        for chunk in chunks:
            source_name = chunk.metadata.get(
                "source", "unknown",
            )
            sections.append(
                f"---\n[Source: {source_name}]\n"
                f"{chunk.content}\n---"
            )
        return "\n".join(sections)

    @staticmethod
    def _extract_sources(
        chunks: list[RetrievalResult],
    ) -> list[str]:
        """Return deduplicated source names (first-seen order)."""
        seen: set[str] = set()
        sources: list[str] = []
        for chunk in chunks:
            name = chunk.metadata.get("source", "unknown")
            if name not in seen:
                seen.add(name)
                sources.append(name)
        return sources
