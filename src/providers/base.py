"""
Abstract base class for LLM providers.

Every provider implements ``generate()`` and ``health_check()``.
Shared retry logic with exponential backoff lives here.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Unified interface for LLM generation backends.

    Subclasses must implement :meth:`_call_api` and
    :meth:`health_check`.
    """

    MAX_RETRIES: int = 3
    BACKOFF_BASE: float = 1.0

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name (e.g. 'Groq')."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Active model identifier."""

    @abstractmethod
    def _call_api(self, prompt: str) -> str:
        """Send a prompt to the provider and return the text.

        This method should NOT implement retries — the base class
        handles retries via :meth:`generate`.
        """

    @abstractmethod
    def health_check(self) -> bool:
        """Return True if the provider is reachable and keys valid."""

    def generate(self, prompt: str) -> str:
        """Generate text with automatic retry and backoff.

        Parameters
        ----------
        prompt:
            The full prompt string.

        Returns
        -------
        str
            The generated text.

        Raises
        ------
        RuntimeError
            If all retry attempts are exhausted.
        """
        last_error: Exception | None = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                logger.debug(
                    "%s call attempt %d/%d",
                    self.provider_name,
                    attempt,
                    self.MAX_RETRIES,
                )
                return self._call_api(prompt)
            except Exception as exc:
                last_error = exc
                wait = self.BACKOFF_BASE * (2 ** (attempt - 1))
                logger.warning(
                    "%s API error (attempt %d/%d): %s "
                    "- retrying in %.1fs",
                    self.provider_name,
                    attempt,
                    self.MAX_RETRIES,
                    exc,
                    wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"{self.provider_name} API call failed after "
            f"{self.MAX_RETRIES} attempts: {last_error}"
        )
