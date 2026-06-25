"""OpenAI LLM provider."""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

from .. import config
from .base import LLMProvider

load_dotenv()
logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI via the ``openai`` Python SDK."""

    def __init__(
        self,
        model: str | None = None,
    ) -> None:
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key or api_key == "your_key_here":
            raise OSError(
                "OPENAI_API_KEY not set. "
                "Get one at: https://platform.openai.com/api-keys"
            )

        self._model = (
            model or config.PROVIDER_MODELS.get(
                "openai", "gpt-4o-mini",
            )
        )
        self._client = OpenAI(api_key=api_key)
        logger.info(
            "OpenAIProvider ready (model=%s)", self._model,
        )

    @property
    def provider_name(self) -> str:
        return "OpenAI"

    @property
    def model_name(self) -> str:
        return self._model

    def _call_api(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config.MAX_OUTPUT_TOKENS,
            temperature=config.TEMPERATURE,
        )
        choice = response.choices[0]
        return choice.message.content or ""

    def health_check(self) -> bool:
        try:
            self._call_api("Say 'ok'.")
            return True
        except Exception:
            return False
