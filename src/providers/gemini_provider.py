"""Google Gemini LLM provider."""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

from .. import config
from .base import LLMProvider

load_dotenv()
logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """Google Gemini via the ``google-genai`` SDK."""

    def __init__(
        self,
        model: str | None = None,
    ) -> None:
        from google import genai

        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key or api_key == "your_key_here":
            raise OSError(
                "GOOGLE_API_KEY not set. "
                "Get one at: https://aistudio.google.com/apikey"
            )

        self._model = (
            model or config.PROVIDER_MODELS.get(
                "gemini", "gemini-2.0-flash",
            )
        )
        self._client = genai.Client(api_key=api_key)
        logger.info("GeminiProvider ready (model=%s)", self._model)

    @property
    def provider_name(self) -> str:
        return "Gemini"

    @property
    def model_name(self) -> str:
        return self._model

    def _call_api(self, prompt: str) -> str:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=config.MAX_OUTPUT_TOKENS,
                temperature=config.TEMPERATURE,
            ),
        )
        return response.text or ""

    def health_check(self) -> bool:
        try:
            self._call_api("Say 'ok'.")
            return True
        except Exception:
            return False
