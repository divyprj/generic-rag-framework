"""Ollama local LLM provider.

Uses ``httpx`` to call the Ollama REST API directly — no SDK
dependency required. Ollama must be running locally.
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv

from .. import config
from .base import LLMProvider

load_dotenv()
logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Ollama local server via REST API."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        import httpx

        self._model = (
            model or config.PROVIDER_MODELS.get(
                "ollama", "llama3",
            )
        )
        self._base_url = (
            base_url or config.OLLAMA_BASE_URL
        )
        self._http = httpx.Client(timeout=120.0)

        # Verify server is reachable
        try:
            resp = self._http.get(
                f"{self._base_url}/api/tags",
            )
            if resp.status_code != 200:
                raise ConnectionError(
                    f"Ollama returned status {resp.status_code}"
                )
        except Exception as exc:
            raise ConnectionError(
                f"Cannot reach Ollama at {self._base_url}. "
                f"Make sure Ollama is running: "
                f"https://ollama.com/download\n"
                f"Error: {exc}"
            ) from exc

        logger.info(
            "OllamaProvider ready (model=%s, url=%s)",
            self._model,
            self._base_url,
        )

    @property
    def provider_name(self) -> str:
        return "Ollama"

    @property
    def model_name(self) -> str:
        return self._model

    def _call_api(self, prompt: str) -> str:
        import httpx

        response = self._http.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": config.TEMPERATURE,
                    "num_predict": config.MAX_OUTPUT_TOKENS,
                },
            },
            timeout=httpx.Timeout(120.0),
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")

    def health_check(self) -> bool:
        try:
            resp = self._http.get(
                f"{self._base_url}/api/tags",
            )
            return resp.status_code == 200
        except Exception:
            return False
