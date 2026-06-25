"""
LLM Provider abstraction layer.

Provides a unified interface for multiple LLM backends.
Only the selected provider's SDK is imported (lazy loading).

Usage::

    from src.providers import get_provider, list_providers

    provider = get_provider("groq")
    answer = provider.generate("What is quantum computing?")
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from .. import config

if TYPE_CHECKING:
    from .base import LLMProvider

load_dotenv()
logger = logging.getLogger(__name__)

# Registry of provider names → module + class
_PROVIDERS: dict[str, dict[str, str]] = {
    "groq": {
        "module": "src.providers.groq_provider",
        "class": "GroqProvider",
        "key_env": "GROQ_API_KEY",
        "key_url": "https://console.groq.com/keys",
    },
    "gemini": {
        "module": "src.providers.gemini_provider",
        "class": "GeminiProvider",
        "key_env": "GOOGLE_API_KEY",
        "key_url": "https://aistudio.google.com/apikey",
    },
    "openai": {
        "module": "src.providers.openai_provider",
        "class": "OpenAIProvider",
        "key_env": "OPENAI_API_KEY",
        "key_url": "https://platform.openai.com/api-keys",
    },
    "ollama": {
        "module": "src.providers.ollama_provider",
        "class": "OllamaProvider",
        "key_env": "",
        "key_url": "",
    },
}


def _normalise_provider_name(name: str | None) -> str:
    """Return a canonical provider name."""
    return (name or config.DEFAULT_PROVIDER).lower().strip()


def get_provider(
    name: str | None = None,
    model: str | None = None,
) -> LLMProvider:
    """Create and return an LLM provider instance.

    Parameters
    ----------
    name:
        Provider identifier (gemini, groq, openai, ollama).
        Defaults to ``config.DEFAULT_PROVIDER``.
    model:
        Override the default model for this provider.

    Returns
    -------
    LLMProvider
        An initialised provider ready to generate.

    Raises
    ------
    ValueError
        If the provider name is unknown.
    """
    import importlib

    name = _normalise_provider_name(name)

    if name not in _PROVIDERS:
        available = ", ".join(_PROVIDERS)
        raise ValueError(
            f"Unknown provider '{name}'. "
            f"Available: {available}"
        )

    info = _PROVIDERS[name]
    module = importlib.import_module(info["module"])
    cls = getattr(module, info["class"])

    kwargs: dict[str, str] = {}
    if model:
        kwargs["model"] = model

    return cls(**kwargs)


def is_provider_configured(name: str) -> tuple[bool, str]:
    """Return whether a provider has the minimum setup needed to run."""
    name = _normalise_provider_name(name)
    info = _PROVIDERS.get(name)
    if not info:
        return False, f"Unknown provider: {name}"

    if name == "ollama":
        try:
            import httpx

            resp = httpx.get(
                f"{config.OLLAMA_BASE_URL}/api/tags",
                timeout=2.0,
            )
            if resp.status_code == 200:
                return True, config.OLLAMA_BASE_URL
            return False, "Server not responding"
        except Exception:
            return False, "Server not running"

    key_env = info["key_env"]
    key = os.environ.get(key_env, "")
    if key and key != "your_key_here":
        return True, f"{key_env} set"
    return False, f"Missing: {key_env}"


def list_providers() -> list[dict[str, str]]:
    """Return status information for all known providers.

    Returns
    -------
    list[dict]
        Each dict has keys: name, model, status, detail.
    """
    results: list[dict[str, str]] = []

    for name in _PROVIDERS:
        model = config.PROVIDER_MODELS.get(name, "unknown")
        configured, detail = is_provider_configured(name)

        if name == "ollama":
            status = "Configured" if configured else "Offline"
        elif configured:
            status = "Configured"
        else:
            status = "Not Configured"

        results.append({
            "name": name,
            "model": model,
            "status": status,
            "detail": detail,
        })

    return results


def get_provider_info(name: str) -> dict[str, str]:
    """Get registry info for a specific provider."""
    name = _normalise_provider_name(name)
    if name not in _PROVIDERS:
        return {}
    return _PROVIDERS[name]
