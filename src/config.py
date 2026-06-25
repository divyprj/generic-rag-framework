"""
Central configuration module for the EDXSO RAG pipeline.

All configurable parameters — paths, chunking strategy, model names,
retrieval thresholds, and generation settings — are defined here so that
every other module can import them from a single source of truth.

Dataset-specific settings are loaded dynamically via
:func:`load_dataset_config`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Project root (resolved once)
# ──────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_ROOT: str = str(PROJECT_ROOT / "data")
CHROMA_PERSIST_DIR: str = str(PROJECT_ROOT / "chroma_db")
RESULTS_DIR: str = str(PROJECT_ROOT / "results")

# ──────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 100

# ──────────────────────────────────────────────
# Embedding
# ──────────────────────────────────────────────
EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM: int = 384

# ──────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────
TOP_K: int = 3
MIN_SIMILARITY_THRESHOLD: float = 0.3

# ──────────────────────────────────────────────
# Generation (shared across providers)
# ──────────────────────────────────────────────
GEMINI_MODEL: str = "gemini-2.0-flash"
MAX_OUTPUT_TOKENS: int = 1024
TEMPERATURE: float = 0.1

# ──────────────────────────────────────────────
# LLM Provider
# ──────────────────────────────────────────────
DEFAULT_PROVIDER: str = "groq"

PROVIDER_MODELS: dict[str, str] = {
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-4o-mini",
    "ollama": "llama3",
}

OLLAMA_BASE_URL: str = "http://localhost:11434"

# Fallback chain (tried in order when --fallback is set)
FALLBACK_PROVIDERS: list[str] = [
    "groq", "gemini", "openai", "ollama",
]

# ──────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────
DEFAULT_DATASET: str = "mars"


# ──────────────────────────────────────────────
# Dataset configuration
# ──────────────────────────────────────────────
@dataclass
class DatasetConfig:
    """Runtime configuration for a specific dataset.

    Attributes
    ----------
    dataset_id:
        Short identifier matching the folder name under ``data/``.
    name:
        Human-readable display name.
    persona:
        System prompt persona for the LLM generator.
    data_dir:
        Absolute path to the dataset's ``documents/`` folder.
    qa_dataset_path:
        Absolute path to the dataset's ``qa_dataset.json``.
    collection_name:
        ChromaDB collection name (unique per dataset).
    """

    dataset_id: str
    name: str
    persona: str
    data_dir: str
    qa_dataset_path: str
    collection_name: str


def load_dataset_config(dataset_id: str) -> DatasetConfig:
    """Load configuration for the given dataset.

    Reads ``data/<dataset_id>/dataset.json`` for the display name and
    persona.  If the file is missing, sensible defaults are derived
    from the folder name.

    Parameters
    ----------
    dataset_id:
        Folder name under ``data/``.

    Returns
    -------
    DatasetConfig
        Fully resolved dataset configuration.

    Raises
    ------
    FileNotFoundError
        If the dataset folder does not exist.
    """
    dataset_dir = Path(DATA_ROOT) / dataset_id
    if not dataset_dir.is_dir():
        available = ", ".join(d.name for d in list_datasets())
        raise FileNotFoundError(
            f"Dataset '{dataset_id}' not found at {dataset_dir}.\n"
            f"Available datasets: {available or 'none'}"
        )

    # Load dataset.json (optional)
    config_path = dataset_dir / "dataset.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            meta = json.load(f)
        name = meta.get("name", dataset_id.replace("_", " ").title())
        persona = meta.get(
            "persona",
            f"You are a knowledgeable assistant specializing"
            f" in {name}.",
        )
    else:
        name = dataset_id.replace("_", " ").title()
        persona = (
            f"You are a knowledgeable assistant specializing"
            f" in {name}."
        )

    return DatasetConfig(
        dataset_id=dataset_id,
        name=name,
        persona=persona,
        data_dir=str(dataset_dir / "documents"),
        qa_dataset_path=str(dataset_dir / "qa_dataset.json"),
        collection_name=f"rag_{dataset_id}",
    )


def list_datasets() -> list[Path]:
    """Discover all available datasets under ``data/``.

    A valid dataset is any subdirectory of ``data/`` that contains
    a ``documents/`` folder.

    Returns
    -------
    list[Path]
        Sorted list of dataset directory paths.
    """
    data_root = Path(DATA_ROOT)
    if not data_root.is_dir():
        return []
    return sorted(
        d for d in data_root.iterdir()
        if d.is_dir() and (d / "documents").is_dir()
    )


def validate_dataset(dataset_id: str) -> list[str]:
    """Run pre-flight checks on a dataset.

    Parameters
    ----------
    dataset_id:
        Folder name under ``data/``.

    Returns
    -------
    list[str]
        List of error messages.  Empty means the dataset is valid.
    """
    errors: list[str] = []
    dataset_dir = Path(DATA_ROOT) / dataset_id

    if not dataset_dir.is_dir():
        errors.append(f"Dataset folder not found: {dataset_dir}")
        return errors

    docs_dir = dataset_dir / "documents"
    if not docs_dir.is_dir():
        errors.append(
            f"Documents folder missing: {docs_dir}"
        )
    else:
        md_files = list(docs_dir.glob("*.md"))
        if not md_files:
            errors.append(
                f"No .md files found in {docs_dir}"
            )
        # Check for empty files
        for f in md_files:
            if f.stat().st_size == 0:
                errors.append(f"Empty document: {f.name}")

    qa_path = dataset_dir / "qa_dataset.json"
    if not qa_path.exists():
        errors.append(
            f"QA dataset missing: {qa_path}"
        )
    else:
        try:
            with open(qa_path, encoding="utf-8") as f:
                data = json.load(f)
            questions = (
                data.get("questions", data)
                if isinstance(data, dict) else data
            )
            if not isinstance(questions, list) or not questions:
                errors.append("QA dataset contains no questions")
            # Check for duplicate IDs
            ids = [q.get("id") for q in questions if "id" in q]
            dupes = {x for x in ids if ids.count(x) > 1}
            if dupes:
                errors.append(
                    f"Duplicate question IDs: {', '.join(dupes)}"
                )
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in qa_dataset.json: {e}")

    return errors
