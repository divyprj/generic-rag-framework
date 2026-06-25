#!/usr/bin/env python3
"""
Generic RAG Framework — Interactive CLI Application

A domain-agnostic, provider-agnostic Retrieval-Augmented Generation
system.  Datasets and LLM providers are selected at runtime.

Usage:
    python app.py                            Interactive (defaults)
    python app.py --dataset quantum          Use a specific dataset
    python app.py --provider groq            Use a specific provider
    python app.py --provider gemini --fallback  Enable fallback
    python app.py --query "..."              Single query mode
    python app.py --list-datasets            Show available datasets
    python app.py --list-providers           Show provider status
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

# ── Suppress noisy third-party warnings ─────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

from dotenv import load_dotenv  # noqa: E402
from rich import box  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.markdown import Markdown  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402

from src import config  # noqa: E402

load_dotenv()

console = Console()


def _provider_display_name(provider_name: str) -> str:
    """Return provider name as users expect to see it."""
    return {
        "openai": "OpenAI",
    }.get(provider_name.lower(), provider_name.title())


# ── Banner ──────────────────────────────────────────────────────────────────


def print_banner(
    ds_config: config.DatasetConfig,
    provider_name: str,
    model_name: str,
) -> None:
    """Display the generic framework banner."""
    title = Text()
    title.append("GENERIC RAG FRAMEWORK\n", style="bold white")
    title.append(
        f"Dataset : {ds_config.name}  |  "
        f"Provider : {provider_name} ({model_name})",
        style="dim white",
    )

    console.print(Panel(
        title,
        border_style="bright_cyan",
        padding=(1, 4),
        title="[bold bright_cyan]RAG[/bold bright_cyan]",
        subtitle=(
            f"[dim]{config.EMBEDDING_MODEL.split('/')[-1]}"
            f"  |  ChromaDB[/dim]"
        ),
    ))
    console.print()


# ── Startup Summary ────────────────────────────────────────────────────────


def print_startup_summary(pipeline, ds_config) -> None:
    """Show a configuration summary after initialisation."""
    doc_dir = Path(ds_config.data_dir)
    doc_count = (
        len(list(doc_dir.glob("*.md")))
        if doc_dir.is_dir() else 0
    )
    chunk_count = pipeline.vector_store.collection_count()
    prov = pipeline.llm_provider

    table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=False,
        padding=(0, 2),
        title="[bold]System Status[/bold]",
        title_style="bright_cyan",
    )
    table.add_column("Key", style="dim", width=22)
    table.add_column("Value", style="bold white")

    table.add_row("Dataset", ds_config.name)
    table.add_row("Dataset ID", ds_config.dataset_id)
    table.add_row("Documents", str(doc_count))
    table.add_row("Chunks Indexed", str(chunk_count))
    table.add_row("", "")
    table.add_row("LLM Provider", prov.provider_name)
    table.add_row("LLM Model", prov.model_name)
    table.add_row(
        "Embedding Model",
        config.EMBEDDING_MODEL.split("/")[-1],
    )
    table.add_row("Vector Store", "ChromaDB (cosine)")
    table.add_row("Chunk Size / Overlap", (
        f"{config.CHUNK_SIZE} / {config.CHUNK_OVERLAP}"
    ))
    table.add_row("Top-K / Threshold", (
        f"{config.TOP_K} / "
        f"{config.MIN_SIMILARITY_THRESHOLD:.2f}"
    ))

    console.print(table)
    console.print(
        "[bold green]  Ready![/bold green]  "
        "Type a question or [dim]help[/dim] for commands.\n"
    )


# ── List Datasets ──────────────────────────────────────────────────────────


def print_dataset_list() -> None:
    """Display all available datasets."""
    datasets = config.list_datasets()

    if not datasets:
        console.print(
            "[yellow]No datasets found.[/yellow]  "
            "Create a folder under data/ with a documents/ "
            "subfolder."
        )
        return

    table = Table(
        box=box.ROUNDED,
        title="[bold]Available Datasets[/bold]",
        title_style="bright_cyan",
        show_header=True,
        header_style="bold",
        padding=(0, 2),
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("ID", style="cyan", width=16)
    table.add_column("Name", style="white")
    table.add_column("Docs", style="green", width=6)

    for i, dataset_dir in enumerate(datasets, 1):
        ds_id = dataset_dir.name
        try:
            ds_cfg = config.load_dataset_config(ds_id)
            name = ds_cfg.name
        except FileNotFoundError:
            name = ds_id.title()

        doc_count = len(
            list((dataset_dir / "documents").glob("*.md"))
        )
        table.add_row(str(i), ds_id, name, str(doc_count))

    console.print(table)
    console.print(
        "\n[dim]Usage:[/dim]  "
        "python app.py --dataset <ID>\n"
    )


# ── List Providers ─────────────────────────────────────────────────────────


def print_provider_list() -> None:
    """Display all available LLM providers and their status."""
    from src.providers import list_providers

    providers = list_providers()

    console.print("[bold bright_cyan]Available Providers[/bold bright_cyan]\n")
    for i, p in enumerate(providers, 1):
        status = p["status"]
        provider_label = _provider_display_name(p["name"])
        if status == "Configured":
            style = "bold green"
        elif "Offline" in status:
            style = "yellow"
        else:
            style = "red"

        console.print(
            f"[dim]{i}.[/dim] [cyan]{provider_label}[/cyan] "
            f"([{style}]{status}[/{style}])"
        )
        console.print(f"   Default Model: [white]{p['model']}[/white]")
        if p["detail"]:
            console.print(f"   [dim]{p['detail']}[/dim]")
        console.print()

    console.print(
        "[dim]Usage:[/dim]  "
        "python app.py --provider <name>\n"
    )


# ── Answer Display ─────────────────────────────────────────────────────────


def display_result(result, question: str | None = None) -> None:
    """Display a RAG query result with rich formatting."""
    if question:
        console.print(Panel(
            Text(question, style="bold white"),
            title=(
                "[bold bright_cyan]Question"
                "[/bold bright_cyan]"
            ),
            border_style="bright_cyan",
            padding=(0, 2),
        ))

    # Answer
    answer_text = result.answer or "No answer generated."
    console.print(Panel(
        Markdown(answer_text),
        title="[bold green]Answer[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))

    # Sources
    sources = result.sources or []
    scores = result.confidence_scores or []
    if sources:
        source_lines: list[str] = []
        for i, src in enumerate(sources, 1):
            doc = src.get("doc_name", "Unknown")
            score = (
                scores[i - 1] if i - 1 < len(scores) else 0.0
            )
            bar = (
                "#" * int(score * 10)
                + "-" * (10 - int(score * 10))
            )
            source_lines.append(
                f"  [cyan]{i}.[/cyan] {doc}  "
                f"[dim]{bar}[/dim]  "
                f"[green]{score:.3f}[/green]"
            )
        console.print(Panel(
            "\n".join(source_lines),
            title=(
                "[bold magenta]Sources"
                "[/bold magenta]"
            ),
            border_style="magenta",
            padding=(0, 2),
        ))

    # Metrics bar
    confidence = (
        sum(scores) / len(scores) if scores else 0.0
    )
    if confidence >= 0.8:
        conf_style = "bold green"
    elif confidence >= 0.5:
        conf_style = "bold yellow"
    else:
        conf_style = "bold red"

    latency_sec = result.latency_ms / 1000.0
    prov_label = (
        f"{result.provider}/{result.model}"
        if result.provider else ""
    )
    console.print(
        f"  [{conf_style}]Confidence:"
        f" {confidence:.0%}[/{conf_style}]"
        f"  [dim]|[/dim]  "
        f"[dim]Latency: {latency_sec:.2f}s[/dim]"
        f"  [dim]|[/dim]  "
        f"[dim]{prov_label}[/dim]\n"
    )


# ── Help ───────────────────────────────────────────────────────────────────


def print_help() -> None:
    """Display available commands."""
    table = Table(
        box=box.ROUNDED,
        title="[bold]Available Commands[/bold]",
        title_style="bright_cyan",
        show_header=True,
        header_style="bold",
        padding=(0, 2),
    )
    table.add_column("Command", style="cyan", width=12)
    table.add_column("Description", style="white")

    table.add_row("help", "Show this help message")
    table.add_row(
        "stats",
        "Show system configuration and index stats",
    )
    table.add_row("clear", "Clear the terminal screen")
    table.add_row("quit", "Exit the application")

    console.print(table)
    console.print()


# ── Stats ──────────────────────────────────────────────────────────────────


def print_stats(pipeline, ds_config) -> None:
    """Display detailed system statistics."""
    doc_dir = Path(ds_config.data_dir)
    doc_count = (
        len(list(doc_dir.glob("*.md")))
        if doc_dir.is_dir() else 0
    )
    chunk_count = pipeline.vector_store.collection_count()
    prov = pipeline.llm_provider

    table = Table(
        box=box.ROUNDED,
        title="[bold]System Statistics[/bold]",
        title_style="bright_cyan",
        show_header=True,
        header_style="bold dim",
        padding=(0, 2),
    )
    table.add_column("Parameter", style="white", width=24)
    table.add_column("Value", style="bold green")

    table.add_row("Dataset", ds_config.name)
    table.add_row("Dataset ID", ds_config.dataset_id)
    table.add_row("Collection", ds_config.collection_name)
    table.add_row("Documents", str(doc_count))
    table.add_row("Chunks Indexed", str(chunk_count))
    table.add_row("", "")
    table.add_row("LLM Provider", prov.provider_name)
    table.add_row("LLM Model", prov.model_name)
    table.add_row(
        "Embedding Model",
        config.EMBEDDING_MODEL.split("/")[-1],
    )
    table.add_row(
        "Embedding Dimensions",
        str(config.EMBEDDING_DIM),
    )
    table.add_row("Vector Store", "ChromaDB (cosine)")
    table.add_row("Temperature", str(config.TEMPERATURE))
    table.add_row(
        "Max Output Tokens",
        str(config.MAX_OUTPUT_TOKENS),
    )
    table.add_row("", "")
    table.add_row("Chunk Size", str(config.CHUNK_SIZE))
    table.add_row("Chunk Overlap", str(config.CHUNK_OVERLAP))
    table.add_row("Top-K", str(config.TOP_K))
    table.add_row(
        "Similarity Threshold",
        f"{config.MIN_SIMILARITY_THRESHOLD:.2f}",
    )

    console.print(table)
    console.print()


# ── Interactive Mode ───────────────────────────────────────────────────────


def interactive_mode(pipeline, ds_config) -> None:
    """Run the interactive query loop."""
    console.print(
        "[dim]Type your question or [bold]help[/bold]"
        " for available commands.[/dim]\n"
    )

    while True:
        try:
            console.print(
                "[bold bright_cyan]RAG >"
                "[/bold bright_cyan] ",
                end="",
            )
            query = input().strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not query:
            continue

        cmd = query.lower()
        if cmd in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break
        if cmd == "clear":
            os.system(
                "cls" if os.name == "nt" else "clear",
            )
            prov = pipeline.llm_provider
            print_banner(
                ds_config,
                prov.provider_name,
                prov.model_name,
            )
            continue
        if cmd == "help":
            print_help()
            continue
        if cmd == "stats":
            print_stats(pipeline, ds_config)
            continue

        console.print()
        with console.status(
            "[bold cyan]Retrieving and generating..."
            "[/bold cyan]",
            spinner="dots",
        ):
            try:
                result = pipeline.query(query)
                display_result(result, question=query)
            except Exception as e:
                _display_generation_error(e, pipeline)


# ── Error Display ──────────────────────────────────────────────────────────


def _display_generation_error(
    error: Exception,
    pipeline,
) -> None:
    """Show a friendly error message for generation failures."""
    prov = pipeline.llm_provider
    error_str = str(error)

    # Detect common error types
    if "quota" in error_str.lower():
        reason = "Quota exhausted."
        fix = (
            "Wait for quota reset or switch provider: "
            "--provider groq"
        )
    elif "401" in error_str or "invalid" in error_str.lower():
        reason = "Invalid API key."
        fix = (
            f"Update your API key in .env for "
            f"{prov.provider_name}."
        )
    elif "429" in error_str or "rate" in error_str.lower():
        reason = "Rate limited."
        fix = "Wait a moment or switch provider."
    elif "timeout" in error_str.lower():
        reason = "Request timed out."
        fix = "Check your network or try again."
    else:
        reason = str(error)
        fix = ""

    console.print(Panel(
        (
            f"[bold]Provider:[/bold]  "
            f"{prov.provider_name}\n"
            f"[bold]Model:[/bold]     "
            f"{prov.model_name}\n"
            f"[bold]Reason:[/bold]    {reason}"
            + (f"\n\n[dim]{fix}[/dim]" if fix else "")
        ),
        title=(
            "[bold red]Generation Failed"
            "[/bold red]"
        ),
        border_style="red",
        padding=(1, 2),
    ))
    console.print()


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the CLI application."""
    parser = argparse.ArgumentParser(
        description=(
            "Generic RAG Framework — Interactive CLI"
        ),
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=config.DEFAULT_DATASET,
        help=(
            "Dataset to use (folder name under data/). "
            f"Default: '{config.DEFAULT_DATASET}'"
        ),
    )
    parser.add_argument(
        "--provider", "-p",
        type=str,
        default=config.DEFAULT_PROVIDER,
        help=(
            "LLM provider (gemini, groq, openai, ollama). "
            f"Default: '{config.DEFAULT_PROVIDER}'"
        ),
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help=(
            "Enable automatic provider fallback on failure"
        ),
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Single query mode: ask one question and exit",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Force re-ingestion of documents",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all available datasets and exit",
    )
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="List all LLM providers and their status",
    )
    args = parser.parse_args()
    args.provider = args.provider.lower().strip()

    # ── List modes ───────────────────────────
    if args.list_datasets:
        print_dataset_list()
        return

    if args.list_providers:
        print_provider_list()
        return

    # ── Validate dataset ─────────────────────
    errors = config.validate_dataset(args.dataset)
    if errors:
        console.print(
            f"[bold red]Error:[/bold red]"
            f" Dataset '{args.dataset}' has issues:\n"
        )
        for err in errors:
            console.print(f"  [red]-[/red] {err}")
        console.print(
            "\nRun [cyan]--list-datasets[/cyan]"
            " to see available datasets."
        )
        sys.exit(1)

    # ── Load dataset config ──────────────────
    ds_config = config.load_dataset_config(args.dataset)

    # ── Initialize provider ──────────────────
    try:
        # Import deferred to avoid loading SDKs early
        from src.providers import (
            get_provider_info,
            is_provider_configured,
        )

        provider_info = get_provider_info(args.provider)
        if not provider_info:
            available = ", ".join(config.PROVIDER_MODELS)
            raise ValueError(
                f"Unknown provider '{args.provider}'. "
                f"Available: {available}"
            )

        configured, detail = is_provider_configured(args.provider)
        if not configured:
            setup_url = provider_info.get("key_url", "")
            help_text = (
                f" Get one at: {setup_url}"
                if setup_url else ""
            )
            raise OSError(
                f"{_provider_display_name(args.provider)} is not configured. "
                f"{detail}.{help_text}"
            )

        from src.rag_pipeline import RAGPipeline

        print_banner(
            ds_config,
            _provider_display_name(args.provider),
            config.PROVIDER_MODELS.get(
                args.provider, "unknown",
            ),
        )

        console.print(
            "[dim]Initializing pipeline...[/dim]",
        )
        pipeline = RAGPipeline(
            dataset_id=args.dataset,
            provider_name=args.provider,
            fallback=args.fallback,
        )
    except (OSError, ConnectionError, ValueError) as exc:
        console.print(
            f"\n[bold red]Error:[/bold red] {exc}\n"
        )
        console.print(
            "Run [cyan]--list-providers[/cyan]"
            " to check provider status."
        )
        sys.exit(1)

    # ── Ingest if needed ─────────────────────
    if args.ingest or not pipeline.is_indexed():
        with console.status(
            "[bold cyan]Ingesting documents...[/bold cyan]",
            spinner="dots",
        ):
            chunk_count = pipeline.ingest()
        console.print(
            f"[green]  Indexed"
            f" [bold]{chunk_count}[/bold] chunks[/green]\n"
        )
    else:
        count = pipeline.vector_store.collection_count()
        console.print(
            f"[green]  Index loaded"
            f" ([bold]{count}[/bold] chunks)[/green]\n"
        )

    # ── Startup summary ──────────────────────
    print_startup_summary(pipeline, ds_config)

    # ── Single query mode ────────────────────
    if args.query:
        with console.status(
            "[bold cyan]Retrieving and generating..."
            "[/bold cyan]",
            spinner="dots",
        ):
            result = pipeline.query(args.query)
        display_result(result, question=args.query)
        return

    # ── Interactive mode ─────────────────────
    interactive_mode(pipeline, ds_config)


if __name__ == "__main__":
    main()
