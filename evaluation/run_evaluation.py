"""Main evaluation runner for the RAG pipeline.

Usage::

    python -m evaluation.run_evaluation            # quantitative only
    python -m evaluation.run_evaluation --human     # + qualitative eval
    python -m evaluation.run_evaluation --questions 5  # first N questions

Generates three output files under ``results/``:

- ``evaluation.json`` — full structured results
- ``evaluation.csv``  — flat CSV (one row per question)
- ``evaluation.md``   — formatted Markdown report with retrieval
  visualizations, per-question breakdowns, and aggregate stats
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# ── project imports ──────────────────────────────────────────────────────────
from evaluation.metrics import compute_all_metrics
from evaluation.qualitative import run_qualitative_evaluation
from evaluation.retrieval_eval import evaluate_retrieval

try:
    from src import config as rag_config
    from src.rag_pipeline import RAGPipeline
except ImportError:
    rag_config = None  # type: ignore[assignment]
    RAGPipeline = None  # type: ignore[assignment,misc]



# ── paths ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

console = Console()

# ── helpers ──────────────────────────────────────────────────────────────────


def _load_qa_dataset(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load the QA dataset JSON file.

    Expected format — list of dicts, each with keys:
        question, expected_answer, key_facts, relevant_doc_ids
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Support both flat list and nested {"questions": [...]} format
    if isinstance(data, dict) and "questions" in data:
        questions = data["questions"]
    elif isinstance(data, list):
        questions = data
    else:
        raise ValueError(
            "qa_dataset.json must be a JSON array"
            " or contain a 'questions' key."
        )
    if limit is not None:
        questions = questions[:limit]
    return questions


def _ensure_results_dir() -> None:
    """Create the results directory if it does not exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _truncate(text: str, max_len: int = 150) -> str:
    """Truncate text for display."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


# ── output generators ───────────────────────────────────────────────────────


def _write_json(results: list[dict[str, Any]], aggregates: dict[str, Any]) -> Path:
    """Write full structured results to JSON."""
    out_path = RESULTS_DIR / "evaluation.json"
    payload = {
        "aggregate_statistics": aggregates,
        "per_question_results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return out_path


def _write_csv(results: list[dict[str, Any]]) -> Path:
    """Write a flat CSV with one row per question."""
    out_path = RESULTS_DIR / "evaluation.csv"

    # Determine columns from first result
    fieldnames = [
        "question_index",
        "question",
        "expected_answer",
        "generated_answer",
        "keyword_f1",
        "rouge_l",
        "semantic_similarity",
        "fact_coverage",
        "hallucination_rate",
        "composite_score",
        "precision_at_k",
        "recall_at_k",
        "mrr",
        "latency_ms",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            flat: dict[str, Any] = {
                "question_index": r.get("question_index"),
                "question": r.get("question"),
                "expected_answer": r.get("expected_answer"),
                "generated_answer": r.get("generated_answer"),
                "latency_ms": r.get("latency_ms"),
            }
            flat.update(r.get("metrics", {}))
            flat.update(r.get("retrieval_metrics", {}))
            writer.writerow(flat)
    return out_path


def _write_markdown(
    results: list[dict[str, Any]],
    aggregates: dict[str, Any],
) -> Path:
    """Generate a formatted Markdown evaluation report."""
    out_path = RESULTS_DIR / "evaluation.md"
    lines: list[str] = []

    lines.append("# RAG Evaluation Report\n")
    lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**Questions evaluated:** {len(results)}\n")

    # ── Aggregate Summary ────────────────────────────────────────────────
    lines.append("\n## Aggregate Performance Summary\n")
    lines.append("| Metric | Mean | Min | Max |")
    lines.append("|--------|------|-----|-----|")

    metric_keys = [
        ("keyword_f1", "Keyword F1"),
        ("rouge_l", "ROUGE-L"),
        ("semantic_similarity", "Semantic Similarity"),
        ("fact_coverage", "Fact Coverage"),
        ("hallucination_rate", "Hallucination Rate"),
        ("composite_score", "Composite RAG Score"),
    ]

    retrieval_keys = [
        ("precision_at_k", "Precision@k"),
        ("recall_at_k", "Recall@k"),
        ("mrr", "MRR"),
    ]

    for key, label in metric_keys + retrieval_keys:
        stats = aggregates.get(key, {})
        mean = stats.get("mean", "—")
        mn = stats.get("min", "—")
        mx = stats.get("max", "—")
        mean_str = f"{mean:.4f}" if isinstance(mean, (int, float)) else str(mean)
        min_str = f"{mn:.4f}" if isinstance(mn, (int, float)) else str(mn)
        max_str = f"{mx:.4f}" if isinstance(mx, (int, float)) else str(mx)
        lines.append(f"| {label} | {mean_str} | {min_str} | {max_str} |")

    lines.append("")

    # ── Per-Question Results Table ───────────────────────────────────────
    lines.append("\n## Per-Question Results\n")
    lines.append(
        "| # | Question | Keyword F1 | ROUGE-L | Sem. Sim. | Fact Cov. "
        "| Halluc. | Composite | P@k | R@k | MRR | Latency (ms) |"
    )
    lines.append(
        "|---|----------|------------|---------|-----------|---------- "
        "|---------|-----------|-----|-----|-----|--------------|"
    )

    for r in results:
        idx = r.get("question_index", "?")
        q = _truncate(r.get("question", ""), 50)
        m = r.get("metrics", {})
        rm = r.get("retrieval_metrics", {})
        lat = r.get("latency_ms", "—")
        lat_str = f"{lat:.0f}" if isinstance(lat, (int, float)) else str(lat)

        def _fmt(val: object, precision: int = 4) -> str:
            """Safely format a metric value."""
            if isinstance(val, (int, float)):
                return f"{val:.{precision}f}"
            return str(val) if val else "—"

        lines.append(
            f"| {idx} | {q} "
            f"| {_fmt(m.get('keyword_f1'))} "
            f"| {_fmt(m.get('rouge_l'))} "
            f"| {_fmt(m.get('semantic_similarity'))} "
            f"| {_fmt(m.get('fact_coverage'))} "
            f"| {_fmt(m.get('hallucination_rate'))} "
            f"| {_fmt(m.get('composite_score'))} "
            f"| {_fmt(rm.get('precision_at_k'))} "
            f"| {_fmt(rm.get('recall_at_k'))} "
            f"| {_fmt(rm.get('mrr'))} "
            f"| {lat_str} |"
        )

    lines.append("")

    # ── Retrieval Visualization ──────────────────────────────────────────
    lines.append("\n## Retrieval Visualization\n")

    for r in results:
        idx = r.get("question_index", "?")
        q = r.get("question", "")
        lines.append(f"### Question {idx}: {q}\n")

        # Retrieved chunks
        chunks = r.get("retrieved_chunks", [])
        if chunks:
            lines.append("**Retrieved Chunks:**\n")
            for i, chunk in enumerate(chunks, 1):
                source = chunk.get("source", "unknown")
                chunk_idx = chunk.get("chunk_index", "?")
                score = chunk.get("score", "—")
                score_str = (
                    f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
                )
                snippet = _truncate(chunk.get("content", ""), 150)
                lines.append(
                    f"{i}. **{source}** (Chunk {chunk_idx}) — "
                    f"Similarity: {score_str}"
                )
                lines.append(f"   > {snippet}\n")
        else:
            lines.append("*No retrieved chunks.*\n")

        # Generated answer
        gen = r.get("generated_answer", "")
        lines.append(f"**Generated Answer:** {gen}\n")

        # Expected answer
        exp = r.get("expected_answer", "")
        lines.append(f"**Expected Answer:** {exp}\n")

        # Metrics summary line
        m = r.get("metrics", {})
        rm = r.get("retrieval_metrics", {})
        metric_parts = []
        for key, label in metric_keys:
            val = m.get(key)
            if val is not None:
                metric_parts.append(f"{label}: {val:.4f}")
        for key, label in retrieval_keys:
            val = rm.get(key)
            if val is not None:
                metric_parts.append(f"{label}: {val:.4f}")
        lines.append(f"**Metrics:** {' | '.join(metric_parts)}\n")
        lines.append("---\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


# ── aggregation ──────────────────────────────────────────────────────────────


def _compute_aggregates(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute mean / min / max for each metric across all questions."""
    metric_keys = [
        "keyword_f1",
        "rouge_l",
        "semantic_similarity",
        "fact_coverage",
        "hallucination_rate",
        "composite_score",
    ]
    retrieval_keys = ["precision_at_k", "recall_at_k", "mrr"]

    agg: dict[str, Any] = {}

    for key in metric_keys:
        values = [
            r["metrics"][key]
            for r in results
            if "metrics" in r and key in r["metrics"]
        ]
        if values:
            agg[key] = {
                "mean": round(sum(values) / len(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
            }

    for key in retrieval_keys:
        values = [
            r["retrieval_metrics"][key]
            for r in results
            if "retrieval_metrics" in r and key in r["retrieval_metrics"]
        ]
        if values:
            agg[key] = {
                "mean": round(sum(values) / len(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
            }

    # Latency stats
    latencies = [r["latency_ms"] for r in results if "latency_ms" in r]
    if latencies:
        agg["latency_ms"] = {
            "mean": round(sum(latencies) / len(latencies), 1),
            "min": round(min(latencies), 1),
            "max": round(max(latencies), 1),
        }

    return agg


# ── console summary ─────────────────────────────────────────────────────────


def _print_summary_table(
    results: list[dict[str, Any]],
    aggregates: dict[str, Any],
) -> None:
    """Print a rich summary table to the console."""
    # Per-question table
    table = Table(title="Per-Question Evaluation Results", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Question", max_width=40)
    table.add_column("KW-F1", justify="right")
    table.add_column("ROUGE-L", justify="right")
    table.add_column("Sem.Sim", justify="right")
    table.add_column("FactCov", justify="right")
    table.add_column("Halluc.", justify="right")
    table.add_column("Composite", justify="right", style="bold")
    table.add_column("P@k", justify="right")
    table.add_column("R@k", justify="right")
    table.add_column("MRR", justify="right")

    for r in results:
        m = r.get("metrics", {})
        rm = r.get("retrieval_metrics", {})
        table.add_row(
            str(r.get("question_index", "?")),
            _truncate(r.get("question", ""), 38),
            f"{m.get('keyword_f1', 0):.3f}",
            f"{m.get('rouge_l', 0):.3f}",
            f"{m.get('semantic_similarity', 0):.3f}",
            f"{m.get('fact_coverage', 0):.3f}",
            f"{m.get('hallucination_rate', 0):.3f}",
            f"{m.get('composite_score', 0):.3f}",
            f"{rm.get('precision_at_k', 0):.3f}",
            f"{rm.get('recall_at_k', 0):.3f}",
            f"{rm.get('mrr', 0):.3f}",
        )

    console.print(table)

    # Aggregate table
    agg_table = Table(title="Aggregate Statistics", show_lines=True)
    agg_table.add_column("Metric", style="cyan")
    agg_table.add_column("Mean", justify="right")
    agg_table.add_column("Min", justify="right")
    agg_table.add_column("Max", justify="right")

    display_labels = {
        "keyword_f1": "Keyword F1",
        "rouge_l": "ROUGE-L",
        "semantic_similarity": "Semantic Similarity",
        "fact_coverage": "Fact Coverage",
        "hallucination_rate": "Hallucination Rate",
        "composite_score": "Composite RAG Score",
        "precision_at_k": "Precision@k",
        "recall_at_k": "Recall@k",
        "mrr": "MRR",
        "latency_ms": "Latency (ms)",
    }

    for key, label in display_labels.items():
        stats = aggregates.get(key, {})
        if stats:
            agg_table.add_row(
                label,
                f"{stats['mean']:.4f}",
                f"{stats['min']:.4f}",
                f"{stats['max']:.4f}",
            )

    console.print(agg_table)


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for ``python -m evaluation.run_evaluation``."""
    default_ds = (
        rag_config.DEFAULT_DATASET if rag_config else "mars"
    )
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation pipeline."
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=default_ds,
        help=(
            "Dataset to evaluate "
            f"(default: '{default_ds}')"
        ),
    )
    default_prov = (
        rag_config.DEFAULT_PROVIDER if rag_config else "groq"
    )
    parser.add_argument(
        "--provider", "-p",
        type=str,
        default=default_prov,
        help=(
            "LLM provider "
            f"(default: '{default_prov}')"
        ),
    )
    parser.add_argument(
        "--human",
        action="store_true",
        help="Launch interactive qualitative evaluation after quantitative.",
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=None,
        help="Number of questions to evaluate (default: all).",
    )
    args = parser.parse_args()
    args.provider = args.provider.lower().strip()

    try:
        from src.providers import get_provider_info

        provider_models = (
            rag_config.PROVIDER_MODELS
            if rag_config else {
                "groq": "",
                "gemini": "",
                "openai": "",
                "ollama": "",
            }
        )
        if not get_provider_info(args.provider):
            available = ", ".join(provider_models)
            console.print(
                f"[red]Error: Unknown provider '{args.provider}'. "
                f"Available: {available}[/red]"
            )
            sys.exit(1)
    except ImportError:
        pass

    # Resolve dataset paths
    if rag_config:
        ds_cfg = rag_config.load_dataset_config(args.dataset)
        qa_path = Path(ds_cfg.qa_dataset_path)
    else:
        qa_path = (
            BASE_DIR / "data" / args.dataset / "qa_dataset.json"
        )

    # ── 1. Load dataset ──────────────────────────────────────────────────
    console.rule("[bold cyan]RAG Evaluation Pipeline[/bold cyan]")

    if not qa_path.exists():
        console.print(
            f"[red]Error: QA dataset not found at {qa_path}[/red]"
        )
        sys.exit(1)

    console.print(
        f"Loading QA dataset from [cyan]{qa_path}[/cyan] ..."
    )
    qa_items = _load_qa_dataset(qa_path, limit=args.questions)
    console.print(f"Loaded [green]{len(qa_items)}[/green] questions.\n")

    # ── 2. Initialize RAG pipeline ───────────────────────────────────────
    if RAGPipeline is None:
        console.print(
            "[red]Error: Could not import src.rag_pipeline.RAGPipeline. "
            "Make sure the RAG pipeline is installed.[/red]"
        )
        sys.exit(1)



    console.print(
        f"Initializing RAG pipeline "
        f"(provider={args.provider}) ..."
    )
    try:
        pipeline = RAGPipeline(
            dataset_id=args.dataset,
            provider_name=args.provider,
        )
        embed_fn = pipeline.embedding_model.embed_text
    except Exception as exc:
        console.print(
            f"[red]Failed to initialize pipeline: {exc}[/red]"
        )
        sys.exit(1)

    console.print("[green]Pipeline ready.[/green]")

    # Auto-ingest if needed
    if not pipeline.is_indexed():
        console.print("Ingesting documents …")
        chunk_count = pipeline.ingest()
        console.print(f"[green]Indexed {chunk_count} chunks.[/green]\n")
    else:
        console.print(
            "Using existing index"
            f" ({pipeline.vector_store.collection_count()}"
            " chunks).\n"
        )

    # ── 3. Run evaluation ────────────────────────────────────────────────
    results: list[dict[str, Any]] = []

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating", total=len(qa_items))

        for idx, qa in enumerate(qa_items):
            question = qa.get("question", "")
            expected_answer = qa.get("expected_answer", "")
            key_facts = qa.get("key_facts", [])
            relevant_doc_ids = qa.get("relevant_doc_ids", [])

            try:
                # Query the RAG pipeline
                rag_result = pipeline.query(question)

                generated_answer = rag_result.answer
                retrieved_chunks = rag_result.retrieved_chunks
                latency_ms = rag_result.latency_ms

                # Extract context text from chunks
                context_texts = [
                    chunk.get("content", "") for chunk in retrieved_chunks
                ]

                # Compute quantitative metrics
                metrics = compute_all_metrics(
                    generated=generated_answer,
                    expected=expected_answer,
                    key_facts=key_facts,
                    context_chunks=context_texts,
                    embed_fn=embed_fn,
                )

                # Compute retrieval metrics
                ret_metrics = evaluate_retrieval(
                    retrieved_chunks=retrieved_chunks,
                    relevant_doc_ids=relevant_doc_ids,
                )

                result_entry: dict[str, Any] = {
                    "question_index": idx,
                    "question": question,
                    "expected_answer": expected_answer,
                    "generated_answer": generated_answer,
                    "retrieved_chunks": retrieved_chunks,
                    "metrics": metrics,
                    "retrieval_metrics": ret_metrics,
                    "latency_ms": latency_ms,
                }
                results.append(result_entry)

            except Exception as exc:
                console.print(
                    f"\n[red]Error on question {idx}: {exc}[/red]"
                )
                traceback.print_exc()
                results.append(
                    {
                        "question_index": idx,
                        "question": question,
                        "expected_answer": expected_answer,
                        "generated_answer": "",
                        "retrieved_chunks": [],
                        "metrics": {},
                        "retrieval_metrics": {},
                        "latency_ms": 0,
                        "error": str(exc),
                    }
                )

            progress.advance(task)

    # ── 4. Aggregate ─────────────────────────────────────────────────────
    successful_results = [r for r in results if "error" not in r]
    aggregates = _compute_aggregates(successful_results)

    # ── 5. Console summary ───────────────────────────────────────────────
    console.print()
    _print_summary_table(results, aggregates)

    # ── 6. Write output files ────────────────────────────────────────────
    _ensure_results_dir()

    json_path = _write_json(results, aggregates)
    csv_path = _write_csv(results)
    md_path = _write_markdown(results, aggregates)

    console.print(f"\n[green]✓[/green] JSON report → [cyan]{json_path}[/cyan]")
    console.print(f"[green]✓[/green] CSV report  → [cyan]{csv_path}[/cyan]")
    console.print(f"[green]✓[/green] MD report   → [cyan]{md_path}[/cyan]")

    # ── 7. Optional human evaluation ─────────────────────────────────────
    if args.human:
        console.print()
        console.rule("[bold cyan]Qualitative Evaluation[/bold cyan]")

        eval_items = [
            {
                "question_index": r["question_index"],
                "question": r["question"],
                "expected_answer": r["expected_answer"],
                "generated_answer": r["generated_answer"],
                "retrieved_chunks": r.get("retrieved_chunks", []),
            }
            for r in results
        ]
        run_qualitative_evaluation(eval_items)

    console.print("\n[bold green]Evaluation complete.[/bold green]")


if __name__ == "__main__":
    main()
