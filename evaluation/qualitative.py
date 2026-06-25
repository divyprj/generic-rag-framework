"""Interactive CLI for human / qualitative evaluation of RAG answers.

Uses ``rich`` for formatted console output.  The evaluator is shown each
question, the expected answer, the generated answer, and retrieved context
snippets, then asked to score four rubric dimensions on a 1-5 scale:

1. **Coherence** — Is the answer well-structured and readable?
2. **Completeness** — Does it cover all key facts from the expected answer?
3. **Factual Accuracy** — Are all stated facts correct?
4. **Groundedness** — Is the answer supported by the retrieved context?

Results are saved incrementally to ``results/human_evaluation.json`` and
can be resumed if interrupted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
from rich.table import Table

# ── constants ────────────────────────────────────────────────────────────────

RUBRIC_DIMENSIONS: list[dict[str, str]] = [
    {
        "key": "coherence",
        "label": "Coherence",
        "description": "Is the answer well-structured and readable?",
    },
    {
        "key": "completeness",
        "label": "Completeness",
        "description": "Does it cover all key facts from the expected answer?",
    },
    {
        "key": "factual_accuracy",
        "label": "Factual Accuracy",
        "description": "Are all stated facts correct?",
    },
    {
        "key": "groundedness",
        "label": "Groundedness",
        "description": "Is the answer supported by the retrieved context?",
    },
]

_BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = _BASE_DIR / "results"
RESULTS_FILE = RESULTS_DIR / "human_evaluation.json"

console = Console()

# ── persistence helpers ──────────────────────────────────────────────────────


def _load_existing_results() -> list[dict[str, Any]]:
    """Load previously saved evaluation results, if any."""
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except (OSError, json.JSONDecodeError):
            pass
    return []


def _save_results(results: list[dict[str, Any]]) -> None:
    """Persist evaluation results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def _already_evaluated_ids(results: list[dict[str, Any]]) -> set[int]:
    """Return the set of question indices already evaluated."""
    return {r["question_index"] for r in results if "question_index" in r}


# ── display helpers ──────────────────────────────────────────────────────────


def _truncate(text: str, max_len: int = 200) -> str:
    """Truncate text to *max_len* characters, appending '…' if trimmed."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _display_question_panel(
    index: int,
    total: int,
    question: str,
    expected_answer: str,
    generated_answer: str,
    context_chunks: list[dict[str, Any]],
) -> None:
    """Render a rich panel with all information the evaluator needs."""
    console.rule(f"[bold cyan]Question {index + 1} / {total}[/bold cyan]")

    # Question
    console.print(
        Panel(question, title="[bold yellow]Question[/bold yellow]", expand=True)
    )

    # Expected answer
    console.print(
        Panel(
            expected_answer,
            title="[bold green]Expected Answer[/bold green]",
            expand=True,
        )
    )

    # Generated answer
    console.print(
        Panel(
            generated_answer,
            title="[bold magenta]Generated Answer[/bold magenta]",
            expand=True,
        )
    )

    # Retrieved context snippets
    if context_chunks:
        ctx_table = Table(
            title="Retrieved Context Snippets",
            show_lines=True,
            expand=True,
        )
        ctx_table.add_column("#", style="dim", width=3)
        ctx_table.add_column("Source", style="cyan", max_width=30)
        ctx_table.add_column("Snippet", ratio=1)
        ctx_table.add_column("Score", justify="right", width=8)

        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.get("source", "unknown")
            content = _truncate(chunk.get("content", ""), 200)
            score = chunk.get("score", "—")
            score_str = (
                f"{score:.4f}"
                if isinstance(score, (int, float))
                else str(score)
            )
            ctx_table.add_row(str(i), source, content, score_str)

        console.print(ctx_table)
    else:
        console.print("[dim]No retrieved context available.[/dim]")

    console.print()


# ── scoring ──────────────────────────────────────────────────────────────────


def _collect_scores() -> dict[str, int]:
    """Prompt the evaluator for 1-5 scores on each rubric dimension."""
    scores: dict[str, int] = {}

    console.print("[bold]Rate the generated answer on each dimension (1-5):[/bold]\n")

    for dim in RUBRIC_DIMENSIONS:
        while True:
            try:
                value = IntPrompt.ask(
                    f"  [cyan]{dim['label']}[/cyan] — {dim['description']}"
                )
                if 1 <= value <= 5:
                    scores[dim["key"]] = value
                    break
                else:
                    console.print(
                        "  [red]Enter an integer"
                        " between 1 and 5.[/red]"
                    )
            except Exception:
                console.print("  [red]Invalid input. Enter an integer 1-5.[/red]")

    return scores


def _collect_notes() -> str:
    """Optionally collect free-text notes from the evaluator."""
    notes = Prompt.ask(
        "\n  [dim]Optional notes (press Enter to skip)[/dim]",
        default="",
    )
    return notes.strip()


# ── public API ───────────────────────────────────────────────────────────────


def run_qualitative_evaluation(
    evaluation_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Launch the interactive human evaluation CLI.

    Args:
        evaluation_items: List of dicts, each with keys:
            - ``question`` (str)
            - ``expected_answer`` (str)
            - ``generated_answer`` (str)
            - ``retrieved_chunks`` (list[dict])
            - ``question_index`` (int) — 0-based index

    Returns:
        Full list of evaluation results (including any previously saved).
    """
    results = _load_existing_results()
    done_ids = _already_evaluated_ids(results)
    total = len(evaluation_items)

    # Filter to unevaluated questions
    pending = [
        item for item in evaluation_items
        if item["question_index"] not in done_ids
    ]

    if not pending:
        console.print(
            "[bold green]All questions have already been evaluated![/bold green]"
        )
        _show_summary(results)
        return results

    console.print(
        f"\n[bold]Starting human evaluation — "
        f"{len(pending)} questions remaining "
        f"({len(done_ids)} already completed).[/bold]\n"
    )

    for item in pending:
        idx = item["question_index"]

        _display_question_panel(
            index=idx,
            total=total,
            question=item["question"],
            expected_answer=item["expected_answer"],
            generated_answer=item["generated_answer"],
            context_chunks=item.get("retrieved_chunks", []),
        )

        scores = _collect_scores()
        notes = _collect_notes()

        result_entry: dict[str, Any] = {
            "question_index": idx,
            "question": item["question"],
            **scores,
            "notes": notes,
        }
        results.append(result_entry)

        # Save incrementally so progress is never lost
        _save_results(results)
        console.print(
            f"\n[green]✓ Saved evaluation for question {idx + 1}/{total}.[/green]\n"
        )

    _show_summary(results)
    return results


def _show_summary(results: list[dict[str, Any]]) -> None:
    """Display summary statistics across all completed evaluations."""
    if not results:
        return

    console.rule("[bold cyan]Evaluation Summary[/bold cyan]")

    summary_table = Table(title="Mean Human Scores", show_lines=True)
    summary_table.add_column("Dimension", style="cyan")
    summary_table.add_column("Mean", justify="right")
    summary_table.add_column("Min", justify="right")
    summary_table.add_column("Max", justify="right")

    for dim in RUBRIC_DIMENSIONS:
        values = [
            r[dim["key"]] for r in results if dim["key"] in r
        ]
        if values:
            mean_val = sum(values) / len(values)
            summary_table.add_row(
                dim["label"],
                f"{mean_val:.2f}",
                str(min(values)),
                str(max(values)),
            )
        else:
            summary_table.add_row(dim["label"], "—", "—", "—")

    console.print(summary_table)
    console.print(
        f"\n[dim]Results saved to {RESULTS_FILE.resolve()}[/dim]\n"
    )
