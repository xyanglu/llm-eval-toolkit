"""JSON and CLI reporters."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ..evaluator import EvalRun

console = Console()


def save_json(run: EvalRun, path: str | Path) -> Path:
    """Save eval results as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "name": run.name,
        "timestamp": run.timestamp,
        "config": run.config,
        "results": [asdict(r) for r in run.results],
        "model_summaries": run.model_summaries,
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    return path


def load_json(path: str | Path) -> EvalRun:
    """Load eval results from JSON."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    from .evaluator import PromptResult
    results = [PromptResult(**r) for r in data["results"]]

    run = EvalRun(
        name=data["name"],
        timestamp=data["timestamp"],
        config=data["config"],
        results=results,
        model_summaries=data.get("model_summaries", {}),
    )
    return run


def print_summary(run: EvalRun) -> None:
    """Print a rich summary table of eval results."""
    console.print(f"\n[bold]Eval Run: {run.name}[/bold]")
    console.print(f"Timestamp: {run.timestamp}\n")

    # Model summary table
    table = Table(title="Model Comparison")
    table.add_column("Model", style="cyan")
    table.add_column("Prompts", justify="right")
    table.add_column("Errors", justify="right")
    table.add_column("Avg Latency", justify="right")
    table.add_column("Tokens/sec", justify="right")

    # Collect all score keys
    score_keys = set()
    for summary in run.model_summaries.values():
        score_keys.update(k for k in summary if k.startswith("avg_"))
    for key in sorted(score_keys):
        label = key.replace("avg_", "").replace("_", " ").title()
        table.add_column(label, justify="right")

    for model_id, summary in run.model_summaries.items():
        row = [
            model_id,
            str(summary.get("total_prompts", 0)),
            str(summary.get("errors", 0)),
            f"{summary.get('avg_latency_ms', 0):.0f}ms",
            f"{summary.get('tokens_per_second', 0):.1f}",
        ]
        for key in sorted(score_keys):
            row.append(str(summary.get(key, "-")))
        table.add_row(*row)

    console.print(table)

    # Per-category breakdown
    if len(run.model_summaries) > 1:
        for model_id in run.model_summaries:
            model_results = [r for r in run.results if r.model_id == model_id]
            cat_table = Table(title=f"\n{model_id} — By Category")
            cat_table.add_column("Category", style="cyan")

            # Find all score keys
            all_score_keys = set()
            for r in model_results:
                all_score_keys.update(
                    k for k, v in r.scores.items() if isinstance(v, (int, float))
                )

            for key in sorted(all_score_keys):
                label = key.replace("_", " ").title()
                cat_table.add_column(label, justify="right")

            categories = sorted(set(r.category for r in model_results))
            for cat in categories:
                cat_results = [r for r in model_results if r.category == cat]
                row = [cat]
                for key in sorted(all_score_keys):
                    values = [r.scores.get(key) for r in cat_results if isinstance(r.scores.get(key), (int, float))]
                    if values:
                        row.append(f"{sum(values)/len(values):.2f}")
                    else:
                        row.append("-")
                cat_table.add_row(*row)

            console.print(cat_table)
