"""CLI entry point for llm-eval."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler

from .config import load_config
from .evaluator import run_eval_sync
from .reporters.json_reporter import load_json, print_summary, save_json

app = typer.Typer(
    name="llm-eval",
    help="LLM Eval Toolkit — evaluate LLM quality across providers.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    config_path: Path = typer.Argument(..., help="Path to eval config YAML"),
    output: Path = typer.Option(None, "-o", "--output", help="Output JSON path"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable verbose logging"),
):
    """Run an evaluation."""
    import logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s", handlers=[RichHandler(console=console)])

    console.print(f"[bold]Loading config:[/bold] {config_path}")
    config = load_config(config_path)

    console.print(f"[bold]Models:[/bold] {', '.join(m.id for m in config.models)}")
    console.print(f"[bold]Metrics:[/bold] {', '.join(config.metrics)}")
    console.print(f"[bold]Prompts:[/bold] builtin (categories: {config.prompt_categories or 'all'})")
    console.print()

    run_obj = run_eval_sync(config)

    # Save results
    out_path = output or Path(f"outputs/{config.name}_{run_obj.timestamp[:19]}.json")
    saved = save_json(run_obj, out_path)
    console.print(f"\n[dim]Results saved to:[/dim] {saved}")

    # Print summary
    print_summary(run_obj)


@app.command()
def compare(
    run1: Path = typer.Argument(..., help="First eval results JSON"),
    run2: Path = typer.Argument(..., help="Second eval results JSON"),
):
    """Compare two eval runs."""
    r1 = load_json(run1)
    r2 = load_json(run2)

    table_title = f"Comparison: {r1.name} vs {r2.name}"
    from rich.table import Table
    table = Table(title=table_title)
    table.add_column("Model", style="cyan")
    table.add_column("Metric", style="white")
    table.add_column(r1.timestamp[:10], justify="right", style="blue")
    table.add_column(r2.timestamp[:10], justify="right", style="green")
    table.add_column("Delta", justify="right")

    all_models = set(r1.model_summaries.keys()) | set(r2.model_summaries.keys())
    for model_id in sorted(all_models):
        s1 = r1.model_summaries.get(model_id, {})
        s2 = r2.model_summaries.get(model_id, {})

        all_keys = set(s1.keys()) | set(s2.keys())
        for key in sorted(all_keys):
            v1 = s1.get(key)
            v2 = s2.get(key)
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                delta = v2 - v1
                delta_str = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
                style = "red" if delta < 0 and "error" not in key.lower() else "green" if delta > 0 and "error" not in key.lower() else "white"
                table.add_row(
                    model_id,
                    key.replace("_", " ").title(),
                    f"{v1:.2f}",
                    f"{v2:.2f}",
                    f"[{style}]{delta_str}[/{style}]",
                )

    console.print(table)


@app.command()
def report(
    results_path: Path = typer.Argument(..., help="Eval results JSON"),
    format: str = typer.Option("cli", "-f", "--format", help="Output format: cli or json"),
):
    """Display an eval report."""
    run_obj = load_json(results_path)
    print_summary(run_obj)


@app.command(name="config-example")
def config_example():
    """Print an example config to stdout."""
    example = """name: "quick-eval"
models:
  - id: "anthropic/claude-sonnet-4"
    provider: openrouter
  - id: "google/gemini-2.0-flash-001"
    provider: openrouter
  - id: "meta-llama/llama-3.3-70b-instruct"
    provider: openrouter

metrics:
  - relevance
  - hallucination
  - instruction_following
  - safety
  - latency

prompts:
  source: builtin
  categories:
    - factual_qa
    - reasoning
    - creative_writing
    - safety
    - instruction_following

settings:
  samples_per_prompt: 3
  temperature: 0.7
  max_tokens: 512
  judge_model: "anthropic/claude-sonnet-4"
"""
    console.print(example)


if __name__ == "__main__":
    app()
