"""Eval configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    id: str
    provider: str = "openrouter"
    base_url: str | None = None
    api_key_env: str | None = None
    params: dict[str, Any] = field(default_factory=lambda: {"temperature": 0.7, "max_tokens": 512})


@dataclass
class EvalConfig:
    name: str = "eval"
    models: list[ModelConfig] = field(default_factory=list)
    metrics: list[str] = field(default_factory=lambda: ["relevance", "latency"])
    prompts_source: str = "builtin"
    prompt_categories: list[str] | None = None
    judge_model: str = "anthropic/claude-sonnet-4"
    samples_per_prompt: int = 3
    temperature: float = 0.7
    max_tokens: int = 512
    output_path: str | None = None


def load_config(path: str | Path) -> EvalConfig:
    """Load and validate an eval config from YAML."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Empty config file: {path}")

    models = []
    for m in raw.get("models", []):
        models.append(ModelConfig(
            id=m["id"],
            provider=m.get("provider", "openrouter"),
            base_url=m.get("base_url"),
            api_key_env=m.get("api_key_env"),
            params={
                "temperature": m.get("temperature", raw.get("settings", {}).get("temperature", 0.7)),
                "max_tokens": m.get("max_tokens", raw.get("settings", {}).get("max_tokens", 512)),
            },
        ))

    settings = raw.get("settings", {})

    return EvalConfig(
        name=raw.get("name", "eval"),
        models=models,
        metrics=raw.get("metrics", ["relevance", "latency"]),
        prompts_source=raw.get("prompts", {}).get("source", "builtin"),
        prompt_categories=raw.get("prompts", {}).get("categories"),
        judge_model=settings.get("judge_model", "anthropic/claude-sonnet-4"),
        samples_per_prompt=settings.get("samples_per_prompt", 3),
        temperature=settings.get("temperature", 0.7),
        max_tokens=settings.get("max_tokens", 512),
        output_path=raw.get("output"),
    )
