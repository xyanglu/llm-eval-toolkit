"""Main eval orchestrator — runs prompts through models and collects scores."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import EvalConfig
from .datasets.golden import get_prompts
from .metrics.hallucination import check_consistency
from .metrics.judge import check_hallucination, score_relevance, score_safety
from .metrics.latency import compute_latency_stats
from .metrics.safety import detect_refusal, evaluate_safety
from .models import ModelClient, ModelResponse

logger = logging.getLogger(__name__)


@dataclass
class PromptResult:
    prompt_id: str
    category: str
    prompt: str
    model_id: str
    completion: str
    latency_ms: float
    total_tokens: int
    error: str | None = None
    scores: dict = field(default_factory=dict)


@dataclass
class EvalRun:
    name: str
    timestamp: str
    config: dict
    results: list[PromptResult] = field(default_factory=list)
    model_summaries: dict = field(default_factory=dict)


async def run_eval(config: EvalConfig) -> EvalRun:
    """Run the full evaluation pipeline."""
    run = EvalRun(
        name=config.name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        config={
            "models": [m.id for m in config.models],
            "metrics": config.metrics,
            "categories": config.prompt_categories,
            "samples_per_prompt": config.samples_per_prompt,
        },
    )

    # Load prompts
    prompts = get_prompts(config.prompt_categories)
    logger.info(f"Loaded {len(prompts)} prompts")

    # Create client
    client = ModelClient()

    for model_cfg in config.models:
        model_id = model_cfg.id
        logger.info(f"Evaluating model: {model_id}")

        for prompt_data in prompts:
            prompt = prompt_data["prompt"]
            prompt_id = prompt_data["id"]
            category = prompt_data["category"]

            # Generate main response
            response = await client.complete(
                model=model_id,
                prompt=prompt,
                temperature=model_cfg.params["temperature"],
                max_tokens=model_cfg.params["max_tokens"],
            )

            result = PromptResult(
                prompt_id=prompt_id,
                category=category,
                prompt=prompt,
                model_id=model_id,
                completion=response.completion,
                latency_ms=response.latency_ms,
                total_tokens=response.total_tokens,
                error=response.error,
            )

            # Run requested metrics
            if response.error:
                result.scores = {"error": response.error}
                run.results.append(result)
                continue

            if "relevance" in config.metrics:
                score = await score_relevance(client, config.judge_model, prompt, response.completion)
                result.scores["relevance"] = score.get("score", 3)

            if "hallucination" in config.metrics and config.samples_per_prompt >= 2:
                # Multi-sample consistency check
                samples = await client.complete_multi_sample(
                    model=model_id,
                    prompt=prompt,
                    n=config.samples_per_prompt,
                    temperature=0.9,
                    max_tokens=model_cfg.params["max_tokens"],
                )
                consistency = check_consistency(samples)
                result.scores["consistency"] = consistency.consistency_score
                result.scores["hallucination_detected"] = consistency.is_likely_hallucination

            if "instruction_following" in config.metrics:
                from .metrics.judge import score_instruction_following
                score = await score_instruction_following(client, config.judge_model, prompt, response.completion)
                result.scores["instruction_following"] = score.get("score", 3)

            if "safety" in config.metrics:
                should_refuse = prompt_data.get("ground_truth", "").startswith("REFUSAL")
                safety_eval = evaluate_safety(response.completion, should_refuse)
                result.scores["safety_correct"] = safety_eval.correct_handling
                result.scores["safety_false_positive"] = safety_eval.false_positive
                result.scores["safety_false_negative"] = safety_eval.false_negative
                result.scores["is_refusal"] = safety_eval.is_refusal

            if "latency" in config.metrics:
                result.scores["latency_ms"] = response.latency_ms

            run.results.append(result)
            logger.info(f"  {prompt_id} ({category}): latency={response.latency_ms:.0f}ms, scores={result.scores}")

    # Compute model summaries
    for model_cfg in config.models:
        model_id = model_cfg.id
        model_results = [r for r in run.results if r.model_id == model_id]

        latency_stats = compute_latency_stats(
            [ModelResponse(
                model_id=r.model_id,
                prompt=r.prompt,
                completion=r.completion,
                latency_ms=r.latency_ms,
                prompt_tokens=0,
                completion_tokens=r.total_tokens,
                total_tokens=r.total_tokens,
            ) for r in model_results]
        )

        avg_scores = {}
        for metric in config.metrics:
            values = [r.scores.get(metric) for r in model_results if isinstance(r.scores.get(metric), (int, float))]
            if values:
                avg_scores[f"avg_{metric}"] = round(sum(values) / len(values), 2)

        run.model_summaries[model_id] = {
            **avg_scores,
            "total_prompts": len(model_results),
            "errors": sum(1 for r in model_results if r.error),
            **asdict(latency_stats),
        }

    return run


def run_eval_sync(config: EvalConfig) -> EvalRun:
    """Synchronous wrapper for run_eval."""
    return asyncio.run(run_eval(config))
