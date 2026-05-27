"""Latency and token metrics."""

from __future__ import annotations

from dataclasses import dataclass

from ..models import ModelResponse


@dataclass
class LatencyResult:
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    avg_tokens_per_response: float
    tokens_per_second: float


def compute_latency_stats(responses: list[ModelResponse]) -> LatencyResult:
    """Compute latency and token statistics from a list of responses."""
    valid = [r for r in responses if not r.error]

    if not valid:
        return LatencyResult(
            avg_latency_ms=0,
            min_latency_ms=0,
            max_latency_ms=0,
            total_prompt_tokens=0,
            total_completion_tokens=0,
            total_tokens=0,
            avg_tokens_per_response=0,
            tokens_per_second=0,
        )

    latencies = [r.latency_ms for r in valid]
    total_prompt = sum(r.prompt_tokens for r in valid)
    total_completion = sum(r.completion_tokens for r in valid)
    total = total_prompt + total_completion
    total_time_s = sum(r.latency_ms for r in valid) / 1000

    return LatencyResult(
        avg_latency_ms=round(sum(latencies) / len(latencies), 1),
        min_latency_ms=round(min(latencies), 1),
        max_latency_ms=round(max(latencies), 1),
        total_prompt_tokens=total_prompt,
        total_completion_tokens=total_completion,
        total_tokens=total,
        avg_tokens_per_response=round(total / len(valid), 1),
        tokens_per_second=round(total / total_time_s, 1) if total_time_s > 0 else 0,
    )
