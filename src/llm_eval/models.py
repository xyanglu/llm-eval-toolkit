"""Async multi-model LLM client via OpenRouter-compatible APIs."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class ModelResponse:
    model_id: str
    prompt: str
    completion: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    error: str | None = None


class ModelClient:
    """Async client for OpenRouter-compatible chat completions."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        default_headers: dict[str, str] | None = None,
    ):
        self.base_url = (base_url or os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/xyanglu/llm-eval-toolkit",
            "X-Title": "LLM Eval Toolkit",
        }
        if default_headers:
            self.headers.update(default_headers)

    async def complete(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        system_prompt: str | None = None,
    ) -> ModelResponse:
        """Send a single chat completion request."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                )
            resp.raise_for_status()
            data = resp.json()
            latency = (time.perf_counter() - start) * 1000

            usage = data.get("usage", {})
            completion = data["choices"][0]["message"]["content"]

            return ModelResponse(
                model_id=model,
                prompt=prompt,
                completion=completion,
                latency_ms=latency,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            logger.error(f"Error calling {model}: {e}")
            return ModelResponse(
                model_id=model,
                prompt=prompt,
                completion="",
                latency_ms=latency,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                error=str(e),
            )

    async def complete_batch(
        self,
        model: str,
        prompts: list[str],
        temperature: float = 0.7,
        max_tokens: int = 512,
        system_prompt: str | None = None,
        concurrency: int = 5,
    ) -> list[ModelResponse]:
        """Send multiple prompts to a model with concurrency control."""
        sem = asyncio.Semaphore(concurrency)

        async def _limited(prompt: str) -> ModelResponse:
            async with sem:
                return await self.complete(model, prompt, temperature, max_tokens, system_prompt)

        tasks = [_limited(p) for p in prompts]
        return await asyncio.gather(*tasks)

    async def complete_multi_sample(
        self,
        model: str,
        prompt: str,
        n: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 512,
        system_prompt: str | None = None,
    ) -> list[ModelResponse]:
        """Generate N samples from the same prompt (for consistency checks)."""
        tasks = [
            self.complete(model, prompt, temperature, max_tokens, system_prompt)
            for _ in range(n)
        ]
        return await asyncio.gather(*tasks)
