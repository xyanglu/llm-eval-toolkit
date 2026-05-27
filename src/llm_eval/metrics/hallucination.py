"""Consistency-based hallucination detection (SelfCheckGPT-lite)."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from ..models import ModelResponse


@dataclass
class HallucinationResult:
    is_likely_hallucination: bool
    consistency_score: float  # 0.0 - 1.0
    inconsistent_claims: list[str]
    details: str


def tokenize_simple(text: str) -> list[str]:
    """Simple word tokenization — lowercase, strip punctuation."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def extract_claims(response: ModelResponse) -> list[str]:
    """Extract factual claims from a response as short phrases."""
    # Simple approach: split into sentences, take short ones as claims
    sentences = re.split(r"[.!?\n]+", response.completion)
    claims = []
    for s in sentences:
        s = s.strip()
        if 10 < len(s) < 200:
            claims.append(s)
    return claims


def check_consistency(samples: list[ModelResponse]) -> HallucinationResult:
    """
    Check consistency across N samples of the same prompt.

    SelfCheckGPT approach: generate N responses, extract claims,
    check if each claim appears consistently across samples.
    Low consistency = likely hallucination.
    """
    if len(samples) < 2:
        return HallucinationResult(
            is_likely_hallucination=False,
            consistency_score=1.0,
            inconsistent_claims=[],
            details="Need at least 2 samples for consistency check",
        )

    # Filter out errors
    valid = [s for s in samples if not s.error]
    if len(valid) < 2:
        return HallucinationResult(
            is_likely_hallucination=False,
            consistency_score=0.0,
            inconsistent_claims=[],
            details="Too many errors to check consistency",
        )

    # Extract and tokenize each response
    token_sets = [set(tokenize_simple(s.completion)) for s in valid]

    # Find common tokens across all samples
    common = token_sets[0]
    for ts in token_sets[1:]:
        common &= ts

    # Find all unique tokens
    all_tokens = set()
    for ts in token_sets:
        all_tokens |= ts

    if not all_tokens:
        return HallucinationResult(
            is_likely_hallucination=False,
            consistency_score=1.0,
            inconsistent_claims=[],
            details="Empty responses",
        )

    # Consistency = ratio of tokens that appear in ALL samples
    consistency_score = len(common) / len(all_tokens)

    # Extract claims from first response and check against others
    first_claims = extract_claims(valid[0])
    inconsistent_claims = []

    for claim in first_claims:
        claim_tokens = set(tokenize_simple(claim))
        # Check if claim tokens appear in majority of other samples
        matches = sum(
            1 for ts in token_sets[1:] if len(claim_tokens & ts) / max(len(claim_tokens), 1) > 0.5
        )
        if matches < len(valid) / 2:
            inconsistent_claims.append(claim)

    is_hallucination = consistency_score < 0.4 or len(inconsistent_claims) > len(first_claims) * 0.5

    return HallucinationResult(
        is_likely_hallucination=is_hallucination,
        consistency_score=round(consistency_score, 3),
        inconsistent_claims=inconsistent_claims[:5],
        details=f"{len(valid)} valid samples, {len(common)}/{len(all_tokens)} common tokens ({consistency_score:.0%})",
    )
