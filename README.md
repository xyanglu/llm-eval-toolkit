# LLM Eval Toolkit

Open-source toolkit for evaluating LLM quality across multiple providers. Run the same prompts against different models, score responses with LLM-as-judge and deterministic metrics, and compare results.

## Features

- **Multi-model comparison**: eval any model via OpenRouter in a single run
- **LLM-as-judge scoring**: relevance, instruction following, safety
- **Hallucination detection**: SelfCheckGPT-lite consistency analysis
- **Safety evaluation**: refusal detection with false positive/negative tracking
- **Latency & token metrics**: response time, throughput, token usage
- **Async & parallel**: concurrent requests for fast evals
- **JSON reports**: machine-readable output for CI integration
- **Rich CLI**: tables, progress, and comparison views

## Install

```bash
git clone https://github.com/xyanglu/llm-eval-toolkit.git
cd llm-eval-toolkit
pip install -e ".[dev]"
```

Requires `OPENROUTER_API_KEY` in your environment.

## Quick Start

```bash
# Generate example config
llm-eval config-example > config/my-eval.yaml

# Run evaluation
llm-eval run config/my-eval.yaml

# Save to specific file
llm-eval run config/my-eval.yaml -o results/baseline.json

# Compare two runs
llm-eval compare results/baseline.json results/after-tuning.json

# Display a report
llm-eval report results/baseline.json
```

## Config Format

```yaml
name: "my-eval"
models:
  - id: "anthropic/claude-sonnet-4"
    provider: openrouter
  - id: "google/gemini-2.0-flash-001"
    provider: openrouter

metrics:
  - relevance             # LLM judge: did it answer correctly?
  - hallucination         # Consistency check across N samples
  - instruction_following # Followed all format/constraint instructions?
  - safety                # Refused harmful, answered safe?
  - latency               # Response time tracking

prompts:
  source: builtin         # Uses 25 built-in prompts across 5 categories
  categories:
    - factual_qa
    - reasoning
    - creative_writing
    - safety
    - instruction_following

settings:
  samples_per_prompt: 3   # For hallucination consistency
  temperature: 0.7
  max_tokens: 512
  judge_model: "anthropic/claude-sonnet-4"
```

## Metrics

| Metric | Method | What it measures |
|--------|--------|-----------------|
| relevance | LLM-as-judge | Factual accuracy and completeness (1-5) |
| hallucination | Multi-sample consistency | Token overlap across N generations (0-1) |
| instruction_following | LLM-as-judge | Followed format, length, tone constraints (1-5) |
| safety | Pattern matching | Correct refusal on harmful / answer on safe |
| latency | Timing | Response time, tokens/sec, token counts |

## Built-in Dataset

25 prompts across 5 categories:
- **Factual QA** (5): knowable facts, checks for hallucination
- **Reasoning** (5): multi-step logic, math, code
- **Creative Writing** (5): fluency, style adherence
- **Safety** (5): harmful requests + one educational edge case
- **Instruction Following** (5): format, length, structure constraints

## How It Works

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│ Config   │────▶│ Evaluator│────▶│ Reporter │
│ (YAML)   │     │ (async)  │     │ (JSON)   │
└──────────┘     └────┬─────┘     └──────────┘
                      │
           ┌──────────┼──────────┐
           ▼          ▼          ▼
      ┌─────────┐ ┌────────┐ ┌────────┐
      │ Model A │ │Model B │ │Model C │
      └────┬────┘ └───┬────┘ └───┬────┘
           │          │          │
           ▼          ▼          ▼
      ┌─────────────────────────────┐
      │  Metrics (judge, safety,   │
      │  hallucination, latency)   │
      └─────────────────────────────┘
```

## Project Structure

```
src/llm_eval/
├── cli.py           # typer CLI
├── config.py        # YAML config loading
├── models.py        # Async OpenRouter client
├── evaluator.py     # Main orchestrator
├── metrics/
│   ├── judge.py     # LLM-as-judge scoring
│   ├── hallucination.py  # Consistency detection
│   ├── safety.py    # Refusal detection
│   └── latency.py   # Token/timing stats
├── reporters/
│   └── json_reporter.py  # JSON save + rich tables
└── datasets/
    └── golden.py    # 25 built-in eval prompts
```

## License

MIT
