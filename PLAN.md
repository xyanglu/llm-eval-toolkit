# LLM Eval Toolkit — Project Plan

## What

An open-source Python toolkit for evaluating LLM quality across multiple providers.
Run the same prompts against different models, score responses with both LLM-as-judge
and deterministic metrics, and get a comparison report.

## Why

- "LLM evaluation" / "benchmarking" / "quality testing" appears in real job postings
- You understand LLM APIs already (Hermes runs on them daily)
- It's genuinely useful — you can eval model options for your own projects
- Shows you can build production-quality tooling, not just notebooks

## Scope

### MVP (3-4 days)

1. Config-driven eval runs (YAML — define models, prompts, metrics)
2. Async parallel model calls via OpenRouter API
3. LLM-as-judge scoring (relevance, accuracy, instruction following)
4. Hallucination detection (SelfCheckGPT-lite: sample N responses, check consistency)
5. Latency + token tracking
6. JSON output + CLI summary table
7. Small golden dataset (~20 prompts across categories)

### Phase 2 (week 2)

8. Safety/refusal scoring (StrongREJECT-lite)
9. Bias detection (simple demographic term analysis)
10. HTML report with charts (matplotlib/plotly)
11. Regression mode (save baselines, compare runs, flag degradations)
12. More model providers (Anthropic direct, OpenAI direct, local Ollama)

### Phase 3 (week 3+)

13. CI integration (run evals as GitHub Actions on every change)
14. Custom eval categories (user-defined scoring rubrics)
15. Dataset import (load from HuggingFace Hub)
16. Public demo / hosted dashboard

## Tech Stack

- Python 3.10+
- httpx + asyncio (parallel API calls)
- OpenRouter API (multi-model access from one key)
- PyYAML (config files)
- rich (CLI output)
- click or typer (CLI interface)
- JSON output by default, HTML report in phase 2

## Repo Structure

```
llm-eval-toolkit/
├── README.md
├── PLAN.md                    # this file
├── pyproject.toml
├── .gitignore
├── config/
│   └── example.yaml           # sample eval config
├── src/
│   └── llm_eval/
│       ├── __init__.py
│       ├── cli.py             # typer CLI entry point
│       ├── config.py          # load/validate YAML configs
│       ├── models.py          # async model client (OpenRouter + others)
│       ├── evaluator.py       # main eval orchestrator
│       ├── metrics/
│       │   ├── __init__.py
│       │   ├── judge.py       # LLM-as-judge scoring
│       │   ├── hallucination.py  # consistency-based hallucination detection
│       │   ├── safety.py      # refusal/strongreject-lite scoring
│       │   ├── bias.py        # demographic term analysis
│       │   └── latency.py     # response time + token counting
│       ├── reporters/
│       │   ├── __init__.py
│       │   ├── json_reporter.py
│       │   ├── cli_reporter.py
│       │   └── html_reporter.py
│       └── datasets/
│           ├── __init__.py
│           ├── golden.py      # built-in golden dataset
│           └── loader.py      # load from HF Hub / local files
├── data/
│   └── golden_prompts.json    # built-in eval prompts
├── outputs/                   # default output dir (gitignored)
└── tests/
    ├── test_config.py
    ├── test_models.py
    ├── test_metrics.py
    └── fixtures/
        └── sample_config.yaml
```

## Eval Config Example

```yaml
# config/example.yaml
name: "quick-eval"
models:
  - id: "anthropic/claude-sonnet-4"
    provider: openrouter
  - id: "google/gemini-2.0-flash-001"
    provider: openrouter
  - id: "meta-llama/llama-3.3-70b-instruct"
    provider: openrouter

metrics:
  - relevance       # LLM judge: does it answer the question?
  - hallucination   # consistency check across N samples
  - instruction_following  # did it follow all constraints?
  - latency         # response time tracking
  - safety          # refusal on harmful prompts

prompts:
  source: builtin   # or: file://data/custom.json, hf://dataset/name
  categories:
    - factual_qa
    - reasoning
    - creative_writing
    - safety

settings:
  samples_per_prompt: 3   # for hallucination consistency check
  temperature: 0.7
  max_tokens: 512
  judge_model: "anthropic/claude-sonnet-4"  # model used for LLM-as-judge
```

## CLI Design

```bash
# Run eval
llm-eval run config/example.yaml

# Run with specific output
llm-eval run config/example.yaml --output results/run1.json

# Compare two runs
llm-eval compare results/run1.json results/run2.json

# List available metrics
llm-eval metrics

# Generate HTML report from JSON results
llm-eval report results/run1.json --format html
```

## Golden Dataset Categories

| Category | # Prompts | What it tests |
|----------|-----------|---------------|
| factual_qa | 5 | Accuracy, no hallucination on knowable facts |
| reasoning | 5 | Multi-step logic, math, code |
| creative | 5 | Fluency, following style/constraints |
| safety | 5 | Refusal on harmful requests, no false positives |
| instruction_following | 5 | Follows format, length, tone constraints |

## Success Criteria

- [ ] Can run `llm-eval run config.yaml` and get scored results
- [ ] Works with 3+ models via OpenRouter
- [ ] Hallucination metric catches at least 1 obvious hallucination
- [ ] README with install + quickstart gets someone running in 5 min
- [ ] At least 1 blog post / Reddit post about the project
