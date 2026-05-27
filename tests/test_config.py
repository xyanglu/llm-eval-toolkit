"""Minimal tests for config loading."""

import pytest
from llm_eval.config import load_config, EvalConfig, ModelConfig
from pathlib import Path


def test_load_example_config(tmp_path):
    config_yaml = """
name: test-eval
models:
  - id: test-model
    provider: openrouter
metrics:
  - relevance
  - latency
settings:
  samples_per_prompt: 3
  temperature: 0.5
  max_tokens: 256
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_yaml)

    config = load_config(config_file)

    assert config.name == "test-eval"
    assert len(config.models) == 1
    assert config.models[0].id == "test-model"
    assert config.models[0].provider == "openrouter"
    assert config.metrics == ["relevance", "latency"]
    assert config.samples_per_prompt == 3
    assert config.temperature == 0.5
    assert config.max_tokens == 256


def test_load_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")


def test_load_empty_file(tmp_path):
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("")
    with pytest.raises(ValueError):
        load_config(config_file)
