"""Evaluation configuration."""

import os
from dataclasses import dataclass


@dataclass
class EvalConfig:
    """Configuration for RAG evaluation."""

    api_base: str = "http://localhost:8000"
    api_key: str = "alice123"
    questions_file: str = "eval_questions.json"
    results_file: str = "eval_results.json"
    top_k: int = 6
    timeout: int = 60

    def __post_init__(self):
        """Load from environment variables."""
        self.api_base = os.environ.get("RAG_API_URL", self.api_base)
        self.api_key = os.environ.get("RAG_API_KEY", self.api_key)


def get_config() -> EvalConfig:
    """Get evaluation configuration."""
    return EvalConfig()
