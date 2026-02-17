"""Evaluation question loader."""

import json
from typing import Dict, List, Any


class QuestionLoader:
    """Load evaluation questions from JSON file."""

    @staticmethod
    def load(path: str = "eval_questions.json") -> List[Dict[str, Any]]:
        """Load evaluation questions.

        Args:
            path: Path to the questions JSON file

        Returns:
            List of question dictionaries
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["evaluation_set"]
