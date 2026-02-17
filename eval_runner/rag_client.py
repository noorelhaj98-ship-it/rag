"""RAG API client."""

from typing import Dict, Any

import requests

from eval_runner.config import EvalConfig


class RAGClient:
    """Client for the RAG API."""

    def __init__(self, config: EvalConfig = None):
        self.config = config or EvalConfig()

    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG API.

        Args:
            question: The question to ask

        Returns:
            API response with answer and sources
        """
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        payload = {"question": question}

        try:
            response = requests.post(
                f"{self.config.api_base}/ask",
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"[ERROR] Query failed: {e}")
            return {"answer": "", "sources": []}

    def health_check(self) -> bool:
        """Check if the RAG server is running.

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            requests.get(f"{self.config.api_base}/api", timeout=5)
            return True
        except:
            return False
