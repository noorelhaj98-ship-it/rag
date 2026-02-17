"""Embedding service client."""

from typing import List

import requests

from app.config import settings


class EmbeddingService:
    """Client for the embedding service."""

    def __init__(self):
        self.service_type = settings.embedding_service_type
        self.api_url = settings.embedding_api_url
        self.timeout = settings.embed_timeout

    def embed(self, text: str) -> List[float]:
        """Generate embedding vector for text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            RuntimeError: If service is not configured or request fails
        """
        if self.service_type != "aragemma":
            raise RuntimeError(f"Unsupported embedding service: {self.service_type}")

        if not self.api_url:
            raise RuntimeError("Missing EMBEDDING_API_URL")

        payload = {"text": text}
        response = requests.post(
            self.api_url, json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()

        # Handle different response formats
        if "embedding" in data:
            return data["embedding"]
        if "embeddings" in data and len(data["embeddings"]) > 0:
            return data["embeddings"][0]
        if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
            return data["data"][0].get("embedding", data["data"][0])

        raise RuntimeError(f"Unexpected embedding response: {data}")


# Global embedding service instance
embedding_service = EmbeddingService()
