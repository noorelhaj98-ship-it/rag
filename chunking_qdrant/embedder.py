"""Embedding service client for Qdrant ingestion."""

import time
from typing import List

import requests

from chunking_qdrant.config import QdrantConfig


class EmbeddingClient:
    """Client for the embedding service."""

    def __init__(self, config: QdrantConfig = None):
        self.config = config or QdrantConfig()
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration."""
        if self.config.embedding_service_type != "aragemma":
            raise RuntimeError(
                f"Unsupported EMBEDDING_SERVICE_TYPE='{self.config.embedding_service_type}'. "
                "Set EMBEDDING_SERVICE_TYPE=aragemma."
            )
        if not self.config.embedding_api_url:
            raise RuntimeError(
                "Missing EMBEDDING_API_URL env var (required for aragemma)."
            )

    def embed_single(self, text: str) -> List[float]:
        """Get embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        payload = {"text": text}
        response = requests.post(
            self.config.embedding_api_url,
            json=payload,
            timeout=self.config.embed_timeout
        )

        if response.status_code == 422:
            print(f"[embed/aragemma] 422 response: {response.text[:1000]}")

        response.raise_for_status()
        data = response.json()

        # Handle different response schemas
        if "embedding" in data:
            return data["embedding"]
        if "embeddings" in data and len(data["embeddings"]) > 0:
            return data["embeddings"][0]
        if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
            return data["data"][0].get("embedding", data["data"][0])

        raise RuntimeError(f"Unexpected embedding response schema: {data}")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts with retry logic.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []

        for i, text in enumerate(texts):
            for attempt in range(6):
                try:
                    embedding = self.embed_single(text)
                    embeddings.append(embedding)

                    if (i + 1) % 10 == 0:
                        print(f"[embed/aragemma] processed {i + 1}/{len(texts)}")

                    break

                except Exception as e:
                    wait = min(2 ** attempt, 30)
                    print(
                        f"[embed/aragemma] error on text {i}: {e} | retrying in {wait}s"
                    )
                    time.sleep(wait)

            else:
                raise RuntimeError(
                    f"Failed to embed text after retries: {text[:100]}..."
                )

        return embeddings
