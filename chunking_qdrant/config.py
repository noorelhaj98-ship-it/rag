"""Qdrant ingestion configuration."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)


@dataclass
class QdrantConfig:
    """Configuration for Qdrant ingestion."""

    embedding_service_type: str = "aragemma"
    embedding_api_url: str = None
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = None
    collection: str = "realsoft_chunks"
    hybrid_collection: str = "realsoft_chunks_hybrid"
    input_jsonl: str = "chunks.jsonl"
    batch_size: int = 64
    upsert_batch_size: int = 256
    embed_timeout: int = 120

    def __post_init__(self):
        """Load from environment variables."""
        if self.embedding_service_type == "aragemma":
            self.embedding_service_type = os.environ.get(
                "EMBEDDING_SERVICE_TYPE", "aragemma"
            ).lower()
        if self.embedding_api_url is None:
            self.embedding_api_url = os.environ.get("EMBEDDING_API_URL")
        if self.qdrant_url == "http://localhost:6333":
            self.qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        if self.qdrant_api_key is None:
            self.qdrant_api_key = os.environ.get("QDRANT_API_KEY") or None
        if self.collection == "realsoft_chunks":
            self.collection = os.environ.get("QDRANT_COLLECTION", "realsoft_chunks")
        if self.hybrid_collection == "realsoft_chunks_hybrid":
            self.hybrid_collection = os.environ.get(
                "QDRANT_COLLECTION", "realsoft_chunks_hybrid"
            )


def get_config() -> QdrantConfig:
    """Get Qdrant configuration from environment."""
    return QdrantConfig()
