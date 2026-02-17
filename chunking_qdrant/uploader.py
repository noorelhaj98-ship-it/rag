"""Qdrant upload functionality."""

import uuid
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from chunking_qdrant.config import QdrantConfig
from chunking_qdrant.bm25 import BM25


class QdrantUploader:
    """Upload chunks to Qdrant with hybrid vectors."""

    def __init__(self, config: QdrantConfig = None):
        self.config = config or QdrantConfig()
        self.client = QdrantClient(
            url=self.config.qdrant_url,
            api_key=self.config.qdrant_api_key
        )
        self.bm25 = BM25()

    def ensure_hybrid_collection(self, dense_dim: int) -> None:
        """Create hybrid collection if it doesn't exist.

        Args:
            dense_dim: Dimension of dense vectors
        """
        existing = [c.name for c in self.client.get_collections().collections]
        if self.config.hybrid_collection in existing:
            return

        self.client.create_collection(
            collection_name=self.config.hybrid_collection,
            vectors_config={
                "dense": qmodels.VectorParams(
                    size=dense_dim,
                    distance=qmodels.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": qmodels.SparseVectorParams(
                    modifier=qmodels.Modifier.IDF
                )
            },
        )
        print(
            f"[qdrant] created hybrid collection '{self.config.hybrid_collection}' "
            f"dense_dim={dense_dim}"
        )

    def chunk_to_point(
        self, chunk: Dict[str, Any], vector: List[float]
    ) -> qmodels.PointStruct:
        """Convert chunk to Qdrant point with hybrid vectors.

        Args:
            chunk: Chunk dictionary
            vector: Dense embedding vector

        Returns:
            Qdrant PointStruct
        """
        stable_id = uuid.uuid5(uuid.NAMESPACE_URL, chunk["chunk_id"])
        text = chunk.get("text", "")

        # Generate sparse BM25 vector
        sparse_indices, sparse_values = self.bm25.compute_sparse_vector(text)

        payload = {
            "document_id": chunk.get("document_id"),
            "chunk_id": chunk.get("chunk_id"),
            "page_number": chunk.get("page_number"),
            "source_file": chunk.get("source_file"),
            "chunk_start_char": chunk.get("chunk_start_char"),
            "chunk_end_char": chunk.get("chunk_end_char"),
            "chunking_strategy": chunk.get("chunking_strategy"),
            "created_at": chunk.get("created_at"),
            "text": text,
        }

        return qmodels.PointStruct(
            id=str(stable_id),
            vector={
                "dense": vector,
                "sparse": qmodels.SparseVector(
                    indices=sparse_indices, values=sparse_values
                ),
            },
            payload=payload,
        )

    def upload_batch(self, points: List[qmodels.PointStruct]) -> None:
        """Upload a batch of points to Qdrant.

        Args:
            points: List of PointStruct to upload
        """
        for i in range(0, len(points), self.config.upsert_batch_size):
            batch = points[i:i + self.config.upsert_batch_size]
            self.client.upsert(
                collection_name=self.config.hybrid_collection,
                points=batch
            )

    def get_collection_name(self) -> str:
        """Get the target collection name."""
        return self.config.hybrid_collection
