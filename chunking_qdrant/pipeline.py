"""Ingestion pipeline orchestration."""

from typing import List, Any

from chunking_qdrant.config import QdrantConfig, get_config
from chunking_qdrant.loader import ChunkLoader
from chunking_qdrant.embedder import EmbeddingClient
from chunking_qdrant.uploader import QdrantUploader


def batched(lst: List[Any], n: int):
    """Batch a list into chunks of size n."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def estimate_avg_len(chunks: List[dict]) -> float:
    """Estimate average text length in words."""
    lengths = [
        len((c.get("text") or "").split())
        for c in chunks
        if (c.get("text") or "").strip()
    ]
    return (sum(lengths) / max(len(lengths), 1)) or 1.0


class IngestionPipeline:
    """Orchestrate the full ingestion pipeline."""

    def __init__(self, config: QdrantConfig = None):
        self.config = config or get_config()
        self.loader = ChunkLoader()
        self.embedder = EmbeddingClient(self.config)
        self.uploader = QdrantUploader(self.config)

    def run(self) -> None:
        """Run the full ingestion pipeline."""
        print("DEBUG ENV TYPE =", self.config.embedding_service_type)
        print("[cfg] EMBEDDING_SERVICE_TYPE =", self.config.embedding_service_type)
        print("[cfg] EMBEDDING_API_URL =", self.config.embedding_api_url)
        print("[cfg] QDRANT_URL =", self.config.qdrant_url)
        print("[cfg] COLLECTION =", self.config.hybrid_collection)

        # Load chunks
        chunks = self.loader.load_chunks(self.config.input_jsonl)
        print(f"[load] chunks: {len(chunks)}")

        # Embed first batch to learn vector dimension
        first_texts = [
            c["text"] for c in chunks[: min(self.config.batch_size, len(chunks))]
        ]
        first_vectors = self.embedder.embed_batch(first_texts)
        dim = len(first_vectors[0])
        print(f"[embed] dim={dim}")

        # Create hybrid collection
        self.uploader.ensure_hybrid_collection(dim)

        # Process and upload all chunks
        total = 0
        for chunk_batch in batched(chunks, self.config.batch_size):
            texts = [c["text"] for c in chunk_batch]
            vectors = self.embedder.embed_batch(texts)

            points = [
                self.uploader.chunk_to_point(chunk_batch[i], vectors[i])
                for i in range(len(chunk_batch))
            ]

            self.uploader.upload_batch(points)

            total += len(points)
            print(f"[upsert] total={total}")

        print("[done] ingestion complete")


def main() -> None:
    """Main entry point."""
    pipeline = IngestionPipeline()
    pipeline.run()
