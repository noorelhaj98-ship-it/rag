"""Hybrid retrieval service with dense and sparse vectors."""

import re
from typing import Any, Dict, List, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http.models import Fusion, FusionQuery, Prefetch, SparseVector

from app.config import settings
from app.services.reranker import reranker


class SimpleBM25:
    """Simple BM25 implementation for sparse vectors."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, vocab_size: int = 10000):
        self.k1 = k1
        self.b = b
        self.vocab_size = vocab_size

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r"\b[a-zA-Z]+\b", text.lower())

    def _hash_token(self, token: str) -> int:
        """Hash token to fixed vocabulary space."""
        return hash(token) % self.vocab_size

    def compute_sparse_vector(self, text: str) -> Tuple[List[int], List[float]]:
        """Compute sparse vector as (indices, values) for BM25."""
        tokens = self._tokenize(text)
        if not tokens:
            return [], []

        # Count term frequencies
        term_counts = {}
        for token in tokens:
            idx = self._hash_token(token)
            term_counts[idx] = term_counts.get(idx, 0) + 1

        # Compute BM25-like scores
        doc_len = len(tokens)
        indices = []
        values = []

        for idx, freq in term_counts.items():
            tf = freq
            idf = 1.0  # Simplified IDF
            score = idf * (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_len / 10)
            )
            indices.append(idx)
            values.append(score)

        return indices, values


class RetrievalService:
    """Hybrid retrieval service combining dense and sparse vectors."""

    def __init__(self):
        self.client = QdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key
        )
        self.collection = settings.qdrant_collection
        self.bm25 = SimpleBM25()

    def embed_sparse(self, text: str) -> SparseVector:
        """Generate sparse BM25 vector for text."""
        indices, values = self.bm25.compute_sparse_vector(text)
        return SparseVector(indices=indices, values=values)

    def retrieve_hybrid(
        self, query_text: str, query_vector: List[float]
    ) -> List[Dict[str, Any]]:
        """Hybrid retrieval using dense + BM25 sparse vectors with RRF fusion.

        Args:
            query_text: Original query text for BM25
            query_vector: Dense embedding vector

        Returns:
            List of retrieved documents with metadata
        """
        sparse_vector = self.embed_sparse(query_text)

        print(
            f"[hybrid] Config: dense={settings.top_k_dense}, "
            f"keyword={settings.top_k_keyword}, final={settings.top_k_final}, "
            f"fusion={settings.fusion_type}"
        )

        # Determine fusion type
        fusion = Fusion.RRF if settings.fusion_type == "RRF" else Fusion.SCORE_FUSION

        # Prefetch from both vector types
        prefetch = [
            Prefetch(query=query_vector, using="dense", limit=settings.top_k_dense),
            Prefetch(
                query=sparse_vector, using="sparse", limit=settings.top_k_keyword
            ),
        ]

        # Log individual branch results
        dense_response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            using="dense",
            limit=settings.top_k_dense,
            with_payload=True,
        )
        print(f"[dense] Retrieved {len(dense_response.points)} dense vector hits")

        keyword_response = self.client.query_points(
            collection_name=self.collection,
            query=sparse_vector,
            using="sparse",
            limit=settings.top_k_keyword,
            with_payload=True,
        )
        print(f"[keyword] Retrieved {len(keyword_response.points)} keyword/BM25 hits")

        # Fuse results
        response = self.client.query_points(
            collection_name=self.collection,
            prefetch=prefetch,
            query=FusionQuery(fusion=fusion),
            limit=settings.reranker_top_k
            if settings.reranker_enabled
            else settings.top_k_final,
            with_payload=True,
        )
        hits = response.points

        print(f"[hybrid] Fused into {len(hits)} results using {settings.fusion_type}")

        # Convert to output format
        results = []
        for idx, h in enumerate(hits):
            payload = h.payload or {}
            results.append(
                {
                    "rank": idx + 1,
                    "score": float(h.score),
                    "text": (payload.get("text") or "").strip()[:100] + "...",
                    "chunk_id": payload.get("chunk_id"),
                    "page_number": payload.get("page_number"),
                    "source_file": payload.get("source_file"),
                }
            )

        # Debug logging
        print(f"\n[DEBUG] FUSED CANDIDATES (top {min(20, len(results))}):")
        for item in results[:20]:
            print(
                f"  Rank {item['rank']:2d}: Page {item['page_number']:3d} | "
                f"Score {item['score']:.4f} | {item['chunk_id'][:30]}..."
            )

        # Apply reranking if enabled
        if settings.reranker_enabled and results:
            print(f"\n[reranker] Applying reranking to {len(results)} candidates...")
            results = reranker.rerank(query_text, results)

            print(f"\n[DEBUG] RERANKED RESULTS (top {min(10, len(results))}):")
            for idx, item in enumerate(results[:10], 1):
                rerank_score = item.get("rerank_score", "N/A")
                if isinstance(rerank_score, float):
                    print(
                        f"  New Rank {idx:2d}: Page {item['page_number']:3d} | "
                        f"Combined Score {item['score']:.4f} | "
                        f"Rerank Score {rerank_score:.4f} | {item['chunk_id'][:30]}..."
                    )
                else:
                    print(
                        f"  New Rank {idx:2d}: Page {item['page_number']:3d} | "
                        f"Combined Score {item['score']:.4f} | {item['chunk_id'][:30]}..."
                    )

            results = results[: settings.top_k_final]
            print(f"\n[reranker] Reranked to top-{len(results)} results")

        # Clean up temporary fields
        for item in results:
            item.pop("rank", None)
            item.pop("rerank_score", None)

        return results

    def retrieve_dense(self, query_vector: List[float]) -> List[Dict[str, Any]]:
        """Fallback dense-only retrieval.

        Args:
            query_vector: Dense embedding vector

        Returns:
            List of retrieved documents
        """
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            using="dense",
            limit=settings.top_k,
            with_payload=True,
        )

        results = []
        for h in response.points:
            payload = h.payload or {}
            results.append(
                {
                    "score": float(h.score),
                    "text": (payload.get("text") or "").strip(),
                    "chunk_id": payload.get("chunk_id"),
                    "page_number": payload.get("page_number"),
                    "source_file": payload.get("source_file"),
                }
            )
        return results


# Global retrieval service instance
retrieval_service = RetrievalService()
