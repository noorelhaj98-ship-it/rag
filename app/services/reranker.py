"""Cross-encoder reranker service."""

import re
from typing import Any, Dict, List


class SimpleCrossEncoderReranker:
    """Cross-encoder style reranker with transformer fallback."""

    def __init__(self):
        self.use_transformer = False
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Attempt to load transformer model, fallback to keyword-based."""
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            self.use_transformer = True
            print("[reranker] Loaded cross-encoder model: ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            print(f"[reranker] Using fallback reranker (no transformer): {e}")
            self.model = None

    def _tokenize(self, text: str) -> set:
        """Simple tokenization for fallback scoring."""
        return set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))

    def _fallback_score(self, query: str, text: str) -> float:
        """Fallback scoring based on keyword overlap."""
        query_tokens = self._tokenize(query)
        text_tokens = self._tokenize(text)

        if not query_tokens:
            return 0.0

        # Jaccard similarity
        intersection = query_tokens & text_tokens
        union = query_tokens | text_tokens
        jaccard = len(intersection) / len(union) if union else 0.0

        # Term frequency bonus
        text_lower = text.lower()
        tf_score = sum(text_lower.count(term) for term in query_tokens) / max(
            len(text.split()), 1
        )

        # Exact match bonus
        exact_match = 1.0 if query.lower() in text.lower() else 0.0

        return 0.4 * jaccard + 0.4 * tf_score + 0.2 * exact_match

    def rerank(
        self, query: str, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rerank documents based on query relevance.

        Args:
            query: Search query
            documents: List of document dictionaries with 'text' field

        Returns:
            Reranked documents sorted by relevance
        """
        if not documents:
            return documents

        if self.use_transformer:
            # Use real cross-encoder
            pairs = [[query, doc["text"]] for doc in documents]
            scores = self.model.predict(pairs)

            for i, doc in enumerate(documents):
                doc["rerank_score"] = float(scores[i])

            return sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        else:
            # Use fallback scoring
            scores = [self._fallback_score(query, doc["text"]) for doc in documents]

            for i, doc in enumerate(documents):
                original_score = doc.get("score", 0.0)
                rerank_score = float(scores[i])
                doc["score"] = 0.3 * original_score + 0.7 * rerank_score
                doc["rerank_score"] = rerank_score

            return sorted(documents, key=lambda x: x["score"], reverse=True)


# Global reranker instance
reranker = SimpleCrossEncoderReranker()
