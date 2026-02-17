"""BM25 sparse vector generation."""

import re
from typing import List, Tuple


class BM25:
    """BM25 implementation for sparse vectors."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, vocab_size: int = 10000):
        self.k1 = k1
        self.b = b
        self.vocab_size = vocab_size

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return re.findall(r'\b[a-zA-Z]+\b', text.lower())

    def _hash_token(self, token: str) -> int:
        """Hash token to fixed vocabulary space."""
        return hash(token) % self.vocab_size

    def compute_sparse_vector(self, text: str) -> Tuple[List[int], List[float]]:
        """Compute sparse BM25 vector.

        Args:
            text: Input text

        Returns:
            Tuple of (indices, values)
        """
        tokens = self._tokenize(text)
        if not tokens:
            return [], []

        term_counts = {}
        for token in tokens:
            idx = self._hash_token(token)
            term_counts[idx] = term_counts.get(idx, 0) + 1

        doc_len = len(tokens)
        indices = []
        values = []

        for idx, freq in term_counts.items():
            tf = freq
            idf = 1.0
            score = idf * (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_len / 10)
            )
            indices.append(idx)
            values.append(score)

        return indices, values
