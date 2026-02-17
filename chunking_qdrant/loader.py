"""Chunk loading from JSONL."""

import json
from typing import List, Dict, Any


class ChunkLoader:
    """Load chunks from JSONL files."""

    @staticmethod
    def load_chunks(path: str) -> List[Dict[str, Any]]:
        """Load chunks from a JSONL file.

        Args:
            path: Path to the JSONL file

        Returns:
            List of chunk dictionaries with non-empty text
        """
        chunks: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                text = (obj.get("text") or "").strip()
                if text:
                    chunks.append(obj)
        return chunks
