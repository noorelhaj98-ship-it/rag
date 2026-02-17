"""Chunk export utilities shared across chunking packages."""

import csv
import json
from typing import List, Dict, Any


class JSONLExporter:
    """Export chunks to JSONL format."""

    @staticmethod
    def export(chunks: List[Dict[str, Any]], path: str) -> None:
        """Write chunks to JSONL file."""
        with open(path, "w", encoding="utf-8") as f:
            for obj in chunks:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")


class CSVExporter:
    """Export chunks to CSV format."""

    FIELDNAMES = [
        "document_id", "chunk_id", "page_number", "source_file",
        "chunk_start_char", "chunk_end_char",
        "chunking_strategy", "created_at", "text"
    ]

    @classmethod
    def export(cls, chunks: List[Dict[str, Any]], path: str) -> None:
        """Write chunks to CSV file."""
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cls.FIELDNAMES)
            writer.writeheader()
            for obj in chunks:
                writer.writerow({k: obj.get(k) for k in cls.FIELDNAMES})
