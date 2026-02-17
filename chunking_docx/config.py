"""DOCX chunking configuration."""

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DOCXChunkingConfig:
    """Configuration for DOCX chunking."""

    source_file: str = "document.docx"
    out_jsonl: str = "chunks.jsonl"
    out_csv: str = "chunks.csv"
    target_chars: int = 1200
    overlap_chars: int = 150
    chunking_strategy: str = "recursive"

    @property
    def document_id(self) -> str:
        """Generate stable document ID from filename."""
        return hashlib.sha1(
            Path(self.source_file).name.encode("utf-8")
        ).hexdigest()[:12]


def get_default_config() -> DOCXChunkingConfig:
    """Get default DOCX chunking configuration."""
    return DOCXChunkingConfig()
