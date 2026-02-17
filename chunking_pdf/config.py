"""PDF chunking configuration."""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PDFChunkingConfig:
    """Configuration for PDF chunking."""

    source_file: str = "The Content of RealSoft (1).pdf"
    out_jsonl: str = "chunks.jsonl"
    out_csv: str = "chunks.csv"
    target_chars: int = 1200
    overlap_chars: int = 150
    chunking_strategy: str = "recursive"

    @property
    def document_id(self) -> str:
        """Generate stable document ID from filename."""
        return hashlib.sha1(
            os.path.basename(self.source_file).encode("utf-8")
        ).hexdigest()[:12]


def get_default_config() -> PDFChunkingConfig:
    """Get default PDF chunking configuration."""
    return PDFChunkingConfig()
