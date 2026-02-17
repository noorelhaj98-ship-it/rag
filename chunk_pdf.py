"""PDF Chunking - Entry point.

This module serves as the entry point for PDF chunking.
It imports and re-exports from the modular chunking_pdf package.
"""

from common.exporters import JSONLExporter, CSVExporter
from chunking_pdf.cli import main
from chunking_pdf.config import PDFChunkingConfig, get_default_config
from chunking_pdf.chunk_builder import ChunkBuilder

# Backward compatibility - keep old function names
build_chunks = ChunkBuilder.build_chunks
write_jsonl = JSONLExporter.export
write_csv = CSVExporter.export

# Backward compatibility - config constants
SOURCE_FILE = get_default_config().source_file
OUT_JSONL = get_default_config().out_jsonl
OUT_CSV = get_default_config().out_csv
DOCUMENT_ID = get_default_config().document_id
CHUNKING_STRATEGY = get_default_config().chunking_strategy
TARGET_CHARS = get_default_config().target_chars
OVERLAP_CHARS = get_default_config().overlap_chars

__all__ = [
    "main",
    "PDFChunkingConfig",
    "get_default_config",
    "ChunkBuilder",
    "JSONLExporter",
    "CSVExporter",
    "build_chunks",
    "write_jsonl",
    "write_csv",
]

if __name__ == "__main__":
    main()
