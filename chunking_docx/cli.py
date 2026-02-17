"""Command-line interface for DOCX chunking."""

import argparse
import sys

from common.exporters import JSONLExporter, CSVExporter
from chunking_docx.config import DOCXChunkingConfig, get_default_config
from chunking_docx.chunk_builder import ChunkBuilder


def main():
    """Main entry point for DOCX chunking CLI."""
    parser = argparse.ArgumentParser(description="Chunk DOCX files")
    parser.add_argument(
        "source",
        nargs="?",
        help="Source DOCX file (default: from config)"
    )
    parser.add_argument(
        "--jsonl",
        default="chunks.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--csv",
        default="chunks.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--target-chars",
        type=int,
        default=1200,
        help="Target chunk size in characters"
    )
    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=150,
        help="Overlap between chunks"
    )

    args = parser.parse_args()

    # Build config
    config = DOCXChunkingConfig(
        source_file=args.source or get_default_config().source_file,
        out_jsonl=args.jsonl,
        out_csv=args.csv,
        target_chars=args.target_chars,
        overlap_chars=args.overlap_chars
    )

    # Build chunks
    print(f"Processing DOCX: {config.source_file}")
    builder = ChunkBuilder(config)
    chunks = builder.build_chunks(config.source_file)

    # Export
    JSONLExporter.export(chunks, config.out_jsonl)
    CSVExporter.export(chunks, config.out_csv)

    print(f"Wrote {len(chunks)} chunks to {config.out_jsonl} and {config.out_csv}")


if __name__ == "__main__":
    main()
