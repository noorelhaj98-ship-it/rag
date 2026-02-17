"""Build chunks from PDF pages."""

import hashlib
import os
import re
from datetime import datetime, timezone
from typing import List, Dict, Any

from common.splitter import TextSplitter
from chunking_pdf.config import PDFChunkingConfig
from chunking_pdf.extractor import PDFExtractor
from chunking_pdf.text_cleaner import TextCleaner


class ChunkBuilder:
    """Build chunks from PDF documents."""

    def __init__(self, config: PDFChunkingConfig = None):
        self.config = config or PDFChunkingConfig()
        self.extractor = PDFExtractor()
        self.splitter = TextSplitter(
            target_chars=self.config.target_chars,
            overlap_chars=self.config.overlap_chars
        )

    def _generate_chunk_id(
        self, document_id: str, page_number: int, chunk_index: int, text: str
    ) -> str:
        """Generate stable chunk ID."""
        h = hashlib.sha1(
            f"{document_id}|p{page_number}|c{chunk_index}|{text[:200]}".encode("utf-8")
        ).hexdigest()[:16]
        return f"{document_id}_p{page_number:03d}_c{chunk_index:03d}_{h}"

    def build_chunks(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Build chunks from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of chunk dictionaries
        """
        created_at = datetime.now(timezone.utc).isoformat()
        pages = self.extractor.extract_pages(pdf_path)
        chunks = []
        global_char_cursor = 0

        for page in pages:
            page_number = page["page_number"]
            page_text = page["text"] or ""
            page_text_stripped = page_text.strip()

            if not page_text_stripped:
                global_char_cursor += len(page_text)
                continue

            # Split and pack with overlap
            pieces = self.splitter.split_recursive(page_text)
            pieces = self.splitter.pack_with_overlap(pieces)

            # Build chunks for this page
            page_search_start = 0
            for idx, chunk_text in enumerate(pieces, start=1):
                # Find character offsets
                chunk_start_char, chunk_end_char = self._find_offsets(
                    page_text, chunk_text, page_search_start
                )

                if chunk_start_char != -1:
                    page_search_start = chunk_start_char + max(1, len(chunk_text) // 3)

                # Generate ID and clean text
                chunk_id = self._generate_chunk_id(
                    self.config.document_id, page_number, idx, chunk_text
                )
                chunk_text = TextCleaner.fix_mojibake(chunk_text)

                chunks.append({
                    "document_id": self.config.document_id,
                    "chunk_id": chunk_id,
                    "text": chunk_text.strip(),
                    "page_number": page_number,
                    "source_file": os.path.basename(pdf_path),
                    "chunk_start_char": chunk_start_char,
                    "chunk_end_char": chunk_end_char,
                    "chunking_strategy": self.config.chunking_strategy,
                    "created_at": created_at
                })

            global_char_cursor += len(page_text)

        return chunks

    def _find_offsets(
        self, page_text: str, chunk_text: str, search_start: int
    ) -> tuple:
        """Find character offsets of chunk within page text."""
        normalized_page = re.sub(r"\s+", " ", page_text)
        normalized_chunk = re.sub(r"\s+", " ", chunk_text).strip()

        found_at = normalized_page.find(normalized_chunk, search_start)

        if found_at == -1:
            # Fallback: reset search
            found_at = normalized_page.find(normalized_chunk)

        if found_at == -1:
            return None, None

        return found_at, found_at + len(normalized_chunk)
