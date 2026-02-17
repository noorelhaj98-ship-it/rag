"""Build chunks from DOCX content."""

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

from common.splitter import TextSplitter
from chunking_docx.config import DOCXChunkingConfig
from chunking_docx.extractor import DOCXExtractor


class ChunkBuilder:
    """Build chunks from DOCX documents."""

    def __init__(self, config: DOCXChunkingConfig = None):
        self.config = config or DOCXChunkingConfig()
        self.extractor = DOCXExtractor()
        self.splitter = TextSplitter(
            target_chars=self.config.target_chars,
            overlap_chars=self.config.overlap_chars
        )

    def _generate_chunk_id(
        self, document_id: str, para_number: int, chunk_index: int, text: str
    ) -> str:
        """Generate stable chunk ID."""
        h = hashlib.sha1(
            f"{document_id}|para{para_number}|c{chunk_index}|{text[:200]}".encode("utf-8")
        ).hexdigest()[:16]
        return f"{document_id}_para{para_number:03d}_c{chunk_index:03d}_{h}"

    def build_chunks(self, docx_path: str) -> List[Dict[str, Any]]:
        """Build chunks from a DOCX file.

        Args:
            docx_path: Path to the DOCX file

        Returns:
            List of chunk dictionaries
        """
        docx_path = Path(docx_path)
        if not docx_path.exists():
            raise FileNotFoundError(f"DOCX file not found: {docx_path}")

        created_at = datetime.now(timezone.utc).isoformat()

        # Extract paragraphs and tables
        paragraphs = self.extractor.extract_paragraphs(str(docx_path))
        tables = self.extractor.extract_tables(str(docx_path))

        # Combine all content
        all_content = paragraphs + tables
        all_content.sort(key=lambda x: x["paragraph_number"])

        chunks = []
        global_char_cursor = 0

        for item in all_content:
            para_number = item["paragraph_number"]
            para_text = item["text"] or ""
            para_text_stripped = para_text.strip()

            if not para_text_stripped:
                global_char_cursor += len(para_text)
                continue

            pieces = self.splitter.split_recursive(para_text)
            pieces = self.splitter.pack_with_overlap(pieces)

            page_search_start = 0
            for idx, chunk_text in enumerate(pieces, start=1):
                normalized_para = re.sub(r"\s+", " ", para_text)
                normalized_chunk = re.sub(r"\s+", " ", chunk_text).strip()

                found_at = normalized_para.find(normalized_chunk, page_search_start)
                if found_at == -1:
                    found_at = normalized_para.find(normalized_chunk)

                if found_at == -1:
                    chunk_start_char = None
                    chunk_end_char = None
                else:
                    chunk_start_char = found_at
                    chunk_end_char = found_at + len(normalized_chunk)

                if found_at != -1:
                    page_search_start = found_at + max(1, len(normalized_chunk) // 3)

                para_num = para_number if isinstance(para_number, int) else 0
                chunk_id = self._generate_chunk_id(
                    self.config.document_id, para_num, idx, chunk_text
                )

                chunks.append({
                    "document_id": self.config.document_id,
                    "chunk_id": chunk_id,
                    "text": chunk_text.strip(),
                    "page_number": para_num,
                    "source_file": docx_path.name,
                    "chunk_start_char": chunk_start_char,
                    "chunk_end_char": chunk_end_char,
                    "chunking_strategy": self.config.chunking_strategy,
                    "created_at": created_at
                })

            global_char_cursor += len(para_text)

        return chunks
