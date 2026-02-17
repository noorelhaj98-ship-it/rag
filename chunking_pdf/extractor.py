"""PDF text extraction."""

from typing import List, Dict, Any


class PDFExtractor:
    """Extract text from PDF files using PyMuPDF."""

    def __init__(self):
        self._fitz = None

    def _get_fitz(self):
        """Lazy import PyMuPDF."""
        if self._fitz is None:
            try:
                import fitz
                self._fitz = fitz
            except ImportError as e:
                raise RuntimeError(
                    "Missing dependency: PyMuPDF. Install with: pip install pymupdf"
                ) from e
        return self._fitz

    def extract_pages(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from each page of a PDF.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of dicts with page_number (1-based) and text
        """
        fitz = self._get_fitz()
        doc = fitz.open(pdf_path)
        pages = []

        for i in range(len(doc)):
            page = doc[i]
            text = page.get_text("text") or ""
            # Normalize line endings
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            pages.append({"page_number": i + 1, "text": text})

        return pages
