"""Text cleaning utilities for PDF extraction."""


class TextCleaner:
    """Clean and fix text encoding issues from PDF extraction."""

    # Common mojibake replacements
    MOJIBAKE_REPLACEMENTS = {
        "â€™": "'",
        "â€˜": "'",
        "â€œ": '"',
        "â€�": '"',
        "â€”": "—",
        "â€“": "–",
        "â€¦": "…",
        "Â ": " ",
        "Â": "",
    }

    @classmethod
    def fix_mojibake(cls, text: str) -> str:
        """Fix common PDF/Windows mojibake sequences."""
        if not text:
            return text

        # First try the roundtrip fix
        try:
            text = text.encode("latin-1").decode("utf-8")
        except Exception:
            pass

        # Apply explicit replacements
        for bad, good in cls.MOJIBAKE_REPLACEMENTS.items():
            text = text.replace(bad, good)

        return text

    @classmethod
    def normalize_line_endings(cls, text: str) -> str:
        """Normalize line endings to Unix style."""
        return text.replace("\r\n", "\n").replace("\r", "\n")
