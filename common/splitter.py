"""Text splitting utilities shared across chunking packages."""

from typing import List


class TextSplitter:
    """Split text into chunks using recursive strategy."""

    SEPARATORS = [
        "\n\n",   # paragraphs
        "\n",     # lines
        ". ",     # sentences
        " "       # words
    ]

    def __init__(self, target_chars: int = 1200, overlap_chars: int = 150):
        self.target_chars = target_chars
        self.overlap_chars = overlap_chars

    def split_recursive(self, text: str) -> List[str]:
        """Recursively split text by increasingly smaller separators.

        Returns pieces each <= target_chars (best effort).
        """
        text = text.strip()
        if not text:
            return []

        if len(text) <= self.target_chars:
            return [text]

        for sep in self.SEPARATORS:
            if sep in text:
                parts = text.split(sep)
                rebuilt = self._rebuild_parts(parts, sep)

                if rebuilt and sum(len(x) for x in rebuilt) >= int(0.6 * len(text)):
                    return rebuilt

        # Fallback: hard split
        return [
            text[i:i + self.target_chars].strip()
            for i in range(0, len(text), self.target_chars)
        ]

    def _rebuild_parts(self, parts: List[str], sep: str) -> List[str]:
        """Rebuild parts respecting target size."""
        rebuilt = []
        buf = ""

        for p in parts:
            if not p:
                continue

            sep_str = sep if buf else ""
            if sep == " ":
                sep_str = " " if buf else ""

            candidate = (buf + sep_str + p).strip()

            if len(candidate) <= self.target_chars:
                buf = candidate
            else:
                if buf:
                    rebuilt.append(buf)

                if len(p) > self.target_chars:
                    rebuilt.extend(self.split_recursive(p))
                    buf = ""
                else:
                    buf = p.strip()

        if buf:
            rebuilt.append(buf)

        return rebuilt

    def pack_with_overlap(self, pieces: List[str]) -> List[str]:
        """Pack pieces with overlap between consecutive chunks."""
        if not pieces:
            return []

        packed = []
        prev = ""

        for p in pieces:
            p = p.strip()
            if not p:
                continue

            if prev and self.overlap_chars > 0:
                overlap = prev[-self.overlap_chars:]
                if not p.startswith(overlap):
                    p = (overlap + "\n" + p).strip()

            packed.append(p)
            prev = p

        return packed
