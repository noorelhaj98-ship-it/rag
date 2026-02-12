import os
import re
import json
import csv
import hashlib
from datetime import datetime, timezone

# ---- Config ----
SOURCE_FILE = "The Content of RealSoft (1).pdf"
OUT_JSONL = "chunks.jsonl"
OUT_CSV = "chunks.csv"

DOCUMENT_ID = hashlib.sha1(os.path.basename(SOURCE_FILE).encode("utf-8")).hexdigest()[:12]
CHUNKING_STRATEGY = "recursive"
TARGET_CHARS = 1200          # target chunk size (characters)
OVERLAP_CHARS = 150          # overlap to preserve context

# ---- PDF text extraction ----
def extract_pages_text_pymupdf(pdf_path: str):
    """
    Returns: list of dicts: [{page_number: 1-based, text: "..."}]
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise RuntimeError("Missing dependency: PyMuPDF. Install with: pip install pymupdf") from e

    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        page = doc[i]
        # "text" is usually fine; consider "blocks" if you need layout-aware extraction later.
        text = page.get_text("text") or ""
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        pages.append({"page_number": i + 1, "text": text})
    return pages

SEPARATORS = [
    "\n\n",   # paragraphs
    "\n",     # lines
    ". ",     # sentences (approx)
    " "       # words
]

def fix_mojibake(s: str) -> str:
    """
    Robust cleanup for common PDF/Windows mojibake sequences.
    """
    if not s:
        return s

    # First try the roundtrip fix (works for some cases)
    try:
        s2 = s.encode("latin-1").decode("utf-8")
        s = s2
    except Exception:
        pass

    # Then apply explicit fixes (works for your â€™ / â€” cases)
    replacements = {
        "â€™": "’",
        "â€˜": "‘",
        "â€œ": "“",
        "â€�": "”",
        "â€”": "—",
        "â€“": "–",
        "â€¦": "…",
        "Â ": " ",
        "Â": "",
    }
    for bad, good in replacements.items():
        s = s.replace(bad, good)

    return s



def split_recursive(text: str, max_chars: int):
    """
    Recursively split text by increasingly smaller separators until pieces fit.
    Returns a list of pieces (strings) each <= max_chars (best effort).
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    # Try separators from coarse to fine
    for sep in SEPARATORS:
        if sep in text:
            parts = text.split(sep)
            # Re-add separator when joining later (except last)
            rebuilt = []
            buf = ""
            for idx, p in enumerate(parts):
                if not p:
                    continue
                candidate = (buf + (sep if buf else "") + p).strip() if sep != " " else (buf + (" " if buf else "") + p).strip()
                if len(candidate) <= max_chars:
                    buf = candidate
                else:
                    if buf:
                        rebuilt.append(buf)
                    # if single part too big, recurse
                    if len(p) > max_chars:
                        rebuilt.extend(split_recursive(p, max_chars))
                        buf = ""
                    else:
                        buf = p.strip()
            if buf:
                rebuilt.append(buf)

            # If we actually made progress, return
            if rebuilt and sum(len(x) for x in rebuilt) >= int(0.6 * len(text)):
                return rebuilt

    # Fallback hard split
    return [text[i:i + max_chars].strip() for i in range(0, len(text), max_chars)]

def pack_with_overlap(pieces, overlap_chars: int):
    """
 
    """
    if not pieces:
        return []

    packed = []
    prev = ""
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        if prev and overlap_chars > 0:
            overlap = prev[-overlap_chars:]
            # avoid duplicating if already starts similarly
            if not p.startswith(overlap):
                p = (overlap + "\n" + p).strip()
        packed.append(p)
        prev = p
    return packed

def stable_chunk_id(document_id: str, page_number: int, chunk_index: int, text: str):
    h = hashlib.sha1(f"{document_id}|p{page_number}|c{chunk_index}|{text[:200]}".encode("utf-8")).hexdigest()[:16]
    return f"{document_id}_p{page_number:03d}_c{chunk_index:03d}_{h}"

# ---- Main pipeline ----
def build_chunks(pdf_path: str):
    created_at = datetime.now(timezone.utc).isoformat()

    pages = extract_pages_text_pymupdf(pdf_path)

    chunks = []
    global_char_cursor = 0  # if you want doc-level offsets; we also provide page-local below

    for page in pages:
        page_number = page["page_number"]
        page_text = page["text"] or ""
        page_text_stripped = page_text.strip()

     
        if not page_text_stripped:
            global_char_cursor += len(page_text)
            continue

        pieces = split_recursive(page_text, TARGET_CHARS)
        pieces = pack_with_overlap(pieces, OVERLAP_CHARS)

        page_search_start = 0
        for idx, chunk_text in enumerate(pieces, start=1):
          
            normalized_page = re.sub(r"\s+", " ", page_text)
            normalized_chunk = re.sub(r"\s+", " ", chunk_text).strip()

            found_at = normalized_page.find(normalized_chunk, page_search_start)
            if found_at == -1:
                # fallback: reset search start and try again
                found_at = normalized_page.find(normalized_chunk)
                if found_at == -1:
                    # last resort: omit exact offsets
                    chunk_start_char = None
                    chunk_end_char = None
                else:
                    chunk_start_char = found_at
                    chunk_end_char = found_at + len(normalized_chunk)
            else:
                chunk_start_char = found_at
                chunk_end_char = found_at + len(normalized_chunk)

            # Advance search start to keep subsequent matches stable
            if found_at != -1:
                page_search_start = found_at + max(1, len(normalized_chunk) // 3)

            chunk_id = stable_chunk_id(DOCUMENT_ID, page_number, idx, chunk_text)

            chunk_text = fix_mojibake(chunk_text)


            chunks.append({
                "document_id": DOCUMENT_ID,
                "chunk_id": chunk_id,
                "text": chunk_text.strip(),
                "page_number": page_number,
                "source_file": os.path.basename(pdf_path),
                "chunk_start_char": chunk_start_char,
                "chunk_end_char": chunk_end_char,
                "chunking_strategy": CHUNKING_STRATEGY,
                "created_at": created_at
            })

        global_char_cursor += len(page_text)

    return chunks

def write_jsonl(chunks, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for obj in chunks:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def write_csv(chunks, path: str):
    # CSV-friendly column order
    fieldnames = [
        "document_id", "chunk_id", "page_number", "source_file",
        "chunk_start_char", "chunk_end_char",
        "chunking_strategy", "created_at", "text"
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for obj in chunks:
            w.writerow({k: obj.get(k) for k in fieldnames})

if __name__ == "__main__":
    chunks = build_chunks(SOURCE_FILE)
    write_jsonl(chunks, OUT_JSONL)
    write_csv(chunks, OUT_CSV)
    print(f"Wrote {len(chunks)} chunks to {OUT_JSONL} and {OUT_CSV}")
