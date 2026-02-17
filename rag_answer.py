import os
import time
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from pathlib import Path

# Import shared utilities
from rag_utils import (
    EMBEDDING_SERVICE_TYPE, EMBEDDING_API_URL, LOCAL_EMBEDDING_MODEL,
    QDRANT_URL, QDRANT_API_KEY, COLLECTION,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    EMBED_TIMEOUT, TOP_K, MAX_CONTEXT_CHARS_PER_CHUNK, MAX_CONTEXT_TOTAL_CHARS,
    embed_query, compress_context, deepseek_answer
)

load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)

TOP_K = 6
MAX_CONTEXT_CHARS_PER_CHUNK = 450  # keep short
MAX_CONTEXT_TOTAL_CHARS = 2400     # keep short


# embed_query function is now imported from rag_utils


def retrieve(qvec: List[float]) -> List[Dict[str, Any]]:
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    response = qdrant.query_points(
        collection_name=COLLECTION,
        query=qvec,
        limit=TOP_K,
        with_payload=True,
    )

    results: List[Dict[str, Any]] = []
    for h in response.points:
        payload = h.payload or {}
        results.append(
            {
                "score": float(h.score),
                "text": (payload.get("text") or "").strip(),
                "chunk_id": payload.get("chunk_id"),
                "page_number": payload.get("page_number"),
                "source_file": payload.get("source_file"),
            }
        )
    return results


# compress_context and deepseek_answer functions are now imported from rag_utils


def main():
    import sys
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:]).strip()
    else:
        question = input("Ask a question: ").strip()

    if not question:
        print("No question provided. Example:")
        print('  python rag_answer.py "What services does RealSoft offer?"')
        return

    qvec = embed_query(question)
    hits = retrieve(qvec)

    if not hits:
        print("No results found.")
        return

    context = compress_context(hits)
    answer = deepseek_answer(question, context)

    print("\nAnswer:\n")
    print(answer)

    print("\nSources used (top hits):")
    for i, h in enumerate(hits[:3], start=1):
        print(f"{i}) score={h['score']:.4f} page={h['page_number']} chunk_id={h['chunk_id']} file={h['source_file']}")

if __name__ == "__main__":
    main()