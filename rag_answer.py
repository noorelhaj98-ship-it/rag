import os
import time
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)

# ---- Config (env-driven) ----
EMBEDDING_SERVICE_TYPE = os.environ.get("EMBEDDING_SERVICE_TYPE", "aragemma").lower()
EMBEDDING_API_URL = os.environ.get("EMBEDDING_API_URL")

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None
COLLECTION = os.environ.get("QDRANT_COLLECTION", "realsoft_chunks")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

EMBED_TIMEOUT = 120

TOP_K = 6
MAX_CONTEXT_CHARS_PER_CHUNK = 450  # keep short
MAX_CONTEXT_TOTAL_CHARS = 2400     # keep short


def embed_query(text: str) -> List[float]:
    if EMBEDDING_SERVICE_TYPE != "aragemma":
        raise RuntimeError("Set EMBEDDING_SERVICE_TYPE=aragemma")
    if not EMBEDDING_API_URL:
        raise RuntimeError("Missing EMBEDDING_API_URL")

    payload = {"text": text}  # AraGemma expects single string, not list

    for attempt in range(6):
        try:
            r = requests.post(EMBEDDING_API_URL, json=payload, timeout=EMBED_TIMEOUT)
            if r.status_code >= 400:
                print("[embed] status:", r.status_code)
                print("[embed] body:", r.text[:2000])
            r.raise_for_status()
            data = r.json()

            if "embeddings" in data:
                return data["embeddings"][0]
            if "embedding" in data:
                return data["embedding"]
            if "data" in data and data["data"] and "embedding" in data["data"][0]:
                return data["data"][0]["embedding"]
            if "vector" in data:
                return data["vector"]

            raise RuntimeError(f"Unexpected embedding response: {data}")

        except Exception as e:
            wait = min(2 ** attempt, 10)
            print(f"[embed] error: {e} | retrying in {wait}s")
            time.sleep(wait)

    raise RuntimeError("Embedding failed after retries.")


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


def compress_context(items: List[Dict[str, Any]]) -> str:
    """
    Keep only a small snippet from each chunk, and cap total context size,
    so answers become short + targeted (and cheaper).
    """
    blocks = []
    total = 0

    for it in items:
        t = " ".join(it["text"].split())  # normalize whitespace
        snippet = t[:MAX_CONTEXT_CHARS_PER_CHUNK]

        block = (
            f"[source: {it.get('source_file')} | page: {it.get('page_number')} | chunk_id: {it.get('chunk_id')}]\n"
            f"{snippet}\n"
        )
        if total + len(block) > MAX_CONTEXT_TOTAL_CHARS:
            break

        blocks.append(block)
        total += len(block)

    return "\n".join(blocks)


def deepseek_answer(question: str, context: str) -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY in .env")

    url = f"{DEEPSEEK_BASE_URL}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    system = (
        "You are a precise assistant. Answer ONLY using the provided context. "
        "Be concise and specific: 1-2 sentences maximum. "
        "If the answer is not in the context, say: 'Not found in the provided documents.'"
    )

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
            },
        ],
        "temperature": 0.2,
        "max_tokens": 120,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code >= 400:
        print("[deepseek] status:", r.status_code)
        print("[deepseek] body:", r.text[:2000])
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()




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
