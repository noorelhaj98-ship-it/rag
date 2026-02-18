import os
import time
import requests
from typing import List
from dotenv import load_dotenv
from qdrant_client import QdrantClient

from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)

# ---- Config (env-driven) ----
EMBEDDING_SERVICE_TYPE = os.environ.get("EMBEDDING_SERVICE_TYPE", "aragemma").lower()
EMBEDDING_API_URL = os.environ.get("EMBEDDING_API_URL")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None
COLLECTION = os.environ.get("QDRANT_COLLECTION", "realsoft_chunks_hybrid")

EMBED_TIMEOUT = 120


def embed_query(text: str) -> List[float]:
    if EMBEDDING_SERVICE_TYPE != "aragemma":
        raise RuntimeError("Set EMBEDDING_SERVICE_TYPE=aragemma")

    if not EMBEDDING_API_URL:
        raise RuntimeError("Missing EMBEDDING_API_URL")

    # Aragemma expects: {"text": "single string"}
    payload = {"text": text}

    for attempt in range(6):
        try:
            r = requests.post(EMBEDDING_API_URL, json=payload, timeout=EMBED_TIMEOUT)
            if r.status_code == 422:
                print("[embed] 422:", r.text[:2000])
            r.raise_for_status()
            data = r.json()

            # Support common response shapes
            if "embedding" in data:
                return data["embedding"]

            if "embeddings" in data:
                return data["embeddings"][0]

            if "data" in data and isinstance(data["data"], list) and data["data"] and "embedding" in data["data"][0]:
                return data["data"][0]["embedding"]

            raise RuntimeError(f"Unexpected embedding response: {data}")

        except Exception as e:
            wait = min(2 ** attempt, 30)
            print(f"[embed] error: {e} | retrying in {wait}s")
            time.sleep(wait)

    raise RuntimeError("Embedding failed after retries.")


def main():
    query = input("Enter your search query: ").strip()

    qvec = embed_query(query)

    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    results = qdrant.query_points(
        collection_name=COLLECTION,
        query=qvec,
        limit=5,
        with_payload=True,
    ).points

    print("\nTop Matches:\n")

    for i, r in enumerate(results, start=1):
        print(f"Result {i} | score={r.score}")
        print(r.payload["text"])
        print("-" * 80)


    print("\nTop results:\n")
    for i, r in enumerate(results, start=1):
        payload = r.payload or {}
        text = (payload.get("text") or "").strip()
        print(f"{i}) score={r.score:.4f} chunk_id={payload.get('chunk_id')} source={payload.get('source_file')}")
        print(text[:600])
        print("-" * 80)


if __name__ == "__main__":
    main()
