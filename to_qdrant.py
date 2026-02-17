import os
import json
import time
import uuid
import re
from typing import List, Dict, Any, Tuple

import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)


# ---- Config (env-driven) ----
EMBEDDING_SERVICE_TYPE = os.environ.get("EMBEDDING_SERVICE_TYPE", "local").lower()
EMBEDDING_API_URL = os.environ.get("EMBEDDING_API_URL")  # required if aragemma
LOCAL_EMBEDDING_MODEL = os.environ.get("LOCAL_EMBEDDING_MODEL", "intfloat/multilingual-e5-large")

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None

# Keep your original COLLECTION var (not used for hybrid upsert anymore in this step)
COLLECTION = os.environ.get("QDRANT_COLLECTION", "realsoft_chunks")

# Hybrid collection name (you can set QDRANT_COLLECTION in .env to this too)
HYBRID_COLLECTION = os.environ.get("QDRANT_COLLECTION", "realsoft_chunks_hybrid")

INPUT_JSONL = "chunks.jsonl"

BATCH_SIZE = 64
UPSERT_BATCH_SIZE = 256
EMBED_TIMEOUT = 120  # seconds


def load_chunks(path: str) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = (obj.get("text") or "").strip()
            if text:
                chunks.append(obj)
    return chunks


def embed_single(text: str) -> List[float]:
    """Get embedding for a single text using either Aragemma API or local model."""
    if EMBEDDING_SERVICE_TYPE == "aragemma":
        if not EMBEDDING_API_URL:
            raise RuntimeError("Missing EMBEDDING_API_URL for aragemma embedding service.")
    
        payload = {"text": text}
        r = requests.post(EMBEDDING_API_URL, json=payload, timeout=EMBED_TIMEOUT)
    
        if r.status_code == 422:
            print(f"[embed/aragemma] 422 response: {r.text[:1000]}")
    
        r.raise_for_status()
        data = r.json()
    
        # Handle different response schemas
        if "embedding" in data:
            return data["embedding"]
        if "embeddings" in data and len(data["embeddings"]) > 0:
            return data["embeddings"][0]
        if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
            return data["data"][0].get("embedding", data["data"][0])
    
        raise RuntimeError(f"Unexpected embedding response schema: {data}")
    elif EMBEDDING_SERVICE_TYPE == "local":
        # Use local embedding model
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
            # For multilingual-e5-large, we need to add a prefix
            prefixed_text = f"query: {text}"
            embedding = model.encode([prefixed_text])[0].tolist()
            return embedding
        except ImportError:
            raise RuntimeError("Install sentence-transformers: pip install sentence-transformers")
        except Exception as e:
            raise RuntimeError(f"Local embedding failed: {e}")
    else:
        raise RuntimeError(
            f"Unsupported EMBEDDING_SERVICE_TYPE='{EMBEDDING_SERVICE_TYPE}'. "
            "Supported types: 'aragemma', 'local'."
        )


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Returns embeddings for a list of texts using either Aragemma API or local model.
    """
    embeddings = []
    for i, text in enumerate(texts):
        for attempt in range(6):
            try:
                embedding = embed_single(text)
                embeddings.append(embedding)

                if (i + 1) % 10 == 0:
                    if EMBEDDING_SERVICE_TYPE == "aragemma":
                        print(f"[embed/aragemma] processed {i + 1}/{len(texts)}")
                    else:
                        print(f"[embed/local] processed {i + 1}/{len(texts)}")

                break

            except Exception as e:
                wait = min(2 ** attempt, 30)
                if EMBEDDING_SERVICE_TYPE == "aragemma":
                    print(f"[embed/aragemma] error on text {i}: {e} | retrying in {wait}s")
                else:
                    print(f"[embed/local] error on text {i}: {e} | retrying in {wait}s")
                time.sleep(wait)

        else:
            raise RuntimeError(f"Failed to embed text after retries: {text[:100]}...")

    return embeddings


def ensure_hybrid_collection(qdrant: QdrantClient, dense_dim: int) -> None:
    existing = [c.name for c in qdrant.get_collections().collections]
    if HYBRID_COLLECTION in existing:
        return

    qdrant.create_collection(
        collection_name=HYBRID_COLLECTION,
        vectors_config={
            "dense": qmodels.VectorParams(
                size=dense_dim,
                distance=qmodels.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "sparse": qmodels.SparseVectorParams(
                modifier=qmodels.Modifier.IDF
            ),
        },
    )
    print(f"[qdrant] created hybrid collection '{HYBRID_COLLECTION}' dense_dim={dense_dim}")


# ---- BM25 Sparse Vector Generation ----
class SimpleBM25:
    """Simple BM25 implementation for sparse vectors."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vocab_size = 10000
        
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    def _hash_token(self, token: str) -> int:
        return hash(token) % self.vocab_size
    
    def compute_sparse_vector(self, text: str) -> Tuple[List[int], List[float]]:
        tokens = self._tokenize(text)
        if not tokens:
            return [], []
        
        term_counts = {}
        for token in tokens:
            idx = self._hash_token(token)
            term_counts[idx] = term_counts.get(idx, 0) + 1
        
        doc_len = len(tokens)
        indices = []
        values = []
        
        for idx, freq in term_counts.items():
            tf = freq
            idf = 1.0
            score = idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / 10))
            indices.append(idx)
            values.append(score)
        
        return indices, values


bm25 = SimpleBM25()


def chunk_to_point(chunk: Dict[str, Any], vector: List[float]) -> qmodels.PointStruct:
    stable_id = uuid.uuid5(uuid.NAMESPACE_URL, chunk["chunk_id"])
    text = chunk.get("text", "")
    
    # Generate sparse BM25 vector
    sparse_indices, sparse_values = bm25.compute_sparse_vector(text)

    payload = {
        "document_id": chunk.get("document_id"),
        "chunk_id": chunk.get("chunk_id"),
        "page_number": chunk.get("page_number"),
        "source_file": chunk.get("source_file"),
        "chunk_start_char": chunk.get("chunk_start_char"),
        "chunk_end_char": chunk.get("chunk_end_char"),
        "chunking_strategy": chunk.get("chunking_strategy"),
        "created_at": chunk.get("created_at"),
        "text": text,
    }

    return qmodels.PointStruct(
        id=str(stable_id),
        vector={
            "dense": vector,
            "sparse": qmodels.SparseVector(indices=sparse_indices, values=sparse_values),
        },
        payload=payload,
    )


def batched(lst: List[Any], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def estimate_avg_len(chunks: List[Dict[str, Any]]) -> float:
    lengths = [len((c.get("text") or "").split()) for c in chunks if (c.get("text") or "").strip()]
    return (sum(lengths) / max(len(lengths), 1)) or 1.0


def main() -> None:
    print("DEBUG ENV TYPE =", EMBEDDING_SERVICE_TYPE)

    if EMBEDDING_SERVICE_TYPE == "aragemma" and not EMBEDDING_API_URL:
        raise RuntimeError("Missing EMBEDDING_API_URL env var (required for aragemma).")
    elif EMBEDDING_SERVICE_TYPE not in ["aragemma", "local"]:
        raise RuntimeError(
            f"Unsupported EMBEDDING_SERVICE_TYPE='{EMBEDDING_SERVICE_TYPE}'. "
            "Supported types: 'aragemma', 'local'."
        )

    print("[cfg] EMBEDDING_SERVICE_TYPE =", EMBEDDING_SERVICE_TYPE)
    print("[cfg] EMBEDDING_API_URL =", EMBEDDING_API_URL)
    print("[cfg] QDRANT_URL =", QDRANT_URL)
    print("[cfg] COLLECTION =", HYBRID_COLLECTION)

    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    chunks = load_chunks(INPUT_JSONL)
    print(f"[load] chunks: {len(chunks)}")

    # Embed first batch to learn vector dim
    first_texts = [c["text"] for c in chunks[: min(BATCH_SIZE, len(chunks))]]
    first_vectors = embed_texts(first_texts)
    dim = len(first_vectors[0])
    print(f"[embed] dim={dim}")

    # ✅ Step 1: create hybrid collection
    ensure_hybrid_collection(qdrant, dim)

    # Upsert all chunks (still dense-only in Step 1)
    total = 0
    for chunk_batch in batched(chunks, BATCH_SIZE):
        texts = [c["text"] for c in chunk_batch]
        vectors = embed_texts(texts)

        points_batch = [chunk_to_point(chunk_batch[i], vectors[i]) for i in range(len(chunk_batch))]

        for p_sub in batched(points_batch, UPSERT_BATCH_SIZE):
            # ✅ Step 1: upsert into hybrid collection
            qdrant.upsert(collection_name=HYBRID_COLLECTION, points=p_sub)

        total += len(points_batch)
        print(f"[upsert] total={total}")

    print("[done] ingestion complete")


if __name__ == "__main__":
    main()

