"""
Shared utilities for the RAG system
Contains common functions used by both rag_answer.py and server.py
"""

import os
import time
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)

# ---- Config (env-driven) ----
EMBEDDING_SERVICE_TYPE = os.environ.get("EMBEDDING_SERVICE_TYPE", "local").lower()
EMBEDDING_API_URL = os.environ.get("EMBEDDING_API_URL")
LOCAL_EMBEDDING_MODEL = os.environ.get("LOCAL_EMBEDDING_MODEL", "intfloat/multilingual-e5-large")

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None
COLLECTION = os.environ.get("QDRANT_COLLECTION", "realsoft_chunks_hybrid")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

EMBED_TIMEOUT = 120
TOP_K = 6
MAX_CONTEXT_CHARS_PER_CHUNK = 450
MAX_CONTEXT_TOTAL_CHARS = 2400


def embed_query(text: str) -> List[float]:
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


def deepseek_answer(question: str, context: str, history: str = "", previous_answer: str = "") -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY")

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