import os
import requests
import yaml
import logging
import json
import re
import math
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

from fastapi import FastAPI, Request, Depends, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from enum import Enum
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import SparseVector
from pathlib import Path

# ---- Rate Limiting ----
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)

# Import shared utilities (embedding + context compression + LLM call)
from rag_utils import (
    EMBEDDING_SERVICE_TYPE, EMBEDDING_API_URL, LOCAL_EMBEDDING_MODEL,
    QDRANT_URL, QDRANT_API_KEY, COLLECTION,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    EMBED_TIMEOUT, TOP_K, MAX_CONTEXT_CHARS_PER_CHUNK, MAX_CONTEXT_TOTAL_CHARS,
    embed_query, compress_context, deepseek_answer
)

# ---- Logfire Setup ----
import logfire
LOGFIRE_TOKEN = os.environ.get("LOGFIRE_TOKEN")
if LOGFIRE_TOKEN:
    logfire.configure(token=LOGFIRE_TOKEN, send_to_logfire=True)
else:
    logfire.configure(send_to_logfire=False)

# ---- File Logging Setup ----
LOG_FILE = Path(__file__).with_name("rag_logs.jsonl")
logger = logging.getLogger("rag_server")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(file_handler)


def log_interaction(entry: Dict[str, Any]):
    entry["timestamp"] = datetime.utcnow().isoformat()
    logger.info(json.dumps(entry, ensure_ascii=False))
    # Also send to Logfire
    logfire.info("rag_query", **entry)


# Load system prompt from YAML
SYSTEM_PROMPT_PATH = Path(__file__).with_name("system_prompt.yaml")
DEFAULT_SYSTEM_PROMPT = (
    "Answer ONLY using the provided context. "
    "Be concise and specific: 1-2 sentences maximum. "
    "If not in context, say: 'Not found in the provided documents.'"
)


def load_system_prompt() -> str:
    try:
        with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict) and "content" in data:
                return data["content"].strip()
            if isinstance(data, str):
                return data.strip()
    except Exception:
        pass
    return DEFAULT_SYSTEM_PROMPT


SYSTEM_PROMPT = load_system_prompt()

# ---- Conversation Memory ----
MAX_HISTORY = 10  # Keep last 10 exchanges
conversation_history: List[Dict[str, str]] = []


def add_to_history(role: str, content: str):
    conversation_history.append({"role": role, "content": content})
    if len(conversation_history) > MAX_HISTORY * 2:
        conversation_history.pop(0)


def get_history_context() -> str:
    if not conversation_history:
        return ""
    lines = []
    for msg in conversation_history[-6:]:  # Last 3 exchanges
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def get_last_answer() -> str:
    """Get the last assistant answer for follow-up questions."""
    for msg in reversed(conversation_history):
        if msg["role"] == "assistant":
            return msg["content"]
    return ""


# ---- BM25 IDF build config (env-driven) ----
BM25_BUILD_IDF = os.environ.get("BM25_BUILD_IDF", "true").lower() == "true"
BM25_IDF_MAX_DOCS = int(os.environ.get("BM25_IDF_MAX_DOCS", "0"))  # 0 = all docs
BM25_IDF_BATCH_SIZE = int(os.environ.get("BM25_IDF_BATCH_SIZE", "256"))

# ---- Hybrid Retrieval Config (env-driven) ----
TOP_K_DENSE = int(os.environ.get("TOP_K_DENSE", "100"))
TOP_K_KEYWORD = int(os.environ.get("TOP_K_KEYWORD", "100"))
TOP_K_FINAL = int(os.environ.get("TOP_K_FINAL", "6"))
FUSION_TYPE = os.environ.get("FUSION_TYPE", "RRF").upper()  # RRF or SCORE_FUSION
RRF_K = int(os.environ.get("RRF_K", "60"))

# ---- Reranker Config ----
RERANKER_ENABLED = os.environ.get("RERANKER_ENABLED", "true").lower() == "true"
RERANKER_TOP_K = int(os.environ.get("RERANKER_TOP_K", "30"))
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "cross-encoder")

# ---- Rate Limiter Setup ----
RATE_LIMIT = os.environ.get("RATE_LIMIT", "10/minute")
limiter = Limiter(key_func=get_remote_address)

# ---- RBAC Setup ----
class Role(str, Enum):
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"


ROLE_PERMISSIONS = {
    Role.ADMIN: ["ask", "history", "clear", "admin"],
    Role.USER: ["ask", "history", "clear"],
    Role.READONLY: ["history"],
}


def load_api_keys() -> Dict[str, Dict[str, str]]:
    """Load API keys from environment variable."""
    keys_config = os.environ.get("API_KEYS", "")
    api_keys = {}
    if keys_config:
        for entry in keys_config.split(","):
            parts = entry.strip().split(":")
            if len(parts) == 3:
                name, key, role = parts
                api_keys[key] = {"name": name, "role": Role(role)}
    return api_keys


API_KEYS = load_api_keys()
RBAC_ENABLED = bool(API_KEYS)
security = HTTPBearer(auto_error=False)


def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict[str, Any]:
    """Verify API token and return user info."""
    if not RBAC_ENABLED:
        return {"name": "anonymous", "role": Role.ADMIN}

    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")

    token = credentials.credentials
    if token not in API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return API_KEYS[token]


def require_permission(permission: str):
    """Decorator to require specific permission."""
    def checker(user: Dict[str, Any] = Depends(verify_token)):
        role = user.get("role", Role.READONLY)
        allowed_permissions = ROLE_PERMISSIONS.get(role, [])
        if permission not in allowed_permissions:
            raise HTTPException(
                status_code=403,
                detail=f"Permission denied: '{permission}' required"
            )
        return user
    return checker


# ✅ FastAPI app
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS (demo UI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static UI
app.mount("/static", StaticFiles(directory="."), name="static")

# Qdrant client
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


class AskRequest(BaseModel):
    question: str
    clear_history: bool = False
    config: Optional[Dict[str, Any]] = None


# -----------------------------
# Logging helpers
# -----------------------------
def _safe_text(s: str, n: int = 200) -> str:
    s = (s or "").strip()
    return s[:n]


def _point_to_log_dict(p, max_text_chars: int = 200) -> Dict[str, Any]:
    payload = getattr(p, "payload", None) or {}
    return {
        "score": float(getattr(p, "score", 0.0)),
        "chunk_id": payload.get("chunk_id"),
        "page_number": payload.get("page_number"),
        "source_file": payload.get("source_file"),
        "text": _safe_text(payload.get("text", ""), max_text_chars),
    }


# ---- BM25 Sparse Vector Generation ----
class SimpleBM25:
    """Simple BM25 implementation for sparse vectors (hashed vocab + optional IDF built from Qdrant)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, vocab_size: int = 10000):
        self.k1 = k1
        self.b = b
        self.vocab_size = vocab_size

        self.doc_freq: Dict[int, int] = {}
        self.idf: Dict[int, float] = {}
        self.total_docs: int = 0

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z\u0600-\u06FF]+", (text or "").lower())

    def _hash_token(self, token: str) -> int:
        return hash(token) % self.vocab_size

    def build_index_from_collection(
        self,
        qdrant_client: QdrantClient,
        collection_name: str,
        max_docs: int = 0,
        batch_size: int = 256,
    ):
        """
        Build IDF index from existing documents in Qdrant.

        Compatible with qdrant-client versions where scroll() returns either:
          - an object with .points and .next_page_offset
          - OR a tuple: (points, next_page_offset)
        """
        print("[bm25] Building IDF index from Qdrant collection...")

        self.doc_freq = {}
        self.idf = {}
        self.total_docs = 0

        next_offset = None
        processed = 0

        while True:
            resp = qdrant_client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )

            if isinstance(resp, tuple):
                points, next_offset = resp
            else:
                points = getattr(resp, "points", None)
                next_offset = getattr(resp, "next_page_offset", None)

            points = points or []
            if not points:
                break

            for pt in points:
                payload = getattr(pt, "payload", None) or {}
                text = payload.get("text", "")
                tokens = set(self._tokenize(text))

                for tok in tokens:
                    idx = self._hash_token(tok)
                    self.doc_freq[idx] = self.doc_freq.get(idx, 0) + 1

                processed += 1
                if max_docs and processed >= max_docs:
                    break

            if processed % 500 == 0:
                print(f"[bm25] Processed {processed} docs...")

            if max_docs and processed >= max_docs:
                break

            if next_offset is None:
                break

        # Determine N
        if max_docs and max_docs > 0:
            self.total_docs = processed
        else:
            try:
                info = qdrant_client.get_collection(collection_name=collection_name)
                self.total_docs = int(getattr(info, "points_count", None) or processed)
            except Exception:
                self.total_docs = processed

        # Compute IDF
        print(f"[bm25] Computing IDF for {len(self.doc_freq)} terms over N={self.total_docs} docs...")
        for idx, df in self.doc_freq.items():
            idf_val = math.log((self.total_docs - df + 0.5) / (df + 0.5))
            self.idf[idx] = max(0.0, float(idf_val))

        print("[bm25] IDF index built successfully!")

    def compute_sparse_vector(self, text: str) -> Tuple[List[int], List[float]]:
        tokens = self._tokenize(text)
        if not tokens:
            return [], []

        term_counts: Dict[int, int] = {}
        for token in tokens:
            idx = self._hash_token(token)
            term_counts[idx] = term_counts.get(idx, 0) + 1

        doc_len = len(tokens)
        avg_doc_len = 100

        indices: List[int] = []
        values: List[float] = []

        for idx, freq in term_counts.items():
            tf = freq
            idf = self.idf.get(idx, 1.0)  # fallback if not built
            score = idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / avg_doc_len))
            indices.append(idx)
            values.append(float(score))

        return indices, values


bm25 = SimpleBM25()
print("[bm25] Simple BM25 initialized")


@app.on_event("startup")
def _startup_bm25_idf():
    """
    Build BM25 IDF index at server startup so BM25 uses real IDF instead of fallback idf=1.0.
    Controlled by env:
      BM25_BUILD_IDF=true/false
      BM25_IDF_MAX_DOCS=0 (0 = all docs)
      BM25_IDF_BATCH_SIZE=256
    """
    if not BM25_BUILD_IDF:
        print("[bm25] IDF indexing skipped (BM25_BUILD_IDF=false). Using idf=1.0 fallback.")
        return

    try:
        bm25.build_index_from_collection(
            qdrant_client=qdrant,
            collection_name=COLLECTION,
            max_docs=BM25_IDF_MAX_DOCS,
            batch_size=BM25_IDF_BATCH_SIZE,
        )
        print(f"[bm25] sanity: idf_terms={len(bm25.idf)} total_docs={bm25.total_docs}")
    except Exception as e:
        print(f"[bm25] WARNING: failed to build IDF index: {e}. Falling back to idf=1.0.")


# ---- Simple Cross-Encoder Reranker ----
class SimpleCrossEncoderReranker:
    """
    Lightweight reranker. Uses a real CrossEncoder if available, otherwise fallback scoring.
    """

    def __init__(self):
        self.use_transformer = False
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            self.use_transformer = True
            print("[reranker] Loaded cross-encoder model: ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            print(f"[reranker] Using fallback reranker (no transformer): {e}")
            self.model = None

    def _tokenize(self, text: str) -> set:
        return set(re.findall(r"\b[a-zA-Z]+\b", (text or "").lower()))

    def _fallback_score(self, query: str, text: str) -> float:
        query_tokens = self._tokenize(query)
        text_tokens = self._tokenize(text)

        if not query_tokens:
            return 0.0

        intersection = query_tokens & text_tokens
        union = query_tokens | text_tokens
        jaccard = len(intersection) / len(union) if union else 0.0

        text_lower = (text or "").lower()
        tf_score = sum(text_lower.count(term) for term in query_tokens) / max(len((text or "").split()), 1)

        exact_match = 1.0 if (query or "").lower() in text_lower else 0.0
        return 0.4 * jaccard + 0.4 * tf_score + 0.2 * exact_match

    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not documents:
            return documents

        if self.use_transformer:
            pairs = [[query, doc.get("text", "")] for doc in documents]
            scores = self.model.predict(pairs)
            for i, doc in enumerate(documents):
                doc["rerank_score"] = float(scores[i])
            return sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        else:
            scores = [self._fallback_score(query, doc.get("text", "")) for doc in documents]
            for i, doc in enumerate(documents):
                original_score = float(doc.get("score", 0.0))
                rerank_score = float(scores[i])
                doc["score"] = 0.3 * original_score + 0.7 * rerank_score
                doc["rerank_score"] = rerank_score
            return sorted(documents, key=lambda x: float(x.get("score", 0.0)), reverse=True)


reranker = SimpleCrossEncoderReranker()


def embed_sparse(text: str) -> SparseVector:
    indices, values = bm25.compute_sparse_vector(text)
    return SparseVector(indices=indices, values=values)


# -----------------------------
# Hybrid Retrieval
# Returns (final_items, retrieval_trace)
# -----------------------------
def retrieve_hybrid(
    query_text: str,
    qvec: List[float],
    reranker_enabled_override: Optional[bool] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from qdrant_client.http.models import Prefetch, FusionQuery, Fusion

    sparse_vector = embed_sparse(query_text)
    reranker_enabled = reranker_enabled_override if reranker_enabled_override is not None else RERANKER_ENABLED

    trace: Dict[str, Any] = {
        "query": query_text,
        "config": {
            "TOP_K_DENSE": TOP_K_DENSE,
            "TOP_K_KEYWORD": TOP_K_KEYWORD,
            "TOP_K_FINAL": TOP_K_FINAL,
            "FUSION_TYPE": FUSION_TYPE,
            "RRF_K": RRF_K,
            "RERANKER_ENABLED": reranker_enabled,
            "RERANKER_TOP_K": RERANKER_TOP_K,
            "RERANKER_MODEL": RERANKER_MODEL,
        },
        "dense_hits": [],
        "keyword_hits": [],
        "fused_results_pre_rerank": [],
    }

    print(f"[hybrid] Config: dense={TOP_K_DENSE}, keyword={TOP_K_KEYWORD}, final={TOP_K_FINAL}, fusion={FUSION_TYPE}")

    if FUSION_TYPE == "RRF":
        fusion = Fusion.RRF
    elif FUSION_TYPE == "SCORE_FUSION":
        fusion = Fusion.SCORE_FUSION
    else:
        fusion = Fusion.RRF

    prefetch = [
        Prefetch(query=qvec, using="dense", limit=TOP_K_DENSE),
        Prefetch(query=sparse_vector, using="sparse", limit=TOP_K_KEYWORD),
    ]

    # Dense branch
    dense_response = qdrant.query_points(
        collection_name=COLLECTION,
        query=qvec,
        using="dense",
        limit=TOP_K_DENSE,
        with_payload=True,
    )
    trace["dense_hits"] = [_point_to_log_dict(p) for p in dense_response.points[:min(50, len(dense_response.points))]]
    print(f"[dense] Retrieved {len(dense_response.points)} dense vector hits")

    # Keyword branch
    keyword_response = qdrant.query_points(
        collection_name=COLLECTION,
        query=sparse_vector,
        using="sparse",
        limit=TOP_K_KEYWORD,
        with_payload=True,
    )
    trace["keyword_hits"] = [_point_to_log_dict(p) for p in keyword_response.points[:min(50, len(keyword_response.points))]]
    print(f"[keyword] Retrieved {len(keyword_response.points)} keyword/BM25 hits")

    # Fusion
    response = qdrant.query_points(
        collection_name=COLLECTION,
        prefetch=prefetch,
        query=FusionQuery(fusion=fusion),
        limit=RERANKER_TOP_K if reranker_enabled else TOP_K_FINAL,
        with_payload=True,
    )
    hits = response.points
    print(f"[hybrid] Fused into {len(hits)} results using {FUSION_TYPE} fusion")

    # Build candidates with FULL text for rerank + context
    out: List[Dict[str, Any]] = []
    for idx, h in enumerate(hits):
        payload = h.payload or {}
        full_text = (payload.get("text") or "").strip()
        out.append(
            {
                "rank": idx + 1,
                "score": float(h.score),
                "text": full_text,
                "chunk_id": payload.get("chunk_id"),
                "page_number": payload.get("page_number"),
                "source_file": payload.get("source_file"),
            }
        )

    # Capture fused results BEFORE reranking
    trace["fused_results_pre_rerank"] = [
        {
            "rank": d["rank"],
            "score": d["score"],
            "chunk_id": d["chunk_id"],
            "page_number": d["page_number"],
            "source_file": d["source_file"],
            "text": _safe_text(d.get("text", ""), 200),
        }
        for d in out[:min(50, len(out))]
    ]

    # Rerank
    if reranker_enabled and len(out) > 0:
        out = reranker.rerank(query_text, out)
        out = out[:TOP_K_FINAL]
    else:
        out = out[:TOP_K_FINAL]

    # Cleanup
    for item in out:
        item.pop("rank", None)
        item.pop("rerank_score", None)

    return out, trace


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return FileResponse("index.html")


@app.get("/api")
def api_info():
    return {
        "message": "RAG API is running",
        "rbac_enabled": RBAC_ENABLED,
        "endpoints": ["/ask", "/history", "/clear", "/admin/rbac"]
    }


@app.get("/admin/rbac")
@limiter.limit(RATE_LIMIT)
def get_rbac_status(request: Request, user: Dict[str, Any] = Depends(require_permission("admin"))):
    return {
        "rbac_enabled": RBAC_ENABLED,
        "configured_users": list(API_KEYS.keys()) if RBAC_ENABLED else [],
        "role_permissions": {k.value: v for k, v in ROLE_PERMISSIONS.items()},
        "current_user": user.get("name"),
        "current_role": user.get("role").value if user.get("role") else None
    }


@app.post("/ask")
@limiter.limit(RATE_LIMIT)
def ask(req: AskRequest, request: Request, user: Dict[str, Any] = Depends(require_permission("ask"))):
    if req.clear_history:
        conversation_history.clear()

    question = req.question
    history = get_history_context()

    reranker_enabled = req.config.get("reranker_enabled") if req.config else RERANKER_ENABLED
    qvec = embed_query(question)

    items, retrieval_trace = retrieve_hybrid(question, qvec, reranker_enabled_override=reranker_enabled)

    if not items:
        answer = "No relevant documents found."
        add_to_history("user", question)
        add_to_history("assistant", answer)

        log_interaction({
            "question": question,
            "answer": answer,
            "sources_count": 0,
            "sources": [],
            "history_used": bool(history),
            "retrieval_trace": retrieval_trace,
            "final_context": {"chunks_sent": [], "context_text": "", "context_total_chars": 0},
        })
        return {"answer": answer, "sources": []}

    context = compress_context(items)

    final_context = {
        "chunks_sent": [
            {
                "score": i.get("score"),
                "chunk_id": i.get("chunk_id"),
                "page_number": i.get("page_number"),
                "source_file": i.get("source_file"),
                "text": _safe_text(i.get("text", ""), MAX_CONTEXT_CHARS_PER_CHUNK),
            }
            for i in items
        ],
        "context_text": context,
        "context_total_chars": len(context),
    }

    previous_answer = get_last_answer()
    answer = deepseek_answer(question, context, history, previous_answer)

    add_to_history("user", question)
    add_to_history("assistant", answer)

    sources = [
        {
            "score": i["score"],
            "source_file": i["source_file"],
            "page_number": i["page_number"],
            "chunk_id": i["chunk_id"]
        }
        for i in items
    ]

    log_interaction({
        "question": question,
        "answer": answer,
        "sources_count": len(sources),
        "sources": sources,
        "history_used": bool(history),
        "retrieval_trace": retrieval_trace,
        "final_context": final_context,
    })

    return {"answer": answer, "sources": sources}


@app.get("/history")
@limiter.limit(RATE_LIMIT)
def get_history(request: Request, user: Dict[str, Any] = Depends(require_permission("history"))):
    return {"history": conversation_history, "user": user.get("name")}


@app.post("/clear")
@limiter.limit(RATE_LIMIT)
def clear_history(request: Request, user: Dict[str, Any] = Depends(require_permission("clear"))):
    conversation_history.clear()
    log_interaction({
        "action": "clear_history",
        "user": user.get("name"),
        "role": user.get("role").value if user.get("role") else None
    })
    return {"message": "Conversation history cleared", "user": user.get("name")}
