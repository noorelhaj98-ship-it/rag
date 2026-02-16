import os
import requests
import yaml
import logging
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple
from fastapi import FastAPI, Request, Depends, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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

# ---- Config (env-driven) ----
EMBEDDING_SERVICE_TYPE = os.environ.get("EMBEDDING_SERVICE_TYPE", "aragemma").lower()
EMBEDDING_API_URL = os.environ.get("EMBEDDING_API_URL")

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None
COLLECTION = os.environ.get("QDRANT_COLLECTION", "realsoft_chunks")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

TOP_K = 6
EMBED_TIMEOUT = 120
MAX_CONTEXT_CHARS_PER_CHUNK = 450
MAX_CONTEXT_TOTAL_CHARS = 2400

# ---- Hybrid Retrieval Config ----
# Configurable retrieval parameters (env-driven)
TOP_K_DENSE = int(os.environ.get("TOP_K_DENSE", "10"))        # Dense vector prefetch limit
TOP_K_KEYWORD = int(os.environ.get("TOP_K_KEYWORD", "10"))    # Keyword/BM25 prefetch limit
TOP_K_FINAL = int(os.environ.get("TOP_K_FINAL", "6"))         # Final results after fusion
FUSION_TYPE = os.environ.get("FUSION_TYPE", "RRF").upper()    # RRF or SCORE_FUSION
RRF_K = int(os.environ.get("RRF_K", "60"))                    # RRF ranking constant (default 60)

# ---- Reranker Config ----
RERANKER_ENABLED = os.environ.get("RERANKER_ENABLED", "true").lower() == "true"
RERANKER_TOP_K = int(os.environ.get("RERANKER_TOP_K", "20"))  # Candidates to rerank
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "cross-encoder")  # cross-encoder or colbert

# ---- Rate Limiter Setup ----
# Default: 10 requests per minute per IP, configurable via env
RATE_LIMIT = os.environ.get("RATE_LIMIT", "10/minute")

limiter = Limiter(key_func=get_remote_address)

# ---- RBAC Setup ----
class Role(str, Enum):
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"

# Role-based permissions
ROLE_PERMISSIONS = {
    Role.ADMIN: ["ask", "history", "clear", "admin"],
    Role.USER: ["ask", "history", "clear"],
    Role.READONLY: ["history"],
}

# API Keys configuration (loaded from env)
# Format: API_KEYS=user:key1:role1,user:key2:role2
# Example: API_KEYS=user:abc123:user,admin:xyz789:admin,readonly:readonly456:readonly
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

# If no API keys configured, allow all (development mode)
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

# ✅ MUST EXIST
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# allow local html file to call API (demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


class AskRequest(BaseModel):
    question: str
    clear_history: bool = False  # Option to reset conversation


def embed_query(text: str) -> List[float]:
    if EMBEDDING_SERVICE_TYPE != "aragemma":
        raise RuntimeError("Set EMBEDDING_SERVICE_TYPE=aragemma")
    if not EMBEDDING_API_URL:
        raise RuntimeError("Missing EMBEDDING_API_URL")

    payload = {"text": text}  # aragemma expects single text string
    r = requests.post(EMBEDDING_API_URL, json=payload, timeout=EMBED_TIMEOUT)
    r.raise_for_status()
    data = r.json()

    if "embedding" in data:
        return data["embedding"]
    if "embeddings" in data and len(data["embeddings"]) > 0:
        return data["embeddings"][0]
    if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
        return data["data"][0].get("embedding", data["data"][0])

    raise RuntimeError(f"Unexpected embedding response: {data}")


# ---- BM25 Sparse Vector Generation ----
class SimpleBM25:
    """Simple BM25 implementation for sparse vectors."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freq = {}
        self.idf = {}
        self.avg_doc_len = 0
        self.vocab_size = 10000  # Fixed vocabulary size for consistency
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    def _hash_token(self, token: str) -> int:
        """Hash token to fixed vocabulary space."""
        return hash(token) % self.vocab_size
    
    def compute_sparse_vector(self, text: str) -> Tuple[List[int], List[float]]:
        """Compute sparse vector as (indices, values) for BM25."""
        tokens = self._tokenize(text)
        if not tokens:
            return [], []
        
        # Count term frequencies
        term_counts = {}
        for token in tokens:
            idx = self._hash_token(token)
            term_counts[idx] = term_counts.get(idx, 0) + 1
        
        # Compute BM25-like scores (simplified)
        doc_len = len(tokens)
        indices = []
        values = []
        
        for idx, freq in term_counts.items():
            # Simplified BM25 score
            tf = freq
            # Use log-based IDF approximation
            idf = 1.0  # Simplified - in real BM25 this would be log((N - n + 0.5) / (n + 0.5))
            score = idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / 10))
            
            indices.append(idx)
            values.append(score)
        
        return indices, values


bm25 = SimpleBM25()


# ---- Simple Cross-Encoder Reranker ----
class SimpleCrossEncoderReranker:
    """
    Simple cross-encoder style reranker using keyword overlap and semantic signals.
    This is a lightweight implementation - for production, use a real cross-encoder model.
    """
    
    def __init__(self):
        self.use_transformer = False
        try:
            from sentence_transformers import CrossEncoder
            # Try to load a small cross-encoder model
            self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            self.use_transformer = True
            print("[reranker] Loaded cross-encoder model")
        except Exception as e:
            print(f"[reranker] Using fallback reranker (no transformer): {e}")
            self.model = None
    
    def _tokenize(self, text: str) -> set:
        """Simple tokenization for fallback scoring."""
        import re
        return set(re.findall(r'\b[a-zA-Z]+\b', text.lower()))
    
    def _fallback_score(self, query: str, text: str) -> float:
        """Fallback scoring based on keyword overlap and BM25-like signals."""
        query_tokens = self._tokenize(query)
        text_tokens = self._tokenize(text)
        
        if not query_tokens:
            return 0.0
        
        # Jaccard similarity
        intersection = query_tokens & text_tokens
        union = query_tokens | text_tokens
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Term frequency bonus
        text_lower = text.lower()
        tf_score = sum(text_lower.count(term) for term in query_tokens) / max(len(text.split()), 1)
        
        # Exact match bonus
        exact_match = 1.0 if query.lower() in text.lower() else 0.0
        
        # Combine scores
        return 0.4 * jaccard + 0.4 * tf_score + 0.2 * exact_match
    
    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank documents based on query relevance."""
        if not documents:
            return documents
        
        if self.use_transformer:
            # Use real cross-encoder
            pairs = [[query, doc["text"]] for doc in documents]
            scores = self.model.predict(pairs)
        else:
            # Use fallback scoring
            scores = [self._fallback_score(query, doc["text"]) for doc in documents]
        
        # Combine with original scores (weighted average)
        for i, doc in enumerate(documents):
            original_score = doc.get("score", 0.0)
            rerank_score = float(scores[i])
            # Weighted combination: 30% original, 70% rerank
            doc["score"] = 0.3 * original_score + 0.7 * rerank_score
            doc["rerank_score"] = rerank_score
        
        # Sort by new score
        return sorted(documents, key=lambda x: x["score"], reverse=True)


# Initialize reranker
reranker = SimpleCrossEncoderReranker()


def embed_sparse(text: str) -> SparseVector:
    """Generate sparse BM25 vector for text."""
    indices, values = bm25.compute_sparse_vector(text)
    return SparseVector(indices=indices, values=values)


def retrieve_hybrid(query_text: str, qvec: List[float]) -> List[Dict[str, Any]]:
    """Hybrid retrieval using both dense (embedding) and sparse (BM25) vectors.
    
    Configurable via environment variables:
    - TOP_K_DENSE: Number of dense vector candidates (default: 10)
    - TOP_K_KEYWORD: Number of keyword/BM25 candidates (default: 10)
    - TOP_K_FINAL: Final number of results after fusion (default: 6)
    - FUSION_TYPE: RRF or SCORE_FUSION (default: RRF)
    - RRF_K: RRF ranking constant (default: 60)
    """
    from qdrant_client.http.models import Prefetch, FusionQuery, Fusion
    
    # Generate sparse vector for BM25
    sparse_vector = embed_sparse(query_text)
    
    # Log retrieval configuration
    print(f"[hybrid] Config: dense={TOP_K_DENSE}, keyword={TOP_K_KEYWORD}, final={TOP_K_FINAL}, fusion={FUSION_TYPE}")
    
    # Use configured fusion type
    if FUSION_TYPE == "RRF":
        fusion = Fusion.RRF
    elif FUSION_TYPE == "SCORE_FUSION":
        fusion = Fusion.SCORE_FUSION
    else:
        fusion = Fusion.RRF  # Default fallback
    
    # Prefetch from both vector types with configurable limits
    prefetch = [
        Prefetch(
            query=qvec,
            using="dense",
            limit=TOP_K_DENSE,
        ),
        Prefetch(
            query=sparse_vector,
            using="sparse",
            limit=TOP_K_KEYWORD,
        ),
    ]
    
    # Fuse results using configured fusion method
    response = qdrant.query_points(
        collection_name=COLLECTION,
        prefetch=prefetch,
        query=FusionQuery(fusion=fusion),
        limit=RERANKER_TOP_K if RERANKER_ENABLED else TOP_K_FINAL,
        with_payload=True,
    )
    hits = response.points
    
    print(f"[hybrid] Retrieved {len(hits)} results using {FUSION_TYPE} fusion")
    
    # Convert to output format
    out = []
    for h in hits:
        payload = h.payload or {}
        out.append(
            {
                "score": float(h.score),
                "text": (payload.get("text") or "").strip(),
                "chunk_id": payload.get("chunk_id"),
                "page_number": payload.get("page_number"),
                "source_file": payload.get("source_file"),
            }
        )
    
    # Apply reranking if enabled
    if RERANKER_ENABLED and len(out) > 0:
        print(f"[reranker] Applying reranking to {len(out)} candidates...")
        out = reranker.rerank(query_text, out)
        # Take top-K after reranking
        out = out[:TOP_K_FINAL]
        print(f"[reranker] Reranked to top-{len(out)} results")
    
    return out


def retrieve(qvec: List[float]) -> List[Dict[str, Any]]:
    """Fallback to dense-only retrieval for backward compatibility."""
    response = qdrant.query_points(
        collection_name=COLLECTION,
        query=qvec,
        using="dense",
        limit=TOP_K,
        with_payload=True,
    )
    hits = response.points

    out = []
    for h in hits:
        payload = h.payload or {}
        out.append(
            {
                "score": float(h.score),
                "text": (payload.get("text") or "").strip(),
                "chunk_id": payload.get("chunk_id"),
                "page_number": payload.get("page_number"),
                "source_file": payload.get("source_file"),
            }
        )
    return out


def compress_context(items: List[Dict[str, Any]]) -> str:
    blocks = []
    total = 0
    for it in items:
        t = " ".join(it["text"].split())
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

    system = SYSTEM_PROMPT
    
    # Build conversation-aware prompt with previous answer for follow-up questions
    user_content_parts = []
    
    if history:
        user_content_parts.append(f"Previous conversation:\n{history}")
    
    if previous_answer:
        user_content_parts.append(f"My previous answer:\n{previous_answer}")
    
    user_content_parts.append(f"Document Context:\n{context}")
    user_content_parts.append(f"Current question: {question}\n\nAnswer:")
    
    user_content = "\n\n".join(user_content_parts)

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.2,
        "max_tokens": 200,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


@app.get("/")
def root():
    return {
        "message": "RAG API is running",
        "rbac_enabled": RBAC_ENABLED,
        "endpoints": ["/ask", "/history", "/clear", "/admin/rbac"]
    }


@app.get("/admin/rbac")
@limiter.limit(RATE_LIMIT)
def get_rbac_status(request: Request, user: Dict[str, Any] = Depends(require_permission("admin"))):
    """Admin endpoint to check RBAC configuration."""
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
    # Clear history if requested
    if req.clear_history:
        conversation_history.clear()
    
    question = req.question
    
    # Get conversation history context
    history = get_history_context()
    
    qvec = embed_query(question)
    # Use hybrid retrieval (dense + BM25 sparse)
    print(f"[hybrid] Searching for: {question[:50]}...")
    items = retrieve_hybrid(question, qvec)
    print(f"[hybrid] Found {len(items)} results using dense + BM25 RRF")
    
    if not items:
        answer = "No relevant documents found."
        add_to_history("user", question)
        add_to_history("assistant", answer)
        log_interaction({
            "question": question,
            "answer": answer,
            "sources_count": 0,
            "sources": [],
            "history_used": bool(history)
        })
        return {"answer": answer, "sources": []}
    
    context = compress_context(items)
    previous_answer = get_last_answer()
    answer = deepseek_answer(question, context, history, previous_answer)
    
    # Save to conversation history
    add_to_history("user", question)
    add_to_history("assistant", answer)
    
    sources = [{"score": i["score"], "source_file": i["source_file"], "page_number": i["page_number"], "chunk_id": i["chunk_id"]} for i in items]
    
    log_interaction({
        "question": question,
        "answer": answer,
        "sources_count": len(sources),
        "sources": sources,
        "history_used": bool(history)
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
