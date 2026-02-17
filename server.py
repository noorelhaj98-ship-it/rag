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
from fastapi.responses import JSONResponse, FileResponse
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

# Import shared utilities
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

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None
COLLECTION = os.environ.get("QDRANT_COLLECTION", "realsoft_chunks_hybrid")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

TOP_K = 6
EMBED_TIMEOUT = 120
MAX_CONTEXT_CHARS_PER_CHUNK = 450
MAX_CONTEXT_TOTAL_CHARS = 2400

# ---- Hybrid Retrieval Config ----
# Configurable retrieval parameters (env-driven)
TOP_K_DENSE = int(os.environ.get("TOP_K_DENSE", "100"))       # Dense vector prefetch limit
TOP_K_KEYWORD = int(os.environ.get("TOP_K_KEYWORD", "100"))   # Keyword/BM25 prefetch limit
TOP_K_FINAL = int(os.environ.get("TOP_K_FINAL", "6"))         # Final results after fusion
FUSION_TYPE = os.environ.get("FUSION_TYPE", "RRF").upper()    # RRF or SCORE_FUSION
RRF_K = int(os.environ.get("RRF_K", "60"))                    # RRF ranking constant (default 60)

# ---- Reranker Config ----
RERANKER_ENABLED = os.environ.get("RERANKER_ENABLED", "true").lower() == "true"
RERANKER_TOP_K = int(os.environ.get("RERANKER_TOP_K", "30"))  # Candidates to rerank (practical limit for CPU)
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

# Serve static files (HTML UI)
app.mount("/static", StaticFiles(directory="."), name="static")

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


class AskRequest(BaseModel):
    question: str
    clear_history: bool = False  # Option to reset conversation
    config: Optional[Dict[str, Any]] = None  # Request-specific configuration


# embed_query function is now imported from rag_utils


# ---- BM25 Sparse Vector Generation ----
class SimpleBM25:
    """Simple BM25 implementation for sparse vectors."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freq = {}  # Term -> document frequency
        self.idf = {}       # Term -> IDF value
        self.total_docs = 0
        self.vocab_size = 10000  # Fixed vocabulary size for consistency
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Include both English and Arabic characters
        return re.findall(r'[a-zA-Z\u0600-\u06FF]+', text.lower())
    
    def _hash_token(self, token: str) -> int:
        """Hash token to fixed vocabulary space."""
        return hash(token) % self.vocab_size
    
    def build_index_from_collection(self):
        """Build document frequency index from the Qdrant collection."""
        print("[bm25] Building IDF index from collection...")
        
        qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        # Get all documents to build IDF index
        offset = 0
        batch_size = 100
        total_processed = 0
        
        while True:
            response = qdrant.scroll(
                collection_name=COLLECTION,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            points = response.points
            if not points:
                break
            
            # Process batch of documents
            for point in points:
                payload = point.payload or {}
                text = payload.get("text", "")
                tokens = set(self._tokenize(text))  # Use set to count unique terms per doc
                
                for token in tokens:
                    idx = self._hash_token(token)
                    self.doc_freq[idx] = self.doc_freq.get(idx, 0) + 1
            
            total_processed += len(points)
            print(f"[bm25] Processed {total_processed} documents for IDF index...")
            
            if len(points) < batch_size:
                break
            
            # Move to next batch
            offset = points[-1].id
            
        # Calculate total number of documents
        try:
            collection_info = qdrant.get_collection(collection_name=COLLECTION)
            self.total_docs = collection_info.points_count
        except:
            # Fallback to counted documents
            self.total_docs = total_processed
        
        # Calculate IDF values
        print(f"[bm25] Calculating IDF for {len(self.doc_freq)} unique terms across {self.total_docs} documents...")
        for idx, df in self.doc_freq.items():
            # Standard BM25 IDF formula: log((N - n + 0.5) / (n + 0.5))
            idf_value = math.log((self.total_docs - df + 0.5) / (df + 0.5))
            # Ensure positive IDF (handle edge cases)
            self.idf[idx] = max(0.0, idf_value)
        
        print(f"[bm25] IDF index built successfully!")
        
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
        
        # Compute BM25 scores
        doc_len = len(tokens)
        indices = []
        values = []
        
        for idx, freq in term_counts.items():
            tf = freq
            # Use precomputed IDF if available, otherwise default to 1.0
            idf = self.idf.get(idx, 1.0)
            
            # Standard BM25 formula
            # score = IDF * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * doc_len / avg_doc_len))
            # We'll use a reasonable average document length
            avg_doc_len = 100  # Typical average document length in our dataset
            score = idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / avg_doc_len))
            
            indices.append(idx)
            values.append(score)
        
        return indices, values


# Initialize BM25 (skip indexing for now to avoid server startup issues)
bm25 = SimpleBM25()
print("[bm25] Simple BM25 initialized (skipping full IDF index for now)")


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
            # Load real cross-encoder model
            self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            self.use_transformer = True
            print("[reranker] Loaded cross-encoder model: ms-marco-MiniLM-L-6-v2")
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
            # Use real cross-encoder - sort by cross-encoder score only
            pairs = [[query, doc["text"]] for doc in documents]
            scores = self.model.predict(pairs)
            
            # Store cross-encoder score and sort by it only
            for i, doc in enumerate(documents):
                doc["rerank_score"] = float(scores[i])
            
            # Sort by cross-encoder score only (descending - higher is better)
            return sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        else:
            # Use fallback scoring
            scores = [self._fallback_score(query, doc["text"]) for doc in documents]
            
            # Combine with original scores for fallback
            for i, doc in enumerate(documents):
                original_score = doc.get("score", 0.0)
                rerank_score = float(scores[i])
                doc["score"] = 0.3 * original_score + 0.7 * rerank_score
                doc["rerank_score"] = rerank_score
            
            return sorted(documents, key=lambda x: x["score"], reverse=True)


# Initialize reranker
reranker = SimpleCrossEncoderReranker()


def embed_sparse(text: str) -> SparseVector:
    """Generate sparse BM25 vector for text."""
    indices, values = bm25.compute_sparse_vector(text)
    return SparseVector(indices=indices, values=values)


def retrieve_hybrid(query_text: str, qvec: List[float], reranker_enabled_override: Optional[bool] = None) -> List[Dict[str, Any]]:
    """Hybrid retrieval using both dense (embedding) and sparse (BM25) vectors.
    
    Configurable via environment variables:
    - TOP_K_DENSE: Number of dense vector candidates (default: 10)
    - TOP_K_KEYWORD: Number of keyword/BM25 candidates (default: 10)
    - TOP_K_FINAL: Final number of results after fusion (default: 6)
    - FUSION_TYPE: RRF or SCORE_FUSION (default: RRF)
    - RRF_K: RRF ranking constant (default: 60)
    
    Args:
        query_text: The query text
        qvec: Query vector
        reranker_enabled_override: Optional override for reranker setting
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
    
    # Log individual branch results before fusion
    dense_response = qdrant.query_points(
        collection_name=COLLECTION,
        query=qvec,
        using="dense",
        limit=TOP_K_DENSE,
        with_payload=True,
    )
    print(f"[dense] Retrieved {len(dense_response.points)} dense vector hits")
    
    keyword_response = qdrant.query_points(
        collection_name=COLLECTION,
        query=sparse_vector,
        using="sparse",
        limit=TOP_K_KEYWORD,
        with_payload=True,
    )
    print(f"[keyword] Retrieved {len(keyword_response.points)} keyword/BM25 hits")
    
    # Use the override if provided, otherwise use the environment setting
    reranker_enabled = reranker_enabled_override if reranker_enabled_override is not None else RERANKER_ENABLED
    
    # Fuse results using configured fusion method
    response = qdrant.query_points(
        collection_name=COLLECTION,
        prefetch=prefetch,
        query=FusionQuery(fusion=fusion),
        limit=RERANKER_TOP_K if reranker_enabled else TOP_K_FINAL,
        with_payload=True,
    )
    hits = response.points
    
    print(f"[hybrid] Fused into {len(hits)} results using {FUSION_TYPE} fusion")
    
    # Convert to output format
    out = []
    for idx, h in enumerate(hits):
        payload = h.payload or {}
        out.append(
            {
                "rank": idx + 1,
                "score": float(h.score),
                "text": (payload.get("text") or "").strip()[:100] + "...",
                "chunk_id": payload.get("chunk_id"),
                "page_number": payload.get("page_number"),
                "source_file": payload.get("source_file"),
            }
        )
    
    # Log fused candidates (top 20)
    print(f"\n[DEBUG] FUSED CANDIDATES (top {min(20, len(out))}):")
    for item in out[:20]:
        print(f"  Rank {item['rank']:2d}: Page {item['page_number']:3d} | Score {item['score']:.4f} | {item['chunk_id'][:30]}...")
    
    # Apply reranking if enabled
    print(f"\n[DEBUG] RERANKER_ENABLED={reranker_enabled}, len(out)={len(out)}")
    if reranker_enabled and len(out) > 0:
        print(f"[reranker] Applying reranking to {len(out)} candidates...")
        out = reranker.rerank(query_text, out)
        
        # Log reranked results (top 10)
        print(f"\n[DEBUG] RERANKED RESULTS (top {min(10, len(out))}):")
        for idx, item in enumerate(out[:10], 1):
            rerank_score = item.get('rerank_score', 'N/A')
            if isinstance(rerank_score, float):
                print(f"  New Rank {idx:2d}: Page {item['page_number']:3d} | Combined Score {item['score']:.4f} | Rerank Score {rerank_score:.4f} | {item['chunk_id'][:30]}...")
            else:
                print(f"  New Rank {idx:2d}: Page {item['page_number']:3d} | Combined Score {item['score']:.4f} | {item['chunk_id'][:30]}...")
        
        # Take top-K after reranking
        out = out[:TOP_K_FINAL]
        print(f"\n[reranker] Reranked to top-{len(out)} results")
    
    # Clean up text for return
    for item in out:
        del item['rank']
        if 'rerank_score' in item:
            del item['rerank_score']
    
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


# compress_context function is now imported from rag_utils


# deepseek_answer function is now imported from rag_utils


@app.get("/")
def root():
    # Serve the HTML UI
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
    
    # Use request-specific config to override environment settings
    reranker_enabled = req.config.get("reranker_enabled") if req.config else RERANKER_ENABLED
    print(f"[config] Reranker enabled: {reranker_enabled} (request override: {req.config is not None})")
    
    qvec = embed_query(question)
    # Use hybrid retrieval (dense + BM25 sparse)
    print(f"[hybrid] Searching for: {question[:50]}...")
    items = retrieve_hybrid(question, qvec, reranker_enabled_override=reranker_enabled)
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
    
    # Log final context being sent to LLM
    print(f"\n[final context] Using {len(items)} chunks, total context length: {len(context)} chars")
    print(f"[final context] Pages used: {[i['page_number'] for i in items]}")
    
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
